"""Proxy-label fusion utilities for multi-evidence T2* vessel segmentation.

This module fuses three soft evidence maps into a conservative binary proxy label
that can be used for adaptation / booster-style training.

Evidence channels:
- E1: pretrained model probability
- E2: geometric prior (e.g., vesselness / tubularity)
- E3: continuity prior

Example:
    >>> result = fuse_proxy_labels(E1, E2, E3, image=t2star_img, mask=brain_mask)
    >>> result.proxy_full.shape
    (D, H, W)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage as scind

ArrayLike = np.ndarray


@dataclass(frozen=True)
class ProxyFusionConfig:
    """Configuration for multi-evidence proxy fusion."""

    w1: float = 0.45
    w2: float = 0.30
    w3: float = 0.25

    alpha: float = 0.4
    penalty_geometry_floor: float = 0.25
    penalty_continuity_floor: float = 0.30

    seed_threshold: float = 0.65
    candidate_threshold: float = 0.35
    candidate_geometry_floor: float = 0.25
    candidate_continuity_floor: float = 0.30

    grow_connectivity: int = 1
    grow_max_iters: Optional[int] = None

    min_component_size: int = 6
    cleanup_connectivity: int = 1

    keep_debug_maps: bool = True


@dataclass
class ProxyFusionResult:
    """Outputs from proxy-label fusion."""

    proxy_seed: np.ndarray
    proxy_full: np.ndarray
    score_map: np.ndarray
    penalty_map: Optional[np.ndarray] = None
    candidate_map: Optional[np.ndarray] = None
    consensus_map: Optional[np.ndarray] = None
    normalized_e1: Optional[np.ndarray] = None
    normalized_e2: Optional[np.ndarray] = None
    normalized_e3: Optional[np.ndarray] = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return non-empty outputs as a plain dictionary."""
        out: dict[str, np.ndarray] = {
            "proxy_seed": self.proxy_seed,
            "proxy_full": self.proxy_full,
            "score_map": self.score_map,
        }
        optional_maps = {
            "penalty_map": self.penalty_map,
            "candidate_map": self.candidate_map,
            "consensus_map": self.consensus_map,
            "normalized_e1": self.normalized_e1,
            "normalized_e2": self.normalized_e2,
            "normalized_e3": self.normalized_e3,
        }
        for key, value in optional_maps.items():
            if value is not None:
                out[key] = value
        return out


def normalize_evidence_map(evidence_map: ArrayLike, mask: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize an evidence map to [0, 1] inside mask."""
    evidence = np.asarray(evidence_map, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0

    out = np.zeros_like(evidence, dtype=np.float32)
    if not np.any(mask_bool):
        return out

    roi = evidence[mask_bool]
    roi = np.nan_to_num(roi, nan=0.0, posinf=0.0, neginf=0.0)

    v_min = float(np.min(roi))
    v_max = float(np.max(roi))
    if (v_max - v_min) <= eps:
        return out

    out[mask_bool] = (roi - v_min) / (v_max - v_min)
    out[~mask_bool] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_penalty_map(
    E2: ArrayLike,
    E3: ArrayLike,
    mask: ArrayLike,
    geometry_floor: float = 0.25,
    continuity_floor: float = 0.30,
) -> np.ndarray:
    """Build an explicit implausibility penalty map.

    Simple first-pass rule:
    - outside-mask is always 0
    - inside mask, voxels are penalized when BOTH geometry and continuity support
      are low.
    """
    e2 = np.asarray(E2, dtype=np.float32)
    e3 = np.asarray(E3, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0

    low_support = (e2 < float(geometry_floor)) & (e3 < float(continuity_floor))
    penalty = np.zeros_like(e2, dtype=np.float32)
    penalty[mask_bool & low_support] = 1.0
    penalty[~mask_bool] = 0.0
    return penalty


def compute_consensus_score(
    E1: ArrayLike,
    E2: ArrayLike,
    E3: ArrayLike,
    mask: ArrayLike,
    weights: Sequence[float] = (0.45, 0.30, 0.25),
) -> np.ndarray:
    """Compute weighted consensus S = w1*E1 + w2*E2 + w3*E3 inside mask."""
    if len(weights) != 3:
        raise ValueError(f"Expected 3 weights for E1/E2/E3, got {len(weights)}")

    w1, w2, w3 = (float(weights[0]), float(weights[1]), float(weights[2]))
    e1 = np.asarray(E1, dtype=np.float32)
    e2 = np.asarray(E2, dtype=np.float32)
    e3 = np.asarray(E3, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0

    score = (w1 * e1) + (w2 * e2) + (w3 * e3)
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    score[~mask_bool] = 0.0
    return score


def split_seed_and_candidate(
    score_map: ArrayLike,
    E2: ArrayLike,
    E3: ArrayLike,
    mask: ArrayLike,
    seed_threshold: float = 0.65,
    candidate_threshold: float = 0.35,
    candidate_geometry_floor: float = 0.25,
    candidate_continuity_floor: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split penalized score into high-confidence seeds and plausible candidates."""
    score = np.asarray(score_map, dtype=np.float32)
    e2 = np.asarray(E2, dtype=np.float32)
    e3 = np.asarray(E3, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0

    seed = (score >= float(seed_threshold)) & mask_bool
    plausible = (e2 >= float(candidate_geometry_floor)) | (e3 >= float(candidate_continuity_floor))
    candidate = (score >= float(candidate_threshold)) & plausible & mask_bool

    return seed.astype(bool), candidate.astype(bool)


def grow_candidate_from_seed(
    proxy_seed: ArrayLike,
    candidate_map: ArrayLike,
    connectivity: int = 1,
    max_iters: Optional[int] = None,
) -> np.ndarray:
    """Recover candidate voxels reachable from seed via constrained neighborhood growth."""
    seed = np.asarray(proxy_seed) > 0
    candidate = np.asarray(candidate_map) > 0

    if not np.any(seed):
        return np.zeros_like(candidate, dtype=bool)

    grown = seed.copy()
    structure = scind.generate_binary_structure(rank=3, connectivity=int(connectivity))

    step = 0
    while True:
        dilated = scind.binary_dilation(grown, structure=structure)
        new_voxels = dilated & candidate & (~grown)
        if not np.any(new_voxels):
            break
        grown |= new_voxels
        step += 1
        if max_iters is not None and step >= int(max_iters):
            break

    return grown.astype(bool)


def cleanup_proxy(
    proxy_map: ArrayLike,
    mask: ArrayLike,
    min_component_size: int = 6,
    connectivity: int = 1,
) -> np.ndarray:
    """Remove tiny isolated components while preserving conservative vessel extent."""
    proxy_bool = np.asarray(proxy_map) > 0
    mask_bool = np.asarray(mask) > 0
    constrained = proxy_bool & mask_bool

    if min_component_size <= 1:
        return constrained.astype(bool)

    structure = scind.generate_binary_structure(rank=3, connectivity=int(connectivity))
    labels, n_labels = scind.label(constrained, structure=structure)
    if n_labels == 0:
        return np.zeros_like(constrained, dtype=bool)

    counts = np.bincount(labels.ravel())
    keep_ids = np.where(counts >= int(min_component_size))[0]
    keep_ids = keep_ids[keep_ids != 0]

    if keep_ids.size == 0:
        return np.zeros_like(constrained, dtype=bool)

    cleaned = np.isin(labels, keep_ids)
    cleaned[~mask_bool] = False
    return cleaned.astype(bool)


def fuse_proxy_labels(
    E1: ArrayLike,
    E2: ArrayLike,
    E3: ArrayLike,
    image: ArrayLike,
    mask: ArrayLike,
    config: Optional[Union[ProxyFusionConfig, Mapping[str, Any]]] = None,
) -> ProxyFusionResult:
    """Fuse soft evidence maps into seed and full proxy labels.

    Args:
        E1: Pretrained model probability map.
        E2: Geometry prior map (channel 2).
        E3: Continuity prior map (channel 3).
        image: Source 3D T2* image (kept for API parity and future extensions).
        mask: Binary ROI mask.
        config: Optional dataclass or dictionary overriding fusion defaults.

    Returns:
        ProxyFusionResult with:
        - proxy_seed: high-confidence binary seed map
        - proxy_full: fused binary proxy map after growth + cleanup
        - score_map: penalized soft consensus score
        - optional debug maps when keep_debug_maps=True
    """
    _ = image  # Explicitly retained for inspectable API, currently not used directly.

    e1 = np.asarray(E1, dtype=np.float32)
    e2 = np.asarray(E2, dtype=np.float32)
    e3 = np.asarray(E3, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0

    _validate_shapes(e1, e2, e3, mask_bool)

    cfg = _resolve_config(config)

    n1 = normalize_evidence_map(e1, mask_bool)
    n2 = normalize_evidence_map(e2, mask_bool)
    n3 = normalize_evidence_map(e3, mask_bool)

    consensus = compute_consensus_score(
        n1,
        n2,
        n3,
        mask_bool,
        weights=(cfg.w1, cfg.w2, cfg.w3),
    )

    penalty_map = build_penalty_map(
        n2,
        n3,
        mask_bool,
        geometry_floor=cfg.penalty_geometry_floor,
        continuity_floor=cfg.penalty_continuity_floor,
    )

    damp = np.clip(1.0 - (float(cfg.alpha) * penalty_map), 0.0, 1.0)
    score_map = (consensus * damp).astype(np.float32)
    score_map[~mask_bool] = 0.0

    seed_map, candidate_map = split_seed_and_candidate(
        score_map,
        n2,
        n3,
        mask_bool,
        seed_threshold=cfg.seed_threshold,
        candidate_threshold=cfg.candidate_threshold,
        candidate_geometry_floor=cfg.candidate_geometry_floor,
        candidate_continuity_floor=cfg.candidate_continuity_floor,
    )

    grown_map = grow_candidate_from_seed(
        seed_map,
        candidate_map,
        connectivity=cfg.grow_connectivity,
        max_iters=cfg.grow_max_iters,
    )

    full_proxy = cleanup_proxy(
        grown_map,
        mask_bool,
        min_component_size=cfg.min_component_size,
        connectivity=cfg.cleanup_connectivity,
    )

    if cfg.keep_debug_maps:
        return ProxyFusionResult(
            proxy_seed=seed_map.astype(np.uint8),
            proxy_full=full_proxy.astype(np.uint8),
            score_map=score_map.astype(np.float32),
            penalty_map=penalty_map.astype(np.float32),
            candidate_map=candidate_map.astype(np.uint8),
            consensus_map=consensus.astype(np.float32),
            normalized_e1=n1.astype(np.float32),
            normalized_e2=n2.astype(np.float32),
            normalized_e3=n3.astype(np.float32),
        )

    return ProxyFusionResult(
        proxy_seed=seed_map.astype(np.uint8),
        proxy_full=full_proxy.astype(np.uint8),
        score_map=score_map.astype(np.float32),
    )


def _validate_shapes(E1: np.ndarray, E2: np.ndarray, E3: np.ndarray, mask: np.ndarray) -> None:
    """Validate shape consistency and 3D input assumptions."""
    shape = E1.shape
    if E2.shape != shape or E3.shape != shape or mask.shape != shape:
        raise ValueError(
            "All inputs must have the same shape. "
            f"Got E1={E1.shape}, E2={E2.shape}, E3={E3.shape}, mask={mask.shape}."
        )
    if E1.ndim != 3:
        raise ValueError(f"Expected 3D inputs. Got ndim={E1.ndim}.")
    if not np.any(mask):
        raise ValueError("Mask is empty; cannot fuse proxy labels without ROI voxels.")


def _resolve_config(config: Optional[Union[ProxyFusionConfig, Mapping[str, Any]]]) -> ProxyFusionConfig:
    """Resolve config from dataclass/dict/None."""
    if config is None:
        return ProxyFusionConfig()
    if isinstance(config, ProxyFusionConfig):
        return config
    if isinstance(config, Mapping):
        valid_fields = set(ProxyFusionConfig.__dataclass_fields__.keys())
        invalid = sorted(set(config.keys()) - valid_fields)
        if invalid:
            raise ValueError(f"Unknown proxy fusion config keys: {invalid}")
        return ProxyFusionConfig(**dict(config))
    raise TypeError("config must be None, ProxyFusionConfig, or a mapping.")
