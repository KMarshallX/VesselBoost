"""Proxy-label fusion utilities for T2*-weighted multichannel VesselBoost.

This module keeps the initial weak-supervision proxy fusion logic standalone and
easy to ablate. It combines three evidence maps into a conservative proxy label
by using weighted soft fusion, explicit penalties, confidence splitting,
seed-guided candidate recovery, and light connected-component cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage as scind


ArrayLike = np.ndarray
PercentileRange = Tuple[float, float]


@dataclass(frozen=True)
class ProxyFusionResult:
    """Container for proxy-fusion outputs and lightweight diagnostics."""

    fused_score_map: np.ndarray
    penalized_score_map: np.ndarray
    penalty_map: np.ndarray
    seed_mask: np.ndarray
    candidate_mask: np.ndarray
    grown_candidate_mask: np.ndarray
    background_mask: np.ndarray
    final_proxy_mask: np.ndarray
    normalized_e1: np.ndarray
    normalized_e2: np.ndarray
    normalized_e3: np.ndarray
    diagnostics: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        """Return array outputs plus diagnostics in a save-friendly dictionary."""
        return {
            "fused_score_map": self.fused_score_map,
            "penalized_score_map": self.penalized_score_map,
            "penalty_map": self.penalty_map,
            "seed_mask": self.seed_mask,
            "candidate_mask": self.candidate_mask,
            "grown_candidate_mask": self.grown_candidate_mask,
            "background_mask": self.background_mask,
            "final_proxy_mask": self.final_proxy_mask,
            "normalized_e1": self.normalized_e1,
            "normalized_e2": self.normalized_e2,
            "normalized_e3": self.normalized_e3,
            "diagnostics": dict(self.diagnostics),
        }


def normalize_evidence_map(
    evidence_map: ArrayLike,
    mask: Optional[ArrayLike] = None,
    mode: str = "percentile",
    percentiles: PercentileRange = (1.0, 99.0),
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize one evidence map to ``[0, 1]`` with robust edge-case handling.

    Args:
        evidence_map: 3D array to normalize.
        mask: Optional ROI mask. When provided, normalization statistics are
            computed only inside the mask and output is zeroed outside the mask.
        mode: ``"percentile"`` for robust clipping plus min-max normalization,
            or ``"minmax"`` for direct min-max normalization.
        percentiles: Lower and upper percentiles used by robust mode.
        eps: Small constant guarding divide-by-zero.
    """
    evidence = _sanitize_array(evidence_map)
    mask_bool = _resolve_mask(mask, evidence.shape)
    roi = evidence[mask_bool] if mask_bool is not None else evidence.reshape(-1)

    output = np.zeros_like(evidence, dtype=np.float32)
    if roi.size == 0:
        return output

    mode = mode.lower()
    if mode not in {"percentile", "minmax"}:
        raise ValueError(f"Unsupported normalization mode '{mode}'.")

    values = roi.astype(np.float32, copy=False)
    if mode == "percentile":
        low_pct, high_pct = _validate_percentiles(percentiles)
        lower, upper = np.percentile(values, (low_pct, high_pct))
        if np.isclose(lower, upper, atol=eps):
            return output
        scaled_source = np.clip(evidence, lower, upper)
        denominator = float(upper - lower)
        scaled = (scaled_source - lower) / denominator
    else:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if np.isclose(lower, upper, atol=eps):
            return output
        denominator = float(upper - lower)
        scaled = (evidence - lower) / denominator

    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    if mask_bool is None:
        return scaled

    output[mask_bool] = scaled[mask_bool]
    return output


def build_penalty_map(
    E2: ArrayLike,
    E3: ArrayLike,
    mask: ArrayLike,
    artifact_prior: Optional[ArrayLike] = None,
    geometry_floor: float = 0.25,
    continuity_floor: float = 0.30,
    artifact_weight: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Build a transparent penalty map in ``[0, 1]``.

    The penalty combines:
    - outside-mask suppression
    - graded weak-support penalty when both geometry and continuity are weak
    - optional artifact-prior penalty
    """
    e2 = _sanitize_array(E2)
    e3 = _sanitize_array(E3)
    mask_bool = _require_mask(mask, e2.shape)
    _validate_shapes(e2, e3, mask_bool)

    geometry_floor = float(geometry_floor)
    continuity_floor = float(continuity_floor)
    artifact_weight = float(artifact_weight)
    if geometry_floor < 0.0 or continuity_floor < 0.0:
        raise ValueError("Penalty floors must be non-negative.")
    if artifact_weight < 0.0:
        raise ValueError("artifact_weight must be non-negative.")

    penalty = np.zeros_like(e2, dtype=np.float32)
    penalty[~mask_bool] = 1.0

    geometry_deficit = np.clip((geometry_floor - e2) / max(geometry_floor, eps), 0.0, 1.0)
    continuity_deficit = np.clip(
        (continuity_floor - e3) / max(continuity_floor, eps),
        0.0,
        1.0,
    )
    weak_joint_support = (e2 < geometry_floor) & (e3 < continuity_floor) & mask_bool
    penalty[weak_joint_support] = 0.5 * (
        geometry_deficit[weak_joint_support] + continuity_deficit[weak_joint_support]
    )

    if artifact_prior is not None:
        artifact_penalty = normalize_evidence_map(
            artifact_prior,
            mask=mask_bool,
            mode="percentile",
        )
        penalty[mask_bool] = np.maximum(
            penalty[mask_bool],
            np.clip(artifact_weight * artifact_penalty[mask_bool], 0.0, 1.0),
        )

    return np.clip(penalty, 0.0, 1.0).astype(np.float32)


def compute_consensus_score(
    E1: ArrayLike,
    E2: ArrayLike,
    E3: ArrayLike,
    weights: Sequence[float] = (0.45, 0.30, 0.25),
) -> np.ndarray:
    """Compute the weighted soft fusion score ``S = w1*E1 + w2*E2 + w3*E3``."""
    if len(weights) != 3:
        raise ValueError(f"Expected 3 weights, got {len(weights)}.")

    e1 = _sanitize_array(E1)
    e2 = _sanitize_array(E2)
    e3 = _sanitize_array(E3)
    _validate_shapes(e1, e2, e3)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Fusion weights must sum to a positive value.")

    normalized_weights = tuple(float(weight) / weight_sum for weight in weights)
    score = (
        normalized_weights[0] * e1
        + normalized_weights[1] * e2
        + normalized_weights[2] * e3
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def split_seed_and_candidate(
    score_map: ArrayLike,
    E2: ArrayLike,
    E3: ArrayLike,
    mask: ArrayLike,
    seed_threshold: float = 0.65,
    low_threshold: float = 0.35,
    geometry_floor: float = 0.25,
    continuity_floor: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the penalized score into seed, candidate, and background masks."""
    score = _sanitize_array(score_map)
    e2 = _sanitize_array(E2)
    e3 = _sanitize_array(E3)
    mask_bool = _require_mask(mask, score.shape)
    _validate_shapes(score, e2, e3, mask_bool)

    if not 0.0 <= low_threshold <= seed_threshold <= 1.0:
        raise ValueError("Expected thresholds to satisfy 0 <= low_threshold <= seed_threshold <= 1.")

    structural_support = (e2 >= float(geometry_floor)) | (e3 >= float(continuity_floor))
    candidate_mask = (score >= float(low_threshold)) & structural_support & mask_bool
    seed_mask = (score >= float(seed_threshold)) & candidate_mask
    background_mask = mask_bool & (~candidate_mask)

    return seed_mask.astype(bool), candidate_mask.astype(bool), background_mask.astype(bool)


def grow_candidate_from_seed(
    seed_mask: ArrayLike,
    candidate_mask: ArrayLike,
    connectivity: int = 6,
) -> np.ndarray:
    """Promote only candidate voxels that are connected to seed voxels."""
    seed = np.asarray(seed_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    _validate_shapes(seed, candidate)

    if not np.any(seed):
        return np.zeros_like(candidate, dtype=bool)

    structure = _connectivity_structure(connectivity)
    grown = scind.binary_propagation(seed, structure=structure, mask=candidate)
    return grown.astype(bool)


def cleanup_proxy(
    proxy_mask: ArrayLike,
    mask: ArrayLike,
    seed_mask: Optional[ArrayLike] = None,
    min_component_size: int = 6,
    connectivity: int = 6,
) -> np.ndarray:
    """Remove tiny isolated regions while protecting seed-supported components."""
    proxy = np.asarray(proxy_mask, dtype=bool)
    mask_bool = _require_mask(mask, proxy.shape)
    seeds = np.asarray(seed_mask, dtype=bool) if seed_mask is not None else np.zeros_like(proxy)
    _validate_shapes(proxy, mask_bool, seeds)

    constrained = proxy & mask_bool
    protected = seeds & constrained
    if min_component_size <= 1 or not np.any(constrained):
        return constrained.astype(bool)

    structure = _connectivity_structure(connectivity)
    labels, component_count = scind.label(constrained, structure=structure)
    if component_count == 0:
        return np.zeros_like(constrained, dtype=bool)

    counts = np.bincount(labels.ravel())
    keep_ids = np.where(counts >= int(min_component_size))[0]

    protected_ids = np.unique(labels[protected])
    keep_ids = np.union1d(keep_ids, protected_ids)
    keep_ids = keep_ids[keep_ids != 0]

    if keep_ids.size == 0:
        return protected.astype(bool)

    cleaned = np.isin(labels, keep_ids)
    cleaned[~mask_bool] = False
    return cleaned.astype(bool)


def build_proxy_diagnostics(
    seed_mask: ArrayLike,
    candidate_mask: ArrayLike,
    final_proxy_mask: ArrayLike,
    connectivity: int = 26,
) -> Dict[str, int]:
    """Summarize counts needed to inspect fusion behavior."""
    seed = np.asarray(seed_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    final_proxy = np.asarray(final_proxy_mask, dtype=bool)
    _validate_shapes(seed, candidate, final_proxy)

    return {
        "seed_voxel_count": int(np.count_nonzero(seed)),
        "candidate_voxel_count": int(np.count_nonzero(candidate)),
        "final_proxy_voxel_count": int(np.count_nonzero(final_proxy)),
        "seed_component_count": int(_count_components(seed, connectivity)),
        "candidate_component_count": int(_count_components(candidate, connectivity)),
        "final_proxy_component_count": int(_count_components(final_proxy, connectivity)),
    }


def fuse_proxy_labels(
    E1: ArrayLike,
    E2: ArrayLike,
    E3: ArrayLike,
    mask: ArrayLike,
    artifact_prior: Optional[ArrayLike] = None,
    normalize_inputs: bool = True,
    normalization_mode: str = "percentile",
    normalization_percentiles: PercentileRange = (1.0, 99.0),
    weights: Sequence[float] = (0.45, 0.30, 0.25),
    alpha: float = 0.4,
    penalty_geometry_floor: float = 0.25,
    penalty_continuity_floor: float = 0.30,
    seed_threshold: float = 0.65,
    low_threshold: float = 0.35,
    candidate_geometry_floor: float = 0.25,
    candidate_continuity_floor: float = 0.30,
    connectivity: int = 6,
    min_component_size: int = 6,
    cleanup_connectivity: Optional[int] = None,
) -> ProxyFusionResult:
    """Run the full single-round proxy-fusion pipeline."""
    e1 = _sanitize_array(E1)
    e2 = _sanitize_array(E2)
    e3 = _sanitize_array(E3)
    mask_bool = _require_mask(mask, e1.shape)
    _validate_shapes(e1, e2, e3, mask_bool)

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    if normalize_inputs:
        n1 = normalize_evidence_map(
            e1,
            mask=mask_bool,
            mode=normalization_mode,
            percentiles=normalization_percentiles,
        )
        n2 = normalize_evidence_map(
            e2,
            mask=mask_bool,
            mode=normalization_mode,
            percentiles=normalization_percentiles,
        )
        n3 = normalize_evidence_map(
            e3,
            mask=mask_bool,
            mode=normalization_mode,
            percentiles=normalization_percentiles,
        )
    else:
        n1 = np.clip(e1, 0.0, 1.0).astype(np.float32)
        n2 = np.clip(e2, 0.0, 1.0).astype(np.float32)
        n3 = np.clip(e3, 0.0, 1.0).astype(np.float32)
        n1[~mask_bool] = 0.0
        n2[~mask_bool] = 0.0
        n3[~mask_bool] = 0.0

    fused_score = compute_consensus_score(n1, n2, n3, weights=weights)
    fused_score[~mask_bool] = 0.0

    penalty_map = build_penalty_map(
        n2,
        n3,
        mask_bool,
        artifact_prior=artifact_prior,
        geometry_floor=penalty_geometry_floor,
        continuity_floor=penalty_continuity_floor,
    )

    dampening = np.clip(1.0 - (float(alpha) * penalty_map), 0.0, 1.0)
    penalized_score = np.clip(fused_score * dampening, 0.0, 1.0).astype(np.float32)
    penalized_score[~mask_bool] = 0.0

    seed_mask, candidate_mask, background_mask = split_seed_and_candidate(
        penalized_score,
        n2,
        n3,
        mask_bool,
        seed_threshold=seed_threshold,
        low_threshold=low_threshold,
        geometry_floor=candidate_geometry_floor,
        continuity_floor=candidate_continuity_floor,
    )

    grown_candidate_mask = grow_candidate_from_seed(
        seed_mask,
        candidate_mask,
        connectivity=connectivity,
    )

    final_proxy_mask = cleanup_proxy(
        grown_candidate_mask,
        mask_bool,
        seed_mask=seed_mask,
        min_component_size=min_component_size,
        connectivity=cleanup_connectivity or connectivity,
    )

    diagnostics = build_proxy_diagnostics(
        seed_mask,
        candidate_mask,
        final_proxy_mask,
        connectivity=cleanup_connectivity or connectivity,
    )

    return ProxyFusionResult(
        fused_score_map=fused_score.astype(np.float32),
        penalized_score_map=penalized_score.astype(np.float32),
        penalty_map=penalty_map.astype(np.float32),
        seed_mask=seed_mask.astype(np.uint8),
        candidate_mask=candidate_mask.astype(np.uint8),
        grown_candidate_mask=grown_candidate_mask.astype(np.uint8),
        background_mask=background_mask.astype(np.uint8),
        final_proxy_mask=final_proxy_mask.astype(np.uint8),
        normalized_e1=n1.astype(np.float32),
        normalized_e2=n2.astype(np.float32),
        normalized_e3=n3.astype(np.float32),
        diagnostics=diagnostics,
    )


def _sanitize_array(array: ArrayLike) -> np.ndarray:
    """Return a finite ``float32`` copy of an input array."""
    return np.nan_to_num(
        np.asarray(array, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _resolve_mask(mask: Optional[ArrayLike], expected_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Resolve an optional mask, validating shape when present."""
    if mask is None:
        return None
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != expected_shape:
        raise ValueError(
            f"Mask shape mismatch. Expected {expected_shape}, got {mask_bool.shape}."
        )
    if mask_bool.ndim != 3:
        raise ValueError(f"Expected 3D mask, got ndim={mask_bool.ndim}.")
    return mask_bool


def _require_mask(mask: ArrayLike, expected_shape: Tuple[int, ...]) -> np.ndarray:
    """Resolve a required mask and ensure it contains foreground voxels."""
    mask_bool = _resolve_mask(mask, expected_shape)
    if mask_bool is None:
        raise ValueError("A non-empty mask is required for proxy fusion.")
    if not np.any(mask_bool):
        raise ValueError("Mask is empty; provide a foreground ROI mask.")
    return mask_bool


def _validate_percentiles(percentiles: PercentileRange) -> PercentileRange:
    """Validate percentile configuration."""
    low, high = float(percentiles[0]), float(percentiles[1])
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(
            "percentiles must satisfy 0 <= low < high <= 100. "
            f"Received {percentiles}."
        )
    return low, high


def _validate_shapes(*arrays: np.ndarray) -> None:
    """Ensure all arrays share one 3D shape."""
    if not arrays:
        return
    reference_shape = arrays[0].shape
    reference_ndim = arrays[0].ndim
    if reference_ndim != 3:
        raise ValueError(f"Expected 3D inputs, got ndim={reference_ndim}.")
    for array in arrays[1:]:
        if array.shape != reference_shape:
            raise ValueError(
                f"Shape mismatch. Expected {reference_shape}, got {array.shape}."
            )
        if array.ndim != reference_ndim:
            raise ValueError(
                f"Dimension mismatch. Expected ndim={reference_ndim}, got {array.ndim}."
            )


def _connectivity_structure(connectivity: int) -> np.ndarray:
    """Map voxel connectivity ``{6, 18, 26}`` to a SciPy binary structure."""
    connectivity_map = {6: 1, 18: 2, 26: 3}
    if connectivity not in connectivity_map:
        raise ValueError("connectivity must be one of {6, 18, 26}.")
    return scind.generate_binary_structure(rank=3, connectivity=connectivity_map[connectivity])


def _count_components(mask: np.ndarray, connectivity: int) -> int:
    """Count connected components for a 3D boolean mask."""
    if not np.any(mask):
        return 0
    structure = _connectivity_structure(connectivity)
    _, component_count = scind.label(mask.astype(bool), structure=structure)
    return int(component_count)


__all__ = [
    "ProxyFusionResult",
    "normalize_evidence_map",
    "build_penalty_map",
    "compute_consensus_score",
    "split_seed_and_candidate",
    "grow_candidate_from_seed",
    "cleanup_proxy",
    "build_proxy_diagnostics",
    "fuse_proxy_labels",
]
