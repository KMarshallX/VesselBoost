"""Utilities for preparing three-channel T2* input volumes for VesselBoost.

This module keeps T2* preparation isolated from existing data loaders and model
code. The main entry point is :class:`ThreeChanDataPrepper`, a callable object
that converts one native 3D T2* volume and one matching ROI mask into a
three-channel 4D volume in channel-first format: ``[3, D, H, W]``.

Example:
    >>> prepper = ThreeChanDataPrepper()
    >>> volume_3ch = prepper(
    ...     image="/path/sub-01_T2star.nii.gz",
    ...     mask="/path/sub-01_mask.nii.gz",
    ...     save_path="/path/sub-01_T2star_3ch.nii.gz",
    ... )
    >>> volume_3ch.shape
    (3, D, H, W)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import nibabel as nib  # type: ignore
import numpy as np
import scipy.ndimage as scind
from skimage.filters import frangi


logger = logging.getLogger(__name__)

try:
    import ants  # type: ignore
except ImportError:
    ants = None


ArrayLike = np.ndarray
PathLike = Union[str, Path]
ImageInput = Union[PathLike, ArrayLike]
HeaderLike = Optional[nib.Nifti1Header]


class ThreeChanDataPrepper:
    """Prepare 3-channel T2* data for downstream PyTorch VesselBoost workflows.

    Channel order:
    1. Preprocessed intensity channel
    2. Vesselness / tubularity channel (Frangi-style)
    3. Continuity prior channel (directional minIP)
    """

    def __init__(
        self,
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        stretch_percentiles: Tuple[float, float] = (1.0, 99.0),
        normalization: str = "minmax",
        n4_spline_spacing_mm: float = 3.0,
        nlm_noise_model: str = "Rician",
        frangi_sigmas: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5),
        frangi_alpha: float = 0.5,
        frangi_beta: float = 0.5,
        frangi_gamma: Optional[float] = None,
        frangi_black_ridges: bool = True,
        slab_thickness: int = 9,
        skip_antspy_preprocessing: bool = False,
    ):
        self.clip_percentiles = clip_percentiles
        self.stretch_percentiles = stretch_percentiles
        self.normalization = normalization.lower()
        self.n4_spline_spacing_mm = n4_spline_spacing_mm
        self.nlm_noise_model = nlm_noise_model
        self.frangi_sigmas = tuple(frangi_sigmas)
        self.frangi_alpha = frangi_alpha
        self.frangi_beta = frangi_beta
        self.frangi_gamma = frangi_gamma
        self.frangi_black_ridges = frangi_black_ridges
        self.slab_thickness = int(slab_thickness)
        self.skip_antspy_preprocessing = bool(skip_antspy_preprocessing)

        self._validate_init_params()

    def __call__(
        self,
        image: ImageInput,
        mask: ImageInput,
        image_affine: Optional[np.ndarray] = None,
        image_header: HeaderLike = None,
        mask_affine: Optional[np.ndarray] = None,
        mask_header: HeaderLike = None,
        save_path: Optional[PathLike] = None,
    ) -> np.ndarray:
        """Run the three-channel preparation pipeline on one image/mask pair.

        Args:
            image: 3D image volume as path to NIfTI file or in-memory ndarray.
            mask: 3D foreground ROI mask as path to NIfTI file or ndarray.
            image_affine: Optional affine for ndarray image input.
            image_header: Optional NIfTI header for ndarray image input.
            mask_affine: Optional affine for ndarray mask input.
            mask_header: Optional NIfTI header for ndarray mask input.
            save_path: Optional output NIfTI path. If set, the prepared output is
                saved as 4D ``[D, H, W, C]`` for NIfTI compatibility.

        Returns:
            A ``float32`` array of shape ``[3, D, H, W]``.
        """
        image_data, affine, header = self._load_volume(
            image, fallback_affine=image_affine, fallback_header=image_header, is_mask=False
        )
        mask_data, mask_aff, _ = self._load_volume(
            mask, fallback_affine=mask_affine, fallback_header=mask_header, is_mask=True
        )

        self._validate_inputs(image_data, mask_data, affine, mask_aff)
        mask_bool = mask_data > 0

        native = self._replace_non_finite(image_data.astype(np.float32), "native image")
        if self.skip_antspy_preprocessing:
            logger.info(
                "Preprocessing: skipping ANTsPy N4/NLM; assuming input is already preprocessed."
            )
            denoised = native
        else:
            logger.info("Preprocessing: running ANTsPy N4 bias correction + NLM denoising.")
            corrected = self._n4_bias_correct(native, mask_bool)
            denoised = self._nlm_denoise(corrected, mask_bool)
            denoised = self._replace_non_finite(denoised, "post-denoising image")

        logger.info("Step 1/3: first channel preprocessing.")
        ch1 = self._build_intensity_channel(denoised, mask_bool)
        logger.info("Step 2/3: second channel vesselness computation.")
        ch2 = self._build_vesselness_channel(denoised, mask_bool)
        logger.info("Step 3/3: third channel continuity prior.")
        ch3 = self._build_continuity_channel(native, mask_bool)

        prepared = np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)
        prepared = self._replace_non_finite(prepared, "stacked output")
        prepared[:, ~mask_bool] = 0.0

        if save_path is not None:
            self.save_prepared_volume(prepared, save_path, affine=affine, header=header)

        return prepared

    def save_prepared_volume(
        self,
        volume: np.ndarray,
        save_path: PathLike,
        affine: Optional[np.ndarray],
        header: HeaderLike = None,
    ) -> None:
        """Save prepared channel-first output to a 4D NIfTI file.

        The in-memory convention is ``[C, D, H, W]``. For NIfTI storage, data is
        reordered to ``[D, H, W, C]``.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if volume.ndim != 4:
            raise ValueError(f"Expected 4D volume to save, got shape {volume.shape}")
        if volume.shape[0] != 3:
            raise ValueError(f"Expected first dimension to be 3 channels, got {volume.shape[0]}")

        if affine is None:
            affine = np.eye(4, dtype=np.float32)

        save_data = np.transpose(volume, (1, 2, 3, 0))
        save_header = header.copy() if header is not None else None
        nifti = nib.Nifti1Image(save_data.astype(np.float32), affine, save_header)  # type: ignore
        nib.save(nifti, str(save_path))  # type: ignore
        logger.info("Saved prepared volume to %s (stored as [D, H, W, C]).", save_path)

    def _build_intensity_channel(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Build channel-1 (preprocessed intensity channel)."""
        clipped = self._robust_clip(image, mask, self.clip_percentiles)
        inverted = self._invert_intensity(clipped, mask)
        stretched = self._percentile_stretch(inverted, mask, self.stretch_percentiles)
        normalized = self._normalize(stretched, mask, self.normalization)
        normalized = self._replace_non_finite(normalized, "intensity channel")
        normalized[~mask] = 0.0
        return normalized.astype(np.float32)

    def _build_vesselness_channel(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Build channel-2 (Frangi-style vesselness / tubularity channel)."""
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return np.zeros_like(image, dtype=np.float32)

        tight_bbox = self._mask_bounding_box(mask_bool)
        if tight_bbox is None:
            return np.zeros_like(image, dtype=np.float32)

        clean_image = self._replace_non_finite(image.astype(np.float32), "vesselness input")

        # Frangi uses Gaussian-derivative support; keep a conservative halo so
        # the tight-mask crop remains equivalent to full-volume evaluation.
        halo = int(np.ceil(4.0 * max(self.frangi_sigmas))) if self.frangi_sigmas else 0
        working_bbox = self._expand_bbox(tight_bbox, image.shape, halo)
        work_image = clean_image[working_bbox]
        work_mask = mask_bool[working_bbox]
        masked = np.where(work_mask, work_image, 0.0).astype(np.float32)

        # Conservative defaults:
        # - multi-scale sigmas up to 2.5 voxels preserve weaker/smaller tubes
        # - black_ridges=True assumes vessels appear darker on native T2*
        vesselness = frangi(
            masked,
            sigmas=self.frangi_sigmas,
            alpha=self.frangi_alpha,
            beta=self.frangi_beta,
            gamma=self.frangi_gamma,
            black_ridges=self.frangi_black_ridges,
        )
        vesselness = self._replace_non_finite(vesselness, "vesselness response (cropped)")
        vesselness = self._apply_mask(vesselness, work_mask)

        vesselness_full = np.zeros_like(image, dtype=np.float32)
        vesselness_full[working_bbox] = vesselness.astype(np.float32)
        vesselness_full = self._normalize(vesselness_full, mask_bool, "minmax")
        vesselness_full[~mask_bool] = 0.0
        return vesselness_full.astype(np.float32)

    def _build_continuity_channel(self, native_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Build channel-3 using directional minIP continuity prior on native T2*."""
        minip_x = self._directional_minip(native_image, axis=0)
        minip_y = self._directional_minip(native_image, axis=1)
        minip_z = self._directional_minip(native_image, axis=2)

        inv_x = self._invert_intensity(minip_x, mask)
        inv_y = self._invert_intensity(minip_y, mask)
        inv_z = self._invert_intensity(minip_z, mask)

        continuity = (inv_x + inv_y + inv_z) / 3.0
        continuity = self._replace_non_finite(continuity, "continuity channel")
        continuity = self._apply_mask(continuity, mask)
        continuity = self._normalize(continuity, mask, "minmax")
        continuity[~mask] = 0.0
        return continuity.astype(np.float32)

    def _directional_minip(self, volume: np.ndarray, axis: int) -> np.ndarray:
        """Compute 1D centered directional minIP along one axis."""
        return scind.minimum_filter1d(
            volume.astype(np.float32),
            size=self.slab_thickness,
            axis=axis,
            mode="nearest",
        )

    def _n4_bias_correct(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply N4 bias field correction with conservative, explicit defaults."""
        if ants is None:
            raise ImportError(
                "ANTsPy is required for N4 correction but is not available. "
                "Install antspyx (see requirements/environment files)."
            )

        ant_image = ants.from_numpy(image.astype(np.float32))
        ant_mask = ants.from_numpy(mask.astype(np.uint8))
        kwargs = {
            "image": ant_image,
            "mask": ant_mask,
            "spline_param": self.n4_spline_spacing_mm,
        }
        corrected = ants.n4_bias_field_correction(**kwargs)
        return corrected.numpy().astype(np.float32)

    def _nlm_denoise(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply ANTs non-local means denoising with a Rician noise model."""
        if ants is None:
            raise ImportError(
                "ANTsPy is required for denoising but is not available. "
                "Install antspyx (see requirements/environment files)."
            )

        ant_image = ants.from_numpy(image.astype(np.float32))
        ant_mask = ants.from_numpy(mask.astype(np.uint8))
        kwargs = {
            "image": ant_image,
            "mask": ant_mask,
            "noise_model": self.nlm_noise_model,
        }
        try:
            denoised = ants.denoise_image(**kwargs)
        except TypeError:
            logger.warning(
                "Installed ANTsPy did not accept explicit noise model '%s'; "
                "retrying with library defaults.",
                self.nlm_noise_model,
            )
            denoised = ants.denoise_image(image=ant_image, mask=ant_mask)
        return denoised.numpy().astype(np.float32)

    def _load_volume(
        self,
        data: ImageInput,
        fallback_affine: Optional[np.ndarray],
        fallback_header: HeaderLike,
        is_mask: bool,
    ) -> Tuple[np.ndarray, np.ndarray, HeaderLike]:
        """Load one 3D volume from NIfTI path or ndarray with optional metadata."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            nifti = nib.load(str(path))  # type: ignore
            array = nifti.get_fdata().astype(np.float32)  # type: ignore
            affine = np.array(nifti.affine, dtype=np.float32)  # type: ignore
            header = nifti.header.copy()
        elif isinstance(data, np.ndarray):
            array = np.asarray(data, dtype=np.float32)
            affine = np.array(fallback_affine, dtype=np.float32) if fallback_affine is not None else np.eye(4, dtype=np.float32)
            header = fallback_header.copy() if fallback_header is not None else None
        else:
            raise TypeError(
                "Unsupported input type. Expected file path or numpy.ndarray, "
                f"got {type(data)}."
            )

        if array.ndim != 3:
            raise ValueError(
                f"Expected 3D {'mask' if is_mask else 'image'} volume, got shape {array.shape}."
            )

        if is_mask:
            array = (array > 0).astype(np.float32)

        return array, affine, header # type: ignore

    def _validate_inputs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_affine: Optional[np.ndarray],
        mask_affine: Optional[np.ndarray],
    ) -> None:
        """Validate image-mask compatibility and basic input integrity."""
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: image {image.shape} vs mask {mask.shape}."
            )
        if image.ndim != 3 or mask.ndim != 3:
            raise ValueError(
                f"Only 3D inputs are supported. Got image ndim={image.ndim}, mask ndim={mask.ndim}."
            )
        if not np.any(mask > 0):
            raise ValueError("Mask is empty. Provide a foreground ROI mask with at least one voxel.")

        if image_affine is not None and mask_affine is not None and not np.allclose(image_affine, mask_affine, atol=1e-4):
            logger.warning(
                "Image and mask affines differ. Proceeding based on image affine for output."
            )

    def _validate_init_params(self) -> None:
        """Validate constructor configuration."""
        self._validate_percentiles(self.clip_percentiles, "clip_percentiles")
        self._validate_percentiles(self.stretch_percentiles, "stretch_percentiles")
        if self.normalization not in {"minmax", "zscore", "none"}:
            raise ValueError(
                "Unsupported normalization. Use 'minmax', 'zscore', or 'none'."
            )
        if self.slab_thickness < 1:
            raise ValueError("slab_thickness must be >= 1.")
        if self.slab_thickness % 2 == 0:
            logger.warning(
                "slab_thickness=%d is even. Incrementing to %d for centered windows.",
                self.slab_thickness,
                self.slab_thickness + 1,
            )
            self.slab_thickness += 1

    def _validate_percentiles(self, values: Tuple[float, float], name: str) -> None:
        low, high = values
        if not (0.0 <= low < high <= 100.0):
            raise ValueError(
                f"{name} must satisfy 0 <= low < high <= 100. Received {values}."
            )

    def _robust_clip(
        self, image: np.ndarray, mask: np.ndarray, percentiles: Tuple[float, float]
    ) -> np.ndarray:
        """Clip intensities using robust percentiles computed inside the mask."""
        roi = image[mask]
        p_low, p_high = np.percentile(roi, percentiles)
        if np.isclose(p_low, p_high):
            return self._apply_mask(image, mask)
        clipped = np.clip(image, p_low, p_high)
        return clipped.astype(np.float32)

    def _percentile_stretch(
        self, image: np.ndarray, mask: np.ndarray, percentiles: Tuple[float, float]
    ) -> np.ndarray:
        """Stretch intensities to [0, 1] using mask-restricted percentiles."""
        roi = image[mask]
        p_low, p_high = np.percentile(roi, percentiles)
        if np.isclose(p_low, p_high):
            return np.zeros_like(image, dtype=np.float32)
        stretched = (image - p_low) / (p_high - p_low)
        return np.clip(stretched, 0.0, 1.0).astype(np.float32)

    def _invert_intensity(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Invert intensities based on mask-restricted min/max range."""
        roi = image[mask]
        v_min = float(np.min(roi))
        v_max = float(np.max(roi))
        if np.isclose(v_min, v_max):
            return np.zeros_like(image, dtype=np.float32)
        return (v_min + v_max - image).astype(np.float32)

    def _normalize(self, image: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
        """Normalize data inside mask and keep zeros outside mask."""
        out = np.zeros_like(image, dtype=np.float32)
        roi = image[mask]
        if roi.size == 0:
            return out

        if mode == "none":
            out[mask] = roi.astype(np.float32)
            return out

        if mode == "zscore":
            mean = float(np.mean(roi))
            std = float(np.std(roi))
            if np.isclose(std, 0.0):
                return out
            out[mask] = ((roi - mean) / std).astype(np.float32)
            return out

        v_min = float(np.min(roi))
        v_max = float(np.max(roi))
        if np.isclose(v_min, v_max):
            return out
        out[mask] = ((roi - v_min) / (v_max - v_min)).astype(np.float32)
        return out

    def _replace_non_finite(self, array: np.ndarray, where: str) -> np.ndarray:
        """Replace NaN/Inf values with zeros and log when this occurs."""
        if np.any(~np.isfinite(array)):
            logger.warning("Found NaN/Inf values in %s. Replacing with zeros.", where)
            return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return array

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Set outside-mask voxels to zero."""
        masked = np.zeros_like(image, dtype=np.float32)
        masked[mask] = image[mask]
        return masked

    def _mask_bounding_box(
        self, mask: np.ndarray
    ) -> Optional[Tuple[slice, slice, slice]]:
        """Return the tight 3D bounding box of non-zero mask voxels."""
        if not np.any(mask):
            return None
        coords = np.where(mask)
        return tuple(
            slice(int(np.min(axis_coords)), int(np.max(axis_coords)) + 1)
            for axis_coords in coords
        )  # type: ignore[return-value]

    def _expand_bbox(
        self, bbox: Tuple[slice, slice, slice], shape: Tuple[int, int, int], margin: int
    ) -> Tuple[slice, slice, slice]:
        """Expand a 3D bounding box by ``margin`` voxels, clipped to volume bounds."""
        if margin <= 0:
            return bbox

        expanded = []
        for dim, slc in enumerate(bbox):
            start = max(0, slc.start - margin)
            stop = min(shape[dim], slc.stop + margin)
            expanded.append(slice(start, stop))
        return tuple(expanded)  # type: ignore[return-value]


