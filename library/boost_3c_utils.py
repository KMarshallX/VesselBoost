"""
3-channel booster utilities for training and prediction.

This module adds a minimal, non-destructive 3-channel path in parallel to the
existing single-channel booster utilities.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cc3d
import matplotlib.pyplot as plt
import nibabel as nib  # type: ignore
import numpy as np
import scipy.ndimage as scind
import torch
import torchio as tio
from tqdm import tqdm

from .loss_func import choose_DL_model, choose_loss_metric, choose_optimizer, normaliser, standardiser

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# -----------------------------
# Shared helpers
# -----------------------------
def _is_nifti_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))


def _to_channel_first_3(image: np.ndarray, image_name: str) -> np.ndarray:
    """Convert input image to [3, D, H, W] if needed."""
    if image.ndim != 4:
        raise ValueError(
            f"Expected 4D 3-channel image for {image_name}, got shape {image.shape}."
        )

    if image.shape[0] == 3:
        return image.astype(np.float32)

    if image.shape[-1] == 3:
        # NIfTI often stores multichannel as [D, H, W, C]
        return np.moveaxis(image, -1, 0).astype(np.float32)

    raise ValueError(
        f"Expected 3 channels in first or last axis for {image_name}, got shape {image.shape}."
    )


def _to_single_channel_label(label: np.ndarray, label_name: str) -> np.ndarray:
    """Convert label to [D, H, W]."""
    if label.ndim == 3:
        return label.astype(np.float32)

    if label.ndim == 4:
        if label.shape[0] == 1:
            return label[0].astype(np.float32)
        if label.shape[-1] == 1:
            return label[..., 0].astype(np.float32)

    raise ValueError(
        f"Expected 3D single-channel label for {label_name}, got shape {label.shape}."
    )


def _normalize_channelwise(volume: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize a [C, D, H, W] volume channel-wise."""
    out = np.empty_like(volume, dtype=np.float32)
    for c in range(volume.shape[0]):
        if normalization == "standardize":
            out[c] = standardiser(volume[c]).astype(np.float32)
        elif normalization == "normalize":
            out[c] = normaliser(volume[c]).astype(np.float32)
        else:
            out[c] = volume[c].astype(np.float32)
    return out


def _resize_3d(arr: np.ndarray, target_shape: Tuple[int, int, int], order: int = 0) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    zoom_factors = tuple(float(t) / float(s) for t, s in zip(target_shape, arr.shape))
    return scind.zoom(arr, zoom_factors, order=order, mode="nearest")


def _resize_4d_channel_first(volume: np.ndarray, target_shape: Tuple[int, int, int], order: int = 0) -> np.ndarray:
    c = volume.shape[0]
    resized = np.empty((c, *target_shape), dtype=np.float32)
    for ch in range(c):
        resized[ch] = _resize_3d(volume[ch], target_shape, order=order).astype(np.float32)
    return resized


def _calculate_patch_dimensions(original_size: Tuple[int, int, int], patch_size: int = 64) -> Tuple[int, int, int]:
    new_dims: List[int] = []
    for dim in original_size:
        if dim > patch_size and dim % patch_size != 0:
            new_dim = int(np.ceil(dim / patch_size)) * patch_size
        elif dim < patch_size:
            new_dim = patch_size
        else:
            new_dim = dim
        new_dims.append(new_dim)
    return tuple(new_dims)  # type: ignore[return-value]


def _build_torchio_transform(mode: str) -> Optional[tio.Transform]:
    """Build TorchIO transform matching existing augmentation modes."""
    mode = mode.lower()
    if mode == "off":
        return None

    if mode == "all":
        return tio.Compose([
            tio.RandomBlur(p=1),
            tio.RandomBiasField(coefficients=0.5, p=1),
            tio.RandomNoise(std=0.5, p=1),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=1.0, p=1),
            tio.RandomElasticDeformation(num_control_points=9, max_displacement=2, locked_borders=2, p=1),
        ])

    if mode == "random":
        return tio.OneOf({
            tio.RandomBlur(p=1): 0.1,
            tio.RandomBiasField(coefficients=0.5, p=1): 0.1,
            tio.RandomNoise(std=0.5, p=1): 0.1,
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=1.0, p=1): 0.35,
            tio.RandomElasticDeformation(num_control_points=9, max_displacement=2, locked_borders=2, p=1): 0.35,
        })

    if mode == "spatial":
        return tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=1.0, p=1),
            tio.RandomElasticDeformation(num_control_points=9, max_displacement=2, locked_borders=2, p=1),
        ])

    if mode == "intensity":
        return tio.OneOf([
            tio.RandomBlur(p=1),
            tio.RandomBiasField(coefficients=0.5, p=1),
            tio.RandomNoise(std=0.5, p=1),
        ])

    if mode == "flip":
        return tio.RandomFlip(axes=(0, 1, 2), flip_probability=1.0, p=1)

    logger.warning("Unsupported augmentation mode '%s' for 3-channel path. Using 'off'.", mode)
    return None


def _apply_torchio_augmentation_3c(
    image_batch: torch.Tensor,
    seg_batch: torch.Tensor,
    augmentation_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply TorchIO sample-wise for [B, C, D, H, W] image and [B, 1, D, H, W] label."""
    transform = _build_torchio_transform(augmentation_mode)
    if transform is None:
        return image_batch, seg_batch

    images_out: List[torch.Tensor] = []
    labels_out: List[torch.Tensor] = []

    for i in range(image_batch.shape[0]):
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_batch[i]),
            label=tio.LabelMap(tensor=seg_batch[i]),
        )
        transformed = transform(subject)
        images_out.append(transformed["image"].data)
        labels_out.append(transformed["label"].data)

    return torch.stack(images_out, dim=0), torch.stack(labels_out, dim=0)


# -----------------------------
# Data loaders
# -----------------------------
class ThreeChannelSingleLoader:
    """Single subject loader for one 3-channel image and one 3D segmentation mask."""

    def __init__(
        self,
        img_path: PathLike,
        seg_path: PathLike,
        patch_size: Tuple[int, int, int],
        step: int,
        normalization: str = "standardize",
        crop_low_thresh: int = 64,
        batch_multiplier: int = 5,
    ):
        self.img_path = Path(img_path)
        self.seg_path = Path(seg_path)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.step = int(step)
        self.normalization = normalization
        self.crop_low_thresh = int(crop_low_thresh)
        self.batch_multiplier = int(batch_multiplier)

        if not self.img_path.exists():
            raise FileNotFoundError(f"Image not found: {self.img_path}")
        if not self.seg_path.exists():
            raise FileNotFoundError(f"Label not found: {self.seg_path}")

        self.image_arr, self.label_arr = self._load_pair()
        self._spatial_shape = self.label_arr.shape

    def _load_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        img_nifti = nib.load(str(self.img_path))  # type: ignore
        seg_nifti = nib.load(str(self.seg_path))  # type: ignore

        image = _to_channel_first_3(img_nifti.get_fdata().astype(np.float32), self.img_path.name)  # type: ignore
        label = _to_single_channel_label(seg_nifti.get_fdata().astype(np.float32), self.seg_path.name)  # type: ignore

        if image.shape[1:] != label.shape:
            raise ValueError(
                f"Spatial shape mismatch: image {image.shape} vs label {label.shape}"
            )

        image = _normalize_channelwise(image, self.normalization)
        label = np.where(label > 0, 1.0, 0.0).astype(np.float32)

        return image, label

    def _random_crop_bounds(self) -> Tuple[slice, slice, slice]:
        starts: List[int] = []
        sizes: List[int] = []
        for dim in self._spatial_shape:
            low = min(self.crop_low_thresh, dim)
            if low < dim:
                crop = int(np.random.randint(low, dim + 1))
            else:
                crop = dim
            start = 0 if crop == dim else int(np.random.randint(0, dim - crop + 1))
            starts.append(start)
            sizes.append(crop)

        return (
            slice(starts[0], starts[0] + sizes[0]),
            slice(starts[1], starts[1] + sizes[1]),
            slice(starts[2], starts[2] + sizes[2]),
        )

    def _sample_patch_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        crop_slices = self._random_crop_bounds()
        image_crop = self.image_arr[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        label_crop = self.label_arr[crop_slices[0], crop_slices[1], crop_slices[2]]

        if image_crop.shape[1:] != self.patch_size:
            image_crop = _resize_4d_channel_first(image_crop, self.patch_size, order=0)
            label_crop = _resize_3d(label_crop, self.patch_size, order=0)

        return image_crop.astype(np.float32), label_crop.astype(np.float32)

    def __len__(self) -> int:
        return self.step

    def __iter__(self):
        for _ in range(self.step):
            batch_size = self.batch_multiplier + 1
            image_batch = np.empty((batch_size, 3, *self.patch_size), dtype=np.float32)
            label_batch = np.empty((batch_size, 1, *self.patch_size), dtype=np.float32)

            for j in range(batch_size):
                img_patch, seg_patch = self._sample_patch_pair()
                image_batch[j] = img_patch
                label_batch[j, 0] = np.where(seg_patch > 0, 1.0, 0.0)

            yield torch.from_numpy(image_batch), torch.from_numpy(label_batch)


class ThreeChannelMultiLoader:
    """Directory-level loader pairing 3-channel images with single-channel labels."""

    def __init__(
        self,
        img_dir: PathLike,
        seg_dir: PathLike,
        patch_size: Tuple[int, int, int],
        step: int,
        normalization: str = "standardize",
        crop_low_thresh: int = 64,
        batch_multiplier: int = 5,
    ):
        self.img_dir = Path(img_dir)
        self.seg_dir = Path(seg_dir)
        self.patch_size = patch_size
        self.step = step
        self.normalization = normalization
        self.crop_low_thresh = crop_low_thresh
        self.batch_multiplier = batch_multiplier

        self.image_pairs = self._find_image_pairs()
        logger.info("Found %d 3-channel image/label pairs", len(self.image_pairs))

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        if not self.img_dir.exists() or not self.seg_dir.exists():
            raise FileNotFoundError("Image or label directory not found")

        img_files = sorted([f for f in self.img_dir.iterdir() if _is_nifti_file(f)])
        seg_files = sorted([f for f in self.seg_dir.iterdir() if _is_nifti_file(f)])

        if len(img_files) != len(seg_files):
            raise ValueError(
                f"Mismatch: {len(img_files)} image files vs {len(seg_files)} label files"
            )

        seg_lookup = {f.stem.split(".")[0]: f for f in seg_files}
        image_pairs: List[Tuple[Path, Path]] = []

        for img_file in img_files:
            img_stem = img_file.stem.split(".")[0]
            seg_file = seg_lookup.get(img_stem)
            if seg_file is None:
                for seg_key, seg_path in seg_lookup.items():
                    if img_stem in seg_key or seg_key in img_stem:
                        seg_file = seg_path
                        break
            if seg_file is None:
                raise ValueError(f"No matching label found for {img_file.name}")

            image_pairs.append((img_file, seg_file))

        return image_pairs

    def get_loader(self, index: int) -> ThreeChannelSingleLoader:
        if index >= len(self.image_pairs):
            raise IndexError(f"Index {index} out of range")
        img_path, seg_path = self.image_pairs[index]
        return ThreeChannelSingleLoader(
            img_path,
            seg_path,
            self.patch_size,
            self.step,
            normalization=self.normalization,
            crop_low_thresh=self.crop_low_thresh,
            batch_multiplier=self.batch_multiplier,
        )

    def get_all_loaders(self) -> Dict[int, ThreeChannelSingleLoader]:
        return {i: self.get_loader(i) for i in range(len(self.image_pairs))}


# -----------------------------
# Trainer
# -----------------------------
class Trainer3C:
    """3-channel booster trainer aligned with the existing Trainer workflow."""

    def __init__(
        self,
        loss_name: str,
        model_name: str,
        input_channels: int,
        output_channels: int,
        filter_count: int,
        optimizer_name: str,
        learning_rate: float,
        optimizer_gamma: float,
        num_epochs: int,
        batch_multiplier: int,
        patch_size: Tuple[int, int, int],
        augmentation_mode: str,
        threshold: Optional[float] = None,
        connect_threshold: Optional[int] = None,
        crop_low_thresh: int = 64,
    ):
        if input_channels != 3:
            raise ValueError(f"Trainer3C requires input_channels=3, got {input_channels}")

        self.loss_name = loss_name
        self.model_name = model_name
        self.model_config = [input_channels, output_channels, filter_count]

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.optimizer_gamma = optimizer_gamma
        self.num_epochs = num_epochs
        self.batch_multiplier = batch_multiplier

        self.patch_size = patch_size
        self.augmentation_mode = augmentation_mode
        self.threshold = threshold
        self.connect_threshold = connect_threshold
        self.crop_low_thresh = crop_low_thresh

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer3C initialized on device: %s", self.device)

    def _initialize_model(self) -> torch.nn.Module:
        return choose_DL_model(
            self.model_name,
            self.model_config[0],
            self.model_config[1],
            self.model_config[2],
        ).to(self.device)

    def _initialize_loss(self) -> torch.nn.Module:
        return choose_loss_metric(self.loss_name)

    def _initialize_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:  # type: ignore
        patience = int(np.ceil(self.num_epochs * 0.2))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.optimizer_gamma,
            patience=patience,
        )

    def _train_epoch(
        self,
        data_loaders: Dict[int, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,  # type: ignore
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        loss_function: torch.nn.Module,
    ) -> Tuple[float, float]:
        model.train()

        total_loss = 0.0
        total_lr = 0.0
        num_batches = 0

        data_loading_time = 0.0
        model_training_time = 0.0

        iterators = {idx: iter(loader) for idx, loader in data_loaders.items()}

        for file_idx in range(len(data_loaders)):
            start_data = time.time()
            image_batch, label_batch = next(iterators[file_idx])
            image_batch, label_batch = _apply_torchio_augmentation_3c(
                image_batch,
                label_batch,
                self.augmentation_mode,
            )
            image_batch = image_batch.to(self.device, non_blocking=True)
            label_batch = label_batch.to(self.device, non_blocking=True)
            data_loading_time += time.time() - start_data

            start_model = time.time()
            optimizer.zero_grad()
            output = model(image_batch)
            loss = loss_function(output, label_batch)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            model_training_time += time.time() - start_model

            total_loss += float(loss.item())
            total_lr += float(optimizer.param_groups[0]["lr"])
            num_batches += 1

        logger.info(
            "\nData loading time: %.2fs, Model training time: %.2fs",
            data_loading_time,
            model_training_time,
        )

        return total_loss / num_batches, total_lr / num_batches

    def train_model(
        self,
        data_loaders: Dict[int, Any],
        model: torch.nn.Module,
        save_path: PathLike,
    ) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        optimizer = choose_optimizer(self.optimizer_name, model.parameters(), self.learning_rate)
        scheduler = self._initialize_scheduler(optimizer)
        loss_function = self._initialize_loss()

        logger.info("Starting 3-channel training for %d epochs", self.num_epochs)

        for epoch in tqdm(range(self.num_epochs), desc="Training (3C)"):
            avg_loss, avg_lr = self._train_epoch(
                data_loaders, model, optimizer, scheduler, loss_function
            )
            tqdm.write(
                f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.8f}, LR: {avg_lr:.8f}"
            )

        torch.save(model.state_dict(), str(save_path))
        logger.info("3-channel model saved to: %s", save_path)

    def train(
        self,
        processed_path: PathLike,
        segmentation_path: PathLike,
        output_model_path: PathLike,
    ) -> None:
        dataset = ThreeChannelMultiLoader(
            processed_path,
            segmentation_path,
            self.patch_size,
            self.num_epochs,
            normalization="standardize",
            crop_low_thresh=self.crop_low_thresh,
            batch_multiplier=self.batch_multiplier,
        )
        data_loaders = dataset.get_all_loaders()
        model = self._initialize_model()

        logger.info("Training with effective batch size: %d", self.batch_multiplier + 1)
        self.train_model(data_loaders, model, output_model_path)


# -----------------------------
# Predictor
# -----------------------------
class ImagePredictor3C:
    """Patch-based prediction for 3-channel [C, D, H, W] volumes."""

    def __init__(
        self,
        model_name: str,
        input_channel: int,
        output_channel: int,
        filter_number: int,
        input_path: PathLike,
        output_path: PathLike,
    ):
        if input_channel != 3:
            raise ValueError(f"ImagePredictor3C requires input_channel=3, got {input_channel}")

        self.model_name = model_name
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.filter_number = filter_number

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _save_results(
        self,
        image_name: str,
        probability_map: np.ndarray,
        binary_mask: np.ndarray,
        affine: np.ndarray,
        header: Any,
        save_probability: bool = False,
        save_mip: bool = False,
    ) -> None:
        if save_probability:
            prob_image = nib.Nifti1Image(probability_map, affine, header)  # type: ignore
            prob_path = self.output_path / f"SIGMOID_{image_name}"
            nib.save(prob_image, str(prob_path))  # type: ignore

        binary_image = nib.Nifti1Image(binary_mask, affine, header)  # type: ignore
        binary_path = self.output_path / image_name
        nib.save(binary_image, str(binary_path))  # type: ignore

        if save_mip:
            mip = np.max(binary_mask, axis=2)
            mip = np.rot90(mip, axes=(0, 1))
            mip_path = self.output_path / f"{Path(image_name).stem}.jpg"
            plt.imsave(str(mip_path), mip, cmap="gray")

    def _run_patch_inference(self, image_4d: np.ndarray, model: torch.nn.Module) -> np.ndarray:
        """Infer full volume with non-overlapping 64-cube patches."""
        _, d, h, w = image_4d.shape
        patch = 64

        prediction = np.zeros((d, h, w), dtype=np.float32)
        model.eval()
        sigmoid_fn = torch.nn.Sigmoid()

        with torch.no_grad():
            for z in range(0, d, patch):
                for y in range(0, h, patch):
                    for x in range(0, w, patch):
                        patch_arr = image_4d[:, z:z + patch, y:y + patch, x:x + patch]
                        patch_tensor = torch.from_numpy(patch_arr).unsqueeze(0).to(torch.float32).to(self.device)
                        pred = sigmoid_fn(model(patch_tensor)).cpu().numpy()[0, 0]
                        prediction[z:z + patch, y:y + patch, x:x + patch] = pred

        return prediction

    def process_single_image(
        self,
        image_name: str,
        model: torch.nn.Module,
        threshold: float,
        connect_threshold: int,
        save_mip: bool = False,
        save_probability: bool = False,
    ) -> None:
        image_path = self.input_path / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_image = nib.load(str(image_path))  # type: ignore
        header = raw_image.header
        affine = raw_image.affine  # type: ignore
        image_arr = raw_image.get_fdata().astype(np.float32)  # type: ignore

        image_c_first = _to_channel_first_3(image_arr, image_name)
        original_size = image_c_first.shape[1:]

        target_size = _calculate_patch_dimensions(original_size, 64)
        resized = _resize_4d_channel_first(image_c_first, target_size, order=0)
        resized = _normalize_channelwise(resized, normalization="standardize")

        prediction_map = self._run_patch_inference(resized, model)
        prediction_map = _resize_3d(prediction_map, original_size, order=0)

        binary_mask = (prediction_map >= threshold).astype(np.int32)
        cc3d.dust(binary_mask, connect_threshold, connectivity=26, in_place=True)

        self._save_results(
            image_name,
            prediction_map,
            binary_mask,
            affine,
            header,
            save_probability=save_probability,
            save_mip=save_mip,
        )

    def predict_all_images(
        self,
        model_path: PathLike,
        threshold: float,
        connect_threshold: int,
        save_mip: bool = True,
        save_probability: bool = False,
    ) -> None:
        model = choose_DL_model(
            self.model_name,
            self.input_channel,
            self.output_channel,
            self.filter_number,
        ).to(self.device)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self.device.type == "cuda":
            model.load_state_dict(torch.load(str(model_path)))  # type: ignore
        else:
            model.load_state_dict(torch.load(str(model_path), map_location=self.device))  # type: ignore

        image_files = [f for f in self.input_path.iterdir() if _is_nifti_file(f)]
        if not image_files:
            logger.warning("No image files found in %s", self.input_path)
            return

        for image_file in image_files:
            self.process_single_image(
                image_name=image_file.name,
                model=model,
                threshold=threshold,
                connect_threshold=connect_threshold,
                save_mip=save_mip,
                save_probability=save_probability,
            )

        logger.info("Completed 3-channel prediction for %d images", len(image_files))


# -----------------------------
# External API for boost_3c.py
# -----------------------------
def make_prediction_3c(
    model_name: str,
    input_channel: int,
    output_channel: int,
    filter_number: int,
    input_path: PathLike,
    output_path: PathLike,
    thresh: float,
    connect_thresh: int,
    test_model_name: PathLike,
    mip_flag: bool,
    probability_flag: bool = True,
) -> None:
    predictor = ImagePredictor3C(
        model_name,
        input_channel,
        output_channel,
        filter_number,
        input_path,
        output_path,
    )
    predictor.predict_all_images(
        model_path=test_model_name,
        threshold=thresh,
        connect_threshold=connect_thresh,
        save_mip=mip_flag,
        save_probability=probability_flag,
    )


def validate_three_channel_dataset(image_dir: PathLike, label_dir: PathLike) -> int:
    """Validate that all images are 3-channel and labels are single-channel masks."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    dataset = ThreeChannelMultiLoader(
        image_dir,
        label_dir,
        patch_size=(64, 64, 64),
        step=1,
        normalization="none",
        crop_low_thresh=32,
        batch_multiplier=1,
    )

    for idx in range(len(dataset.image_pairs)):
        img_path, seg_path = dataset.image_pairs[idx]

        img = nib.load(str(img_path)).get_fdata().astype(np.float32)  # type: ignore
        seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)  # type: ignore

        img_c = _to_channel_first_3(img, img_path.name)
        seg_1c = _to_single_channel_label(seg, seg_path.name)

        if img_c.shape[1:] != seg_1c.shape:
            raise ValueError(
                f"Spatial mismatch for {img_path.name}: image {img_c.shape} vs label {seg_1c.shape}"
            )

    return len(dataset.image_pairs)
