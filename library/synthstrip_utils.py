"""Utilities to run SynthStrip programmatically.
Original work: 
    A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann.
    SynthStrip: Skull-Stripping for Any Brain Image.
    https://arxiv.org/abs/2203.09974
    https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip
Refactored from:
    https://github.com/nipreps/synthstrip

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
from nitransforms.linear import Affine
from torch import nn

import nibabel as nb
import numpy as np
import torch
import scipy

SYNTHSTRIP_WEIGHTS_FILENAME = "synthstrip.1.pt"
SYNTHSTRIP_WEIGHTS_ENV = "VESSELBOOST_SYNTHSTRIP_WEIGHTS"
SYNTHSTRIP_WEIGHTS_URL = "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_weight_paths(weights_path: Optional[os.PathLike[str] | str] = None) -> list[Path]:
    """Return local SynthStrip weight paths in lookup order."""
    candidates: list[Path] = []

    def add_candidate(path_like: os.PathLike[str] | str) -> None:
        path = Path(path_like).expanduser()
        candidates.append(path)
        if path.name != SYNTHSTRIP_WEIGHTS_FILENAME:
            candidates.append(path / SYNTHSTRIP_WEIGHTS_FILENAME)

    if weights_path:
        add_candidate(weights_path)

    env_path = os.environ.get(SYNTHSTRIP_WEIGHTS_ENV)
    if env_path:
        add_candidate(env_path)

    candidates.extend(
        [
            _repo_root() / "saved_models" / SYNTHSTRIP_WEIGHTS_FILENAME,
            Path.cwd() / "saved_models" / SYNTHSTRIP_WEIGHTS_FILENAME,
            Path.cwd().parent / "saved_models" / SYNTHSTRIP_WEIGHTS_FILENAME,
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def resolve_weights_path(weights_path: Optional[os.PathLike[str] | str] = None) -> Path:
    """
    Resolve the local SynthStrip weights path without attempting network access.
    """
    candidates = _candidate_weight_paths(weights_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "SynthStrip brain extraction requires local model weights, but "
        f"{SYNTHSTRIP_WEIGHTS_FILENAME} was not found.\n"
        f"Place {SYNTHSTRIP_WEIGHTS_FILENAME} in ./saved_models, or set "
        f"{SYNTHSTRIP_WEIGHTS_ENV} to the weights file or containing directory.\n"
        f"Searched:\n  - {searched}"
    )


def _download_destination(weights_path: Optional[os.PathLike[str] | str] = None) -> Optional[Path]:
    if weights_path:
        return Path(weights_path).expanduser()

    env_path = os.environ.get(SYNTHSTRIP_WEIGHTS_ENV)
    if env_path:
        return Path(env_path).expanduser()

    return None


def download_weights(destination: Optional[os.PathLike[str] | str] = None, timeout: int = 60) -> Path:
    """
    Download SynthStrip weights into the requested destination.
    """
    import requests

    destination_path = Path(destination).expanduser() if destination else _repo_root() / "saved_models"
    if destination_path.name != SYNTHSTRIP_WEIGHTS_FILENAME:
        destination_path = destination_path / SYNTHSTRIP_WEIGHTS_FILENAME
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        print(f"\nSynthStrip weights already exist at {destination_path}. Skipping download.")
        return destination_path.resolve()

    print(f"Downloading SynthStrip weights from {SYNTHSTRIP_WEIGHTS_URL}...")
    try:
        response = requests.get(SYNTHSTRIP_WEIGHTS_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Failed to download SynthStrip weights. Check the internet connection, "
            f"or place {SYNTHSTRIP_WEIGHTS_FILENAME} in ./saved_models or set "
            f"{SYNTHSTRIP_WEIGHTS_ENV} to a local weights path."
        ) from exc

    destination_path.write_bytes(response.content)
    print(f"Download complete: {destination_path}")
    return destination_path.resolve()


def get_or_download_weights(weights_path: Optional[os.PathLike[str] | str] = None) -> Path:
    """
    Resolve local SynthStrip weights, downloading them if they are missing.
    """
    try:
        return resolve_weights_path(weights_path)
    except FileNotFoundError:
        try:
            return download_weights(_download_destination(weights_path))
        except RuntimeError as download_error:
            raise RuntimeError(
                "SynthStrip weights were not found locally and could not be downloaded. "
                "An internet connection is required for the automatic download; "
                f"offline deployments must provide {SYNTHSTRIP_WEIGHTS_FILENAME} locally."
            ) from download_error


def load_strip_model(device: torch.device, weights_path: Optional[os.PathLike[str] | str] = None):
    """
    Load the `StripModel` weights from a checkpoint file.
    """

    modelfile = get_or_download_weights(weights_path)

    model = StripModel()
    model.to(device)
    model.eval()

    checkpoint = torch.load(str(modelfile), map_location=device)
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    model.load_state_dict(state)
    return model


def conform(input_nii: nb.nifti1.Nifti1Image) -> nb.nifti1.Nifti1Image:
    """Resample image as SynthStrip likes it (copied from CLI).

    Returns a new `nb.nifti1.Nifti1Image` in a 1mm LIA grid with dimensions clipped
    to the synthstrip-preferred multiples.
    """

    shape = np.array(input_nii.shape[:3])
    affine = input_nii.affine

    corner_centers_ijk = (
        np.array(
            [
                (i, j, k)
                for k in (0, shape[2] - 1)
                for j in (0, shape[1] - 1)
                for i in (0, shape[0] - 1)
            ]
        )
        + 0.5
    )

    corners_xyz = affine @ np.hstack((corner_centers_ijk, np.ones((len(corner_centers_ijk), 1)))).T

    target_affine = np.diag([-1.0, 1.0, -1.0, 1.0])[:, (0, 2, 1, 3)]

    extent = corners_xyz.min(1)[:3], corners_xyz.max(1)[:3]
    target_shape = ((extent[1] - extent[0]) / 1.0 + 0.999).astype(int)

    target_shape = np.clip(np.ceil(np.array(target_shape) / 64).astype(int) * 64, 192, 320)

    target_shape[2], target_shape[1] = target_shape[1:3]

    input_c = affine @ np.hstack((0.5 * (shape - 1), 1.0))
    target_c = target_affine @ np.hstack((0.5 * (target_shape - 1), 1.0))

    target_affine[:3, 3] -= target_c[:3] - input_c[:3]

    nii = Affine(reference=nb.nifti1.Nifti1Image(np.zeros(target_shape), target_affine, None)).apply(input_nii)
    return nii


def resample_like(image: nb.nifti1.Nifti1Image, target: nb.nifti1.Nifti1Image, output_dtype=None, cval=0):
    """Resample the input image to be in the target's grid via identity transform."""

    return Affine(reference=target).apply(image, output_dtype=output_dtype, cval=cval)


def skull_strip(
    image: nb.nifti1.Nifti1Image,
    device: torch.device,
    border: int = 1,
    weights_path: Optional[os.PathLike[str] | str] = None,
) -> Tuple[np.ndarray, nb.nifti1.Nifti1Image]:
    """Run the synthstrip pipeline on an input image and return the mask.

    Parameters
    - image: input image as a Nifti1Image object
    - device: torch device to use. If omitted, it's configured from `gpu`.
    - border: border threshold in mm used to generate final mask
    - weights_path: optional local path to SynthStrip weights

    Returns
    - mask: boolean numpy array of the brain mask in native image grid
    """

    model = load_strip_model(device, weights_path)

    # load input volume
    conformed = conform(image)
    in_data = conformed.get_fdata(dtype='float32')
    in_data -= in_data.min()
    in_data = np.clip(in_data / np.percentile(in_data, 99), 0, 1)
    in_data = in_data[np.newaxis, np.newaxis]

    input_tensor = torch.from_numpy(in_data).to(device)
    with torch.no_grad():
        sdt = model(input_tensor).cpu().numpy().squeeze()

    sdt_target = resample_like(nb.nifti1.Nifti1Image(sdt, conformed.affine, None), image, output_dtype='int16', cval=100)
    sdt_data = np.asanyarray(sdt_target.dataobj).astype('int16')

    components = scipy.ndimage.label(sdt_data.squeeze() < border)[0]
    bincount = np.bincount(components.flatten())[1:]
    mask = components == (np.argmax(bincount) + 1)
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)

    # produce masked image
    img_data = image.get_fdata()
    bg = np.min([0, img_data.min()])
    masked = img_data.copy()
    masked[mask == 0] = bg
    masked_nifti = nb.nifti1.Nifti1Image(masked, image.affine, image.header.copy())

    return mask, masked_nifti

### DL model
class StripModel(nn.Module):
    def __init__(
        self,
        nb_features=16,
        nb_levels=7,
        feat_mult=2,
        max_features=64,
        nb_conv_per_level=2,
        max_pool=2,
        return_mask=False,
    ):
        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out
