---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
buckets:
  - BrainVascuLab/vesselboost-ci
tags:
  - medical-imaging
  - mri
  - tof-mra
  - t2star
  - vessel-segmentation
  - brain-vasculature
  - unet3d
  - pytorch
---
# VesselBoost pretrained weights

## Model purpose

VesselBoost segments small blood vessels in high-resolution human brain MRI. The primary models target time-of-flight magnetic resonance angiography (TOF-MRA). One checkpoint, `t2s_mod_ep1k2_0728`, provides experimental support for T2*-weighted MRI.

These files are PyTorch state dictionaries for use with the VesselBoost prediction and test-time adaptation workflows. They are not standalone Hugging Face Transformers models or hosted inference endpoints.

**Research use only. Not validated for clinical diagnosis, treatment planning, or other clinical decision-making.**

## Architecture and release pin

The checkpoints use the VesselBoost 3D U-Net with one input channel, one output channel, and 16 base filters. The network has four encoder stages, a bridge, four decoder stages with transposed-convolution upsampling and skip connections, and a final 1 x 1 x 1 convolution. Each convolutional block contains two 3 x 3 x 3 convolutions with batch normalization and ReLU activation.

The corresponding source release is pinned to:

- VesselBoost version: `2.0.5`
- Git tag: [`v2.0.5`](https://github.com/KMarshallX/VesselBoost/tree/v2.0.5)
- Git commit: [`3f028bbd6784c8fac82ac872a70aa06de2e162ae`](https://github.com/KMarshallX/VesselBoost/commit/3f028bbd6784c8fac82ac872a70aa06de2e162ae)

See `config.json` for the machine-readable inference configuration.

## Checkpoints

All pretrained checkpoints are stored under `weights/`. Their original filenames and serialization formats are preserved from the original release.

| Checkpoint                                         | MRI contrast     | Description                                                                                                                     |
| -------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [`manual_0429`](weights/manual_0429)                | TOF-MRA          | Primary TOF-MRA checkpoint referenced by the current VesselBoost documentation and automated tests.                            |
| [`omelette1_0429`](weights/omelette1_0429)          | TOF-MRA          | Legacy TOF-MRA checkpoint identified as Omelette variant 1.                                                                     |
| [`omelette2_0429`](weights/omelette2_0429)          | TOF-MRA          | Legacy TOF-MRA checkpoint identified as Omelette variant 2.                                                                     |
| [`t2s_mod_ep1k2_0728`](weights/t2s_mod_ep1k2_0728) | T2*-weighted MRI | Experimental T2*-weighted vessel-segmentation checkpoint. It has not received the same validation as the flagship TOF-MRA model. |

For standard TOF-MRA prediction and test-time adaptation workflows, use the flagship `manual_0429` checkpoint.

## Downloading checkpoints

Install the Hugging Face command-line client:

```bash
python -m pip install huggingface_hub
```

Download the primary TOF-MRA checkpoint:

```bash
hf download BrainVascuLab/VesselBoost \
  weights/manual_0429 \
  --local-dir saved_models
```

The downloaded checkpoint will be available at `saved_models/weights/manual_0429`.

Download every pretrained checkpoint and the checksum manifest:

```bash
hf download BrainVascuLab/VesselBoost \
  --include "weights/*" \
  --local-dir saved_models
```

For reproducible automated workflows, pass `--revision` with a specific Hugging Face commit hash rather than relying on the moving `main` branch.

## Preprocessing and inference

VesselBoost v2.0.5 performs the following inference operations:

1. Load a single-channel NIfTI MRI volume.
2. Resize each spatial dimension to at least 64 voxels and to a multiple of 64, using nearest-neighbor interpolation.
3. Apply whole-volume z-score standardization: subtract the volume mean and divide by its standard deviation. A constant-valued volume is mapped to zeros.
4. Divide the standardized image into non-overlapping `64 x 64 x 64` patches. The optional Gaussian-blending path uses overlapping patches.
5. Apply the 3D U-Net and a sigmoid activation to obtain vessel probabilities.
6. Threshold probabilities at the default value of `0.1`.
7. Remove connected components smaller than `10` voxels using 26-connectivity.
8. Resize the prediction back to the original image dimensions.

VesselBoost preprocessing modes can optionally perform N4 bias-field correction, denoising, both operations, or neither. Use the same preprocessing choices used for validation when comparing results. Brain extraction is optional and requires separate SynthStrip weights; those third-party weights are not part of this model release.

## Integrity verification

SHA-256 checksums for every checkpoint are provided in [`weights/MANIFEST.sha256`](weights/MANIFEST.sha256). After downloading all files, verify them with:

```bash
cd saved_models/weights
sha256sum --check MANIFEST.sha256
```

All four checkpoints should report `OK`.

Load the checkpoints with the pinned VesselBoost source and map tensors to the intended device. When supported by the installed PyTorch version, use `weights_only=True` when loading these state dictionaries.

## Known limitations and expected failure cases

- The models were developed for research MRI data and may not generalize to unseen scanners, field strengths, acquisition protocols, resolutions, populations, pathologies, or non-brain anatomy.
- The primary models target TOF-MRA. Applying them to other contrasts can produce unreliable results; T2* support is explicitly experimental.
- Bright non-vascular structures, noise, motion, ringing, bias fields, susceptibility artifacts, and incomplete brain masking can cause false positives.
- Low vessel contrast, slow or turbulent flow, signal dropout, very small vessels, severe pathology, and partial-volume effects can cause false negatives or disconnected vessels.
- Z-score standardization is performed over the supplied volume. Large background regions, unexpected cropping, NaN or infinite intensities, and constant-valued images can change or invalidate the result.
- Resizing and patch boundaries can alter fine structures. Gaussian blending may reduce patch-boundary artifacts but changes the inference procedure and should be reported.
- The default probability threshold of `0.1` and component cutoff of `10` voxels may require validation for a new dataset. Tuning them on evaluation cases can bias reported performance.
- Training labels for small vessels can be incomplete or imperfect. Predictions should not be interpreted as a complete representation of the cerebral vasculature.
- Detailed provenance for `manual_0429`, `omelette1_0429`, and `omelette2_0429` training runs is documented in our ApertureNeuro journal article *VesselBoost: A Python Toolbox for Small Blood Vessel Segmentation in Human Magnetic Resonance Angiography Data*.

## GitHub Actions CI outputs

The latest generated outputs from VesselBoost's GitHub Actions test workflows are stored in the public [VesselBoost CI bucket](https://huggingface.co/buckets/BrainVascuLab/vesselboost-ci).

The bucket uses the following layout:

```text
github_actions/
├── boost/predicted_labels/
├── docker/saved_model/
├── prediction/predicted_labels/
├── train/saved_model/
└── tta/predicted_labels/
```

These files are automated CI diagnostics, not validated model releases or benchmark results. Each successful push-triggered workflow replaces the previous contents of its corresponding directory.

## Resources and citation

- Paper DOI: [10.52294/001c.123217](https://doi.org/10.52294/001c.123217)
- Published article: [VesselBoost: A Python Toolbox for Small Blood Vessel Segmentation in Human Magnetic Resonance Angiography Data](https://apertureneuro.org/article/123217-vesselboost-a-python-toolbox-for-small-blood-vessel-segmentation-in-human-magnetic-resonance-angiography-data)
- GitHub: [KMarshallX/VesselBoost](https://github.com/KMarshallX/VesselBoost)
- OSF project and original model distribution: [osf.io/abk4p](https://osf.io/abk4p/)

Please cite:

```bibtex
@article{xuVesselBoostPythonToolbox2024,
  title = {VesselBoost: A Python Toolbox for Small Blood Vessel Segmentation in Human Magnetic Resonance Angiography Data},
  author = {Xu, Marshall and Ribeiro, Fernanda L. and Barth, Markus and Bernier, Micha\"el and Bollmann, Steffen and Chatterjee, Soumick and Cognolato, Francesco and Gulban, Omer F. and Itkyal, Vaibhavi and Liu, Siyu and Mattern, Hendrik and Polimeni, Jonathan R. and Shaw, Thomas B. and Speck, Oliver and Bollmann, Saskia},
  journal = {Aperture Neuro},
  volume = {4},
  year = {2024},
  doi = {10.52294/001c.123217}
}
```

## License

The files in this model release are provided under the MIT License. See `LICENSE`.
