### Update Log - 17/Aug/2026

- Version: VesselBoost 2.0.5
- Updated CI tests to enable Gaussian blending during prediction for the `boost`, `prediction`, and `test_time_adaptation` modules.

### Update Log - 14/Aug/2026

- Version: VesselBoost 2.0.4
- Completed the migration of pretrained model distribution and CI-generated outputs to Hugging Face:
  - Organized all pretrained checkpoints under `weights/` in the public [VesselBoost model repository](https://huggingface.co/BrainVascuLab/VesselBoost).
  - Added version-controlled model metadata, licensing, configuration, and SHA-256 manifests under `saved_models/` while keeping checkpoint binaries out of Git.
  - Updated the README, module documentation, and prediction/TTA notebooks to use the new Hugging Face checkpoint paths.
- Replaced OSF uploads from the boost, prediction, training, TTA, and Docker test workflows with uploads to the public [VesselBoost CI bucket](https://huggingface.co/buckets/BrainVascuLab/vesselboost-ci).
  - CI outputs are published only for pushes to `master`; pull-request runs do not receive the bucket write token or publish artifacts.
  - Added a Hugging Face integration workflow that uploads a temporary object, downloads it anonymously, verifies its checksum, and removes it afterward.
- Retained the existing public OSF test image and label as on-demand CI inputs. Tests download them only during a run and verify their SHA-256 checksums.
- Pinned CI checkpoint downloads to Hugging Face revision `2dfcb64056110d819b073ff82934cc54fe3dd773` and verify the primary checkpoint checksum before inference.
- Removed the obsolete OSF upload integration workflow and its credential usage from GitHub Actions.

### Update Log - 13/Aug/2026

- Version: VesselBoost 2.0.3
- Prepared the pretrained checkpoints in `saved_models/` for distribution through the [VesselBoost Hugging Face repository](https://huggingface.co/BrainVascuLab/VesselBoost):
  - Preserved all checkpoint filenames and extensions.
  - Added SHA-256 checksums in `MANIFEST.sha256` for integrity verification.
  - Added a Hugging Face model card with checkpoint descriptions, architecture and preprocessing details, supported MRI contrasts, known limitations, research-use-only notice, licensing, and citation links.
  - Added MIT licensing and Hugging Face metadata for PyTorch image segmentation and medical MRI.
  - Added a machine-readable `config.json` describing the 3D U-Net architecture, channels, filters, patch size, preprocessing, postprocessing defaults, and compatible VesselBoost source release.
- Updated and cleaned the Python dependencies:
  - Added `huggingface_hub` to `environment.yml`, `environment-ci.yml`, and `requirements.txt`.
  - Updated the GPU environment to PyTorch 2.10.0 with CUDA 12.6 and the CPU environments to PyTorch 2.10.0 CPU wheels.
  - Replaced the full `nnunetv2` dependency with the directly used `dynamic-network-architectures` package.
  - Added the directly used `scikit-image` package and removed unused direct dependencies.
- Updated the installation documentation with separate GPU, Conda CPU/CI, and pip CPU workflows.
- Updated checkpoint loading to use PyTorch's `weights_only=True` mode for VesselBoost and SynthStrip state dictionaries.
- Validated all three dependency specifications in clean sandbox environments with package-consistency checks, imports, strict checkpoint loading, CPU smoke tests, and GPU smoke tests.
- Added GitHub release automation that runs only after an internal `dev` to `master` pull request is merged. The workflow validates that the README `Current Version` was updated, creates the corresponding `vX.Y.Z` tag, and publishes a normal GitHub Release with automatically generated release notes. Model weights are not attached to GitHub Releases.

### Update Log - 13/Apr/2026

- Version: VesselBoost 2.0.2
- Decoupled brain extraction from `prep_mode` selection so `--enable_brain_extraction` can be used with any preprocessing mode, including `prep_mode=4`.
- Existing preprocessing behaviour for bias field correction and denoising is unchanged.
- When `--enable_brain_extraction` is used, the --preprocessed_path will be needed to save the brain extracted image, which will be used for subsequent steps (prediction, TTA and boost). Usage example:

```python
    python prediction.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --output_path "./data/pred_seg" \
    --pretrained "./saved_models/manual_0429" \
    --prep_mode 4 \
    --enable_brain_extraction
```

```python
    python test_time_adaptation.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --output_path "./data/pred_seg" \
    --pretrained "./saved_models/manual_0429" \
    --prep_mode 4 \
    --enable_brain_extraction \
    --epochs 100 \
    --learning_rate 1e-3
```

```python
    python boost.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --label_path "./data/seg/" \
    --output_path "./data/boost_seg/" \
    --output_model "./data/boost_seg/boost_model" \
    --prep_mode 4 \
    --enable_brain_extraction \
    --epochs 100 \
    --learning_rate 1e-2
```

### Update Log - 06/Mar/2026

- Version: VesselBoost 2.0.1
- Incorporated Synthstrip (from FreeSurfer) for brain extraction to improve the robustness of the preprocessing step. Usage example:

```python
    python prediction.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --output_path "./data/pred_seg" \
    --pretrained "./saved_models/manual_0429" \
    --prep_mode 1 \
    --enable_brain_extraction \
    --use_blending \
    --overlap_ratio 0.5
```

```python
    python test_time_adaptation.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --output_path "./data/pred_seg" \
    --pretrained "./saved_models/manual_0429" \
    --prep_mode 1 \
    --enable_brain_extraction \
    --epochs 100 \
    --learning_rate 1e-3 \
    --use_blending \
    --overlap_ratio 0.5
```

```python
    python boost.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --label_path "./data/seg/" \
    --output_path "./data/boost_seg/" \
    --output_model "./data/boost_seg/boost_model" \
    --prep_mode 1 \
    --enable_brain_extraction \
    --epochs 100 \
    --learning_rate 1e-2 \
    --use_blending \
    --overlap_ratio 0.5
```

- For prep_mode=4, there will be no brain extraction

### Update Log - 18/Feb/2026

- Offical release of VesselBoost 2.0.0
- Roll back image preprocessing step to standardization, as it is more robust to outliers in the data and can lead to better performance in some cases. The decision to switch back to standardization was based on empirical results and feedback from users, which indicated that standardization provided more consistent and reliable results across different datasets and imaging modalities.
- Added Gaussian blending for patch-based prediction to reduce edge artifacts and improve the quality of the final segmentation. Usage example:

`````````python
    python test_time_adaptation.py \
        --image_path "/data/img/" \
        --output_path "/output_path" \
        --pretrained "/saved_model" \
        --prep_mode 4 \
        --epochs 10 \
        --augmentation_mode "intensity" \
        --learning_rate 1e-3 \
        --use_blending \ # indicator to use Gaussian blending
        --overlap_ratio 0.5 \ # the overlap ratio between adjacent patches
```

### Update Log - 16/Sept/2025

- Pre-release of VesselBoost 2.0.0
- New data augmentation strategies during training (train, TTA and boost)
- Changed image preprocessing step from standardization to normalization
- Improved code structure and readability
- Bugs fixes and performance improvements
- Added support for T2*-weighted imaging (experimental)
- TODO: add notebook & github action test for previous version (1.0.0)
