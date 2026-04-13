### Update Log - 13/Apr/2026
- Version: VesselBoost 2.0.2
- Decoupled brain extraction from `prep_mode` selection so `--enable_brain_extraction` can be used with any preprocessing mode, including `prep_mode=4`.
- Existing preprocessing behaviour for bias field correction and denoising is unchanged.

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

```python
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
