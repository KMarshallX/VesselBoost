### Update Log - 16/Sept/2025
- Pre-release of VesselBoost 2.0.0
- New data augmentation strategies during training (train, TTA and boost)
- Changed image preprocessing step from standardization to normalization
- Improved code structure and readability
- Bugs fixes and performance improvements
- Added support for T2*-weighted imaging (experimental)
- TODO: add notebook & github action test for previous version (1.0.0) 

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