# **Boosting Module**

This module uses a single subject's image data and coarse segmentation to train a model from scratch, then uses the trained model to predict a refined segmentation.

## Example test run of this script:

If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images:

```bash
python boost.py \
--image_path $path_to_images \
--label_path $path_to_labels \
--output_path $path_to_output \
--output_model $path_to_scratch_model \
--prep_mode 4 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--use_blending
```

If you set `prep_mode` to 1, 2, or 3, the module applies (1) N4 bias field correction, (2) denoising, or (3) both N4 bias field correction and denoising. You must provide a path for storing the preprocessed images. The following example applies only N4 bias field correction.

```bash
python boost.py \
--image_path $path_to_images \
--preprocessed_path $path_to_preprocessed_images \
--label_path $path_to_labels \
--output_path $path_to_output \
--output_model $path_to_scratch_model \
--prep_mode 1 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--use_blending
```
