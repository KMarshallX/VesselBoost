# **Boosting Module**
This module takes a single subject data and its coarse segmentation to train a model from scratch, and then use the trained model to predict a refined segmentation of the subject data. 

## Example test run of this script:
If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images:

```bash
python boost.py --ds_path $path_to_images --lb_path $path_to_labels --out_path $path_to_output --outmo $path_to_scratch_model --prep_mode 4 --ep $n_epochs --lr 1e-3
```

If you set prep_mode to 1,2 or 3, which means (1) N4 bias field correction, (2)denosing, or (3) both N4 biasfield correction and denoising will happen, then you have to set a path to store the preprocessed images. In the following example, we set the preprocessing mode to "applying N4 bias field correction only".

```bash
python boost.py --ds_path $path_to_images --ps_path $path_to_preprocessed_images --lb_path $path_to_labels --out_path $path_to_output --outmo $path_to_scratch_model --prep_mode 1 --ep $n_epochs --lr 1e-3
```
