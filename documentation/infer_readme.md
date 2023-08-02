# **Inference Module**
This is a stand-alone module to produce segmentation of input images by using the assigned model.

## Example test run of this script:
```bash
# Set the necessary parameters
# If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images
python inference.py --ds_path $path_to_images --out_path $path_to_output --pretrained $path_to_pretrained_model --prep_mode 4

# If you set prep_mode to 1,2 or 3, which means (1) N4 bias field correction, (2)denosing, or (3) both N4 biasfield correction and denoising will happen, then you have to set a path to store the preprocessed images
python inference.py --ds_path $path_to_images --ps_path $path_to_preprocessed_images --out_path $path_to_output --pretrained $path_to_pretrained_model --prep_mode 3

```
