# **Prediction Module**

This is a stand-alone module to produce segmentation of input images by using the assigned model.

## Pretrained model

Download the primary TOF-MRA checkpoint from the [VesselBoost Hugging Face repository](https://huggingface.co/BrainVascuLab/VesselBoost):

```bash
mkdir -p ./pretrained_models
hf download BrainVascuLab/VesselBoost weights/BM_VB2_aug_all_ep2k_bat_10_0903 --revision 2dfcb64056110d819b073ff82934cc54fe3dd773 --local-dir ./pretrained_models
path_to_pretrained_model="./pretrained_models/weights/BM_VB2_aug_all_ep2k_bat_10_0903"
```

## Example test run of this script:

If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images:

```bash
python prediction.py \
--image_path $path_to_images \
--output_path $path_to_output \
--pretrained $path_to_pretrained_model \
--prep_mode 4 \
--use_blending
```

If you set `prep_mode` to 1, 2, or 3, the module applies (1) N4 bias field correction, (2) denoising, or (3) both N4 bias field correction and denoising. You must provide a path for storing the preprocessed images. The following example applies only N4 bias field correction.

```bash
python prediction.py \
--image_path $path_to_images \
--preprocessed_path $path_to_preprocessed_images \
--output_path $path_to_output \
--pretrained $path_to_pretrained_model \
--prep_mode 1 \
--use_blending
```
