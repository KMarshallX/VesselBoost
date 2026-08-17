# **Test-Time Adaptation Module**

This module uses a pretrained model or provided segmentations as a starting point, then fine-tunes the model on your image data.

## **Prediction with a pretrained model**

### **Prepare prerequisite model**

Pretrained VesselBoost models are published in the [VesselBoost Hugging Face repository](https://huggingface.co/BrainVascuLab/VesselBoost).

To access them, make a directory for the pre-trained models within the vessel_code folder:

```bash
mkdir ./pretrained_models/
```

Download the primary TOF-MRA model from Hugging Face. Pinning the repository revision makes the download reproducible:

```bash
hf download BrainVascuLab/VesselBoost weights/manual_0429 --revision f5cdbee052dde4f2a2a270674fd1c8d64dc8e861 --local-dir ./pretrained_models
```

The downloaded checkpoint is available at `./pretrained_models/weights/manual_0429`.

### **Test-time adaptation without a provided proxy segmentation**

You can apply this module directly to your own data without providing a proxy segmentation. The module will automatically generate proxies and fine-tune the model.
If you set `prep_mode` to 4, no preprocessing occurs, so you do not need to provide a path for storing preprocessed images. The input path must contain only the NIfTI images to process and must differ from the output path.

```bash
python test_time_adaptation.py \
--image_path $path_to_images \
--output_path $path_to_output \
--pretrained $path_to_pretrained_model \
--prep_mode 4 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--use_blending
```

If you set `prep_mode` to 1, 2, or 3, the module applies (1) N4 bias field correction, (2) denoising, or (3) both N4 bias field correction and denoising. You must provide a path for storing the preprocessed images. The following example applies only N4 bias field correction (mode 1):

```bash
python test_time_adaptation.py \
--image_path $path_to_images \
--output_path $path_to_output \
--preprocessed_path $path_to_preprocessed_images \
--pretrained $path_to_pretrained_model \
--prep_mode 1 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--use_blending
```

### **Test-time adaptation with a provided proxy segmentation**

You can provide a proxy segmentation to use for test-time adaptation instead of an automatically generated proxy. You can create this segmentation using any method, such as manual labelling or a non-deep-learning method. The module uses filenames to match raw images with proxy segmentations. If you set `prep_mode` to 4, no preprocessing occurs, so you do not need to provide a path for storing preprocessed images.

```bash
python test_time_adaptation.py \
--image_path $path_to_images \
--proxy_path $path_to_proxy_labels \
--output_path $path_to_output \
--pretrained $path_to_pretrained_model \
--prep_mode 4 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--use_blending
```

If you set `prep_mode` to 1, 2, or 3, the module applies (1) N4 bias field correction, (2) denoising, or (3) both N4 bias field correction and denoising. You must provide a path for storing the preprocessed images. The following example applies only N4 bias field correction (mode 1):

```bash
python test_time_adaptation.py \
--image_path $path_to_images \
--proxy_path $path_to_proxy_labels \
--output_path $path_to_output \
--preprocessed_path $path_to_preprocessed_images \
--pretrained $path_to_pretrained_model \
--prep_mode 1 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--use_blending
```
