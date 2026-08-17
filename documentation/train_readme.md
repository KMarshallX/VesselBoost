# **Training Module**

You can use this module to train your own base model.

## **Create a base model from scratch**

### Prepare the training data

If you are working outside a container, you can store your data under the *data* folder as shown below. If you are working inside a container, you can mount your data folders at *./data/train/* and *./data/label/*.
The segmentation filename must contain the corresponding MRI image's basename.

For example:

- Raw image: `TOF_3895.nii.gz`
- Segmentation: `seg_TOF_3895.nii.gz` or `TOF_3895_seg.nii.gz`

```bash
.
├─archive
│  └─log
├─data
│  ├─label (this is where you store the segmentation ground truth)
│  └─train (this is where you store the image data)
├─infer
├─models
├─readme_img
├─saved_image
├─saved_models
│  └─<pre-trained models>
├─train
├─tta
└─library
```

Set the necessary parameters, and then run the script:

- If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images

```bash
python train.py \
--image_path $path_to_images \
--label_path $path_to_labels \
--prep_mode 4 \
--epochs $n_epochs \
--learning_rate 1e-3 \
--output_model $path_to_model
```

- If you set `prep_mode` to 1, 2, or 3, the module applies (1) N4 bias field correction, (2) denoising, or (3) both N4 bias field correction and denoising. You must provide a path for storing the preprocessed images. The following example applies only N4 bias field correction.

```bash
python train.py \
--image_path $path_to_images \
--label_path $path_to_labels \
--prep_mode 1 \
--preprocessed_path $path_to_preprocessed \
--epochs $n_epochs \
--learning_rate 1e-3 \
--output_model $path_to_model
```
