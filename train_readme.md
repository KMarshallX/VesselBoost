# **Initial Training Module**
You can use this module to train your own base model.
## **Create a base modle from scratch**
### Prepare the training data
If you are working outside of a container, you can store you data under *data* folder as the instruction below. If you are working inside a container, you can mount your data folder to *./data/train/* and  *./data/label/*.\
Please make sure that the name of a segmentation image file should contain the FULL NAME of its corresponding MRI image. \
e.g.:\
[raw image ->] TOF_3895.nii.gz\
[segmentation image ->] seg_TOF_3895.nii.gz or TOF_3895_seg.nii.gz, just make sure it contains the "TOF_3895".


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
└─utils
```

### Then run the script:
```bash
# Move from ./vessel_code to ./vessel_code/train
$ cd train

# Set the necessary parameters
# If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images
$ python train.py --inimg <path_to_input_image> --inlab <path_to_input_label> --prep_mode 4 --ep 5000 --lr 1e-3 --outmo <path_to_output_model>

# If you set prep_mode to 1,2 or 3, which means both/one of denosing and N4 bias field correction will happen, then you have to set a path to store the preprocessed images
$ python train.py --inimg <path_to_input_image> --inlab <path_to_input_label> --prep_mode 1 --ps_path <path_to_preprocessed_image> --ep 5000 --lr 1e-3 --outmo <path_to_output_model>

```
