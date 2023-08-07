# **Test-Time Adaptation Module**
Test-time adaptation module, using a pretrained model or provided segmentations as a basis and finetune on your image data. 
## **Inference with pre-trained model**
### **Prepare prerequisite model**
We currently provide 3 pre-trained models, you can download them to make inference on your images. The models are ... 

To access them, make a directory for the pre-trained models within the vessel_code folder:
```bash
mkdir ./pretrained_models/
```
Download the pre-trained model from osf:

```bash
osf -p abk4p fetch /osfstorage/pretrained_models/manual_ep5000_0621 ./pretrained_models/manual_ep5000_0621
```


### **Test-time adpatation without providing proxy segmentation**
You could apply this module directly on your on data without providing the proxy segmentation. This module will automatically generate proxies and finetune the model.
If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images. Note that the input path must only contain the nifti images for processing and be different to the output path.

```bash
python test_time_adaptation.py --ds_path $path_to_images --out_path $path_to_output --pretrained $path_to_pretrained_model --prep_mode 4 --ep $n_epochs --lr 1e-3 
```

If you set prep_mode to 1,2 or 3, which means (1) N4 bias field correction, (2)denosing, or (3) both N4 biasfield correction and denoising will be applied to the input images, then you have to set a path to store the preprocessed images. In the following example, we set the preprocessing mode to "applying N4 bias field correction only (mode 1)":

```bash
python test_time_adaptation.py --ds_path $path_to_images --out_path $path_to_output --ps_path $path_to_preprocessed_images --pretrained $path_to_pretrained_model --prep_mode 1 --ep $n_epochs --lr 1e-3 
```
### **Test-time adpatation with provided proxy segmentation**
You can provide a proxy segmentation which will be used for the test-time-adaption instead of the automatically generated proxies. This segmentation can be created using any methods, for example manual labelling or other, non-deep-learning methods. We will use the filenames to match the raw images and proxy segmentations. If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images. 

```bash
python test_time_adaptation.py --ds_path $path_to_images --px_path $path_to_proxy_labels --out_path $path_to_output --pretrained $path_to_pretrained_model --prep_mode 4 --ep $n_epochs  --lr 1e-3
```
 If you set prep_mode to 1,2 or 3, which means (1) N4 bias field correction, (2)denosing, or (3) both N4 biasfield correction and denoising will happen, then you have to set a path to store the preprocessed images. In the following example, we set the preprocessing mode to "applying N4 bias field correction only (mode 1)":

```bash
python test_time_adaptation.py --ds_path $path_to_images --px_path $path_to_proxy_labels --out_path $path_to_output --ps_path $path_to_preprocessed_images --pretrained $path_to_pretrained_model --prep_mode 1 --ep $n_epochs --lr 1e-3 
```
