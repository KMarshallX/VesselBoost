# **Test-Time Adaptation Module**
Test-time adaptation module, using a pretrained model or provided segmentations as a basis and finetune on your image data. 
## **Inference with pre-trained model**
### **Prepare prerequisite model**
We currently provide 3 pre-trained models, you can download them to make inference on your images. The models are ... 

To access them, make a directory for the pre-trained models within the vessel_code folder:
```bash
~/vessel_code$ mkdir saved_models
```
Download the pre-trained model from osf:
```
~/vessel_code$ cd saved_models/
~/vessel_code/saved_models$ osf -p jg7cr fetch /saved_models/Init_ep1000_lr1etver
```
Now wait for the models to be downloaded and change back to vessel_code directory:
```bash
~/vessel_code/saved_models$ ..
~/vessel_code$
```

### **Test-time adpatation without providing proxy segmentation**
You could apply this module directly on your on data without providing the proxy segmentation. This module will automatically generate proxies and finetune the model.
If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images. Note that the input path must only contain the nifti images for processing and be different to the output path.

```bash
~/vessel_code/$ python test_time_adaptation.py --ds_path <path_to_input_image_folder> --out_path <path_to_output_image_folder> --pretrained <path_to_pretrained_model> --prep_mode 4 --ep 5000 --lr 1e-3 
```

If you set prep_mode to 1,2 or 3, which means both(1), denosing(2) or N4 bias field correction(3) will be applied to the input images, then you have to set a path to store the preprocessed images.
```bash
$ python test_time_adaptation.py --ds_path <path_to_input_image> --out_path <path_to_output_image> --ps_path <path_to_preprocessed_image> --pretrained <path_to_pretrained_model + model_name> --prep_mode 4 --ep 5000 --lr 1e-3 

```
### **Test-time adpatation with provided proxy segmentation**
You can provide a proxy segmentation which will be used for the test-time-adaption instead of the automatically generated proxies. This segmentation can be created using any methods, for example manual labelling or other, non-deep-learning methods. If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images. 
```bash
~/vessel_code$ python test_time_adaptation.py --ds_path <path_to_input_image_folder> --px_path <path_to_proxy_segmentation_folder> --out_path <path_to_output_image_folder> --pretrained <path_to_pretrained_model> --prep_mode 4 --ep 5000 --lr 1e-3
```
 If you set prep_mode to 1,2 or 3, which means both(1), denosing(2) or N4 bias field correction(3) will happen, then you have to set a path to store the preprocessed images.
```bash
$ python test_time_adaptation.py --ds_path <path_to_input_image_folder> --px_path <path_to_proxy_segmentation_folder> --out_path <path_to_output_image_folder> --ps_path <path_to_preprocessed_image_folder> --pretrained <path_to_pretrained_model> --prep_mode 4 --ep 5000 --lr 1e-3 
```
