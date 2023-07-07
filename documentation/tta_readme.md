# **Test-Time Adaptation Module**
Test-time adaptation module, using a pretrained model as basis and finetune on your data image. 
## **Inference with pre-trained model**
### **Prepare prerequisite model**
We are currently provide 3 pre-trained models, you can download them to make infetence on your images:\

```bash
# make directory for the pre-trained models
~/vessel_code$ mkdir saved_models

# download the pre-trained model
~/vessel_code$ cd saved_models/
~/vessel_code/saved_models$ osf -p jg7cr fetch /saved_models/Init_ep1000_lr1e3_tver
# now wait for the model to be download and enter tta module directory
```

### **Test-time adpatation without providing proxy segmentation**
You could apply this module directly on your on data without providing the proxy segmentation. This module will automatically generate proxies and finetune the model:
```bash
# Set the necessary parameters
# If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images
$ python 3_test_time_adaptation.py --ds_path <path_to_input_image> --out_path <path_to_output_image> --pretrained <path_to_pretrained_model + model_name> --prep_mode 4 --ep 5000 --lr 1e-3 

# If you set prep_mode to 1,2 or 3, which means both/one of denosing and N4 bias field correction will happen, then you have to set a path to store the preprocessed images
$ python 3_test_time_adaptation.py --ds_path <path_to_input_image> --out_path <path_to_output_image> --ps_path <path_to_preprocessed_image> --pretrained <path_to_pretrained_model + model_name> --prep_mode 4 --ep 5000 --lr 1e-3 

```
### **Test-time adpatation with provided proxy segmentation**
```bash
# Set the necessary parameters
# If you set prep_mode to 4, which means no preprocessing will happen, then you don't have to set a path to store the preprocessed images
$ python 3_test_time_adaptation.py --ds_path <path_to_input_image> --px_path <path_to_proxy_segmentation> --out_path <path_to_output_image> --pretrained <path_to_pretrained_model + model_name> --prep_mode 4 --ep 5000 --lr 1e-3 

# If you set prep_mode to 1,2 or 3, which means both/one of denosing and N4 bias field correction will happen, then you have to set a path to store the preprocessed images
$ python 3_test_time_adaptation.py --ds_path <path_to_input_image> --px_path <path_to_proxy_segmentation> --out_path <path_to_output_image> --ps_path <path_to_preprocessed_image> --pretrained <path_to_pretrained_model + model_name> --prep_mode 4 --ep 5000 --lr 1e-3 

```
