# **OpenRecon Usage Example**

## **Current Version** - VesselBoost 2.0.1
### **Notes on the latest version**
If you set '--prep_mode' to 1,2 or 3, which means (1) N4 bias field correction, (2) denosing, or (3) both N4 biasfield correction and denoising will happen, then you have to set a path to store the preprocessed images. In the mean time, we also added an option to enable brain extraction ('--enable_brain_extraction') using Synthstrip (from FreeSurfer) to improve the robustness of the preprocessing step. 

If you set '--prep_mode' to 4, which means **no preprocessing** will happen, then you don't have to set a path to store the preprocessed images. Also, there will be **no brain extraction** for this case. 

For patches-based prediction, we added a new feature to use Gaussian blending to reduce edge artifacts and improve the quality of the final segmentation.  

### **Prediction Module**
For OpenRecon, there are few configurable options to run the prediction module:

1. Set **"Vessel Boost Modules"** _("id": "vbmodules")_ to "prediction" to run the prediction module.
2. Set **"Preprocessing Mode"** _("id": "vbprepmode")_ to 1, 2, 3 or 4 to select the preprocessing method. We recommend setting it to 1 for applying N4 bias field correction.
3. When you set "Preprocessing Mode" to 1, 2 or 3, you can also choose to enable brain extraction by setting **"Brain extraction flag"** _("id": "vbbrainextraction")_ to true. We recommend enabling this feature.

In this case, the system will run the equivalent command below:
```python
    python prediction.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --output_path "./data/pred_seg" \
    --pretrained "./saved_models/manual_0429" \
    --prep_mode 1 \
    --enable_brain_extraction \
    --use_blending \
    --overlap_ratio 0.5
```

### **TTA Module**
The configurable options for running the TTA module are basically the same, but you have to set TWO MORE parameters for TTA:
1. Set **"Vessel Boost Modules"** _("id": "vbmodules")_ to "tta" to run the TTA module.
2. Set **"Epoch number"** _("id": "vbepochs") to the number of epochs you want to run for TTA. The default value is 200 epochs.
3. Set **"Learning rate"** _("id": "vbrate") to the learning rate you want to use for TTA. The default value is 1e-3.

The following equivalent command will be executed:
```python
    python test_time_adaptation.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --output_path "./data/pred_seg" \
    --pretrained "./saved_models/manual_0429" \
    --prep_mode 1 \
    --enable_brain_extraction \
    --epochs 100 \
    --learning_rate 1e-3 \
    --use_blending \
    --overlap_ratio 0.5
```

### **AngiBoost Module**
*Note: This module was designed to adapt booster module on open recon, while the function is the same as TTA. Might be deprecated in future versions.*

The configurable options for running the AngiBoost module are basically the same as TTA, but you have to set **"Vessel Boost Modules"** _("id": "vbmodules")_ to "booster" to run the AngiBooster module. The following equivalent command will be executed:

```python
    python angiboost.py \
    --image_path "./data/img/" \
    --preprocessed_path "./data/preprocessed/" \
    --pretrained "./saved_models/manual_0429" \
    --label_path "./data/seg/" \ # to store the initial segmentation
    --output_path "./data/boost_seg/" \
    --output_model "./data/boost_seg/boost_model" \
    --prep_mode 1 \
    --enable_brain_extraction \
    --epochs 100 \
    --learning_rate 1e-2 
```