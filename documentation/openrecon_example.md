# **OpenRecon Usage Example**

## **Current Version** - VesselBoost 2.0.1
### **Notes on the latest version**
If you set '--prep_mode' to 1,2 or 3, which means (1) N4 bias field correction, (2) denosing, or (3) both N4 biasfield correction and denoising will happen, then you have to set a path to store the preprocessed images. In the mean time, we also added an option to enable brain extraction ('--enable_brain_extraction') using Synthstrip (from FreeSurfer) to improve the robustness of the preprocessing step. 

**For preprocessing, we recommend setting '--prep_mode' to 1 together with '--enable_brain_extraction'**, which means applying N4 bias field correction and brain extraction, as it is faster and can lead to better performance in some cases. 

If you set '--prep_mode' to 4, which means **no preprocessing** will happen, then you don't have to set a path to store the preprocessed images. Also, there will be **no brain extraction** for this case. \

For patches-based prediction, we added an **optional** feature to use Gaussian blending to reduce edge artifacts and improve the quality of the final segmentation, just add the '--use_blending' flag to turn on this feature. Then You can set the overlap ratio between adjacent patches by setting the '--overlap_ratio' parameter.  

### **Prediction Module**

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