# Vessel Code
This repository is only used for storing code, not data/saved images/pretrained models.

## Runnable in CLI
![cli](./readme_img/1.jpg)

## The model is debugged, but needs further optimization
The vessels can be distinguished from the output:\
![ves1](./readme_img/new_slice_13.jpg)
![gt](./readme_img/gt.jpg)

However, the output is not best:\
![ves2](./readme_img/new_slice_62.jpg)\

## Week 3 update:
1. Implemented augmentation (rotation for now)
2. Implemented dice loss (testing)
3. Using slab pool as training set

## Current problems:
1. Zooming of single input patch (e.g. 64*64*54 -> 64*64*64)
2. Optimal epoch number and learning rate
3. https://drive.google.com/drive/u/0/folders/1Ty5B-qa339aCOvK15aeBKttGOmxtRzII

## Week 4 update:
(There was a powerpoint slide)

## Week 5 update
1. New model - Atrous Spatial Pyramid Pooling CNN!
ASPP network handles objects at multiple scales to capture multiscale information on the basis of parallel atrous-based CNN layers by using multiple atrous rates.\
It worked surprisingly good. The following output image is from model trained on only one slab.\
![aspp](./readme_img/asppcnn.jpg)
2. Now the patches are fed into the model in batches!
By using batches, the efficiency of the whole training process has been improved - the training time has shortend, much faster! The accuracy hasn't been affected very much. However, this is not the reason why the model worked when it's trained on one or two slabs but failed for large dataset.(The following output image is from model trained on only one slab.)\
![batch](./readme_img/batch.jpg)
3. (From this bullet point are the brief working log in week 5)
4. (Fernanda's suggestion) Increase the slab pool one slab at a time - When the model is trained on 1 slab, it worked; 2 slabs, still worked; 3,4,5.. slabs, failed. The learning rate was tuned lower when the dataset increased.
5. (Saskia's first suggestion) Check the dataloader - the dataloader worked fine, I rewrote the dataloader to make it feed the data patches in batches, still worked fine, the image and seg matches
6. (Saskia's second suggestion) Load the data patches in batches - it shortened the training time, but has little effect on loss/accuracy of the output image.
7. Add a new optimizer - Stochastic Gradient Descent, it changes variables slowly, slower than Adam. No help.
![adam](./readme_img/adam.jpg)
![sgd](./readme_img/sgd.jpg)
8. Implemented a learning rate scheduler - it can changes the learning rate as per number of epochs
### Confusion
The loss value dropped quickly when the model is training on 1/2 slab(s), but it plateaued at a large loss value for large dataset, and then decreased very slowly.
![1img](./readme_img/1img.jpg)
![4img](./readme_img/4imgs.jpg)
### One assumption

## Week 6 update
**Myth has been solved! The project can go into the next stage!**
![out](./readme_img/out.jpg)

## Week 7 update
some minor changes:
1. global standardisation has been changed back to standardisation of each slab
2. make sure the output segmentation slab has the same slices as the input slab

## Week 8 update
1. Updated some changes to fit the pipeline to the challenge dataset
(implemented a new feature to check the dataset's folder, etc)
2. Now the maximum intensity projection will be generated with the .nii segmentation image
3. A new model : **Single level UNet3D with multipath residual attention block**\
The referenced literature: [paper](https://reader.elsevier.com/reader/sd/pii/S1319157822001069?token=37835F8506108DE8811590CEACADF29905A54BE637DF08974C425E95FB921EC7648913850A7BC48DD533C4D2945C5D42&originRegion=us-east-1&originCreation=20230131144819)
4. Tested the generalizability of the new model.\

### Advantages of the new model:
1. light-weighted
2. good performance on small objects (**this needs further investigation**)
3. Below attached the output results of the new model:\
![mip_4](./readme_img/mip_4.png)
![new_4](./readme_img/val_AtrousUnet_ep10_lr1e4_1slab.png)\
![mip_5](./readme_img/mip_5.png)
![new_5](./readme_img/val_AtrousUnet_ep10_lr1e4_1slab_5.png)\
![mip_6](./readme_img/mip_6.png)
![new_6](./readme_img/val_AtrousUnet_ep10_lr1e4_1slab_6.png)\
![mip_20](./readme_img/mip_20.png)
![new_20](./readme_img/val_AtrousUnet_ep10_lr1e4_1slab_20.png)\

4. Unet3d does not perform well on the challenge dataset (**this needs further investigation**)
More details are missing than the new model\
![mip_4](./readme_img/mip_4.png)
![unet_4](./readme_img/val_Unet_ep10_lr1e3_1slab.png)\


## Week 9 update
**1. Changed the data loading pipeline**\
_Was_ -- The raw image slabs were pathified into fixed-size patches (64,64,64), augmented, then fed into the training pipeline\
_Now_ -- The raw image slabs were resampled for fixed times (epochs), for each time the slab was randomly cropped and resized to a fixed size (64,64,64), augmented, then fed into the training pipeline\
**2. Some parts of the code was re-wrote**\
e.g. The training pipeline:\
```
for single_image_slab in the folder:
    for epoch in range(total_epoch):
        patch = RandomCrop(single_image_slab)
        input_patch = Augmentation(patch)
        output = Training_Model(input_patch)
```
**3. Global standardisation of slab pool changed back to standardising each slab**\
**4. The attempt to use the pre-trained model failed, as it led to memory overflowing errors**\
**5. Fixing bugs**

### Below attached output results of the new pipeline\
**sub004**\
_Ground Truth_\
![mip_4](./readme_img/mip_4.png)\
_Unet 3D_\
![unet_4](./readme_img/Newtrain_unet_ep5000_lr1e3_withAUG_4.png)\
_ASPP_\
![aspp_4](./readme_img/Newtrain_aspp_ep5000_lr1e3_withAUG_4.png)\
_Atrous_\
![atrs_4](./readme_img/Newtrain_atrous_ep5000_lr1e3_withAUG_4.png)\

**sub005**\
_Ground Truth_\
![mip_5](./readme_img/mip_5.png)\
_Unet 3D_\
![unet_5](./readme_img/Newtrain_unet_ep5000_lr1e3_withAUG_5.png)\
_ASPP_\
![aspp_5](./readme_img/Newtrain_aspp_ep5000_lr1e3_withAUG_5.png)\
_Atrous_\
![atrs_5](./readme_img/Newtrain_atrous_ep5000_lr1e3_withAUG_5.png)\

**sub018**\
_Ground Truth_\
![mip_18](./readme_img/mip_18.png)\
_Unet 3D_\
![unet_18](./readme_img/Newtrain_unet_ep5000_lr1e3_withAUG_18.png)\
_ASPP_\
![aspp_18](./readme_img/Newtrain_aspp_ep5000_lr1e3_withAUG_18.png)\
_Atrous_\
![atrs_18](./readme_img/Newtrain_atrous_ep5000_lr1e3_withAUG_18.png)\

**sub020**\
_Ground Truth_\
![mip_20](./readme_img/mip_20.png)\
_Unet 3D_\
![unet_20](./readme_img/Newtrain_unet_ep5000_lr1e3_withAUG_20.png)\
_ASPP_\
![aspp_20](./readme_img/Newtrain_aspp_ep5000_lr1e3_withAUG_20.png)\
_Atrous_\
![atrs_20](./readme_img/Newtrain_atrous_ep5000_lr1e3_withAUG_20.png)\


### Current problems
1. How to further improve the performance of the model / quality of the output image
2. How to check the continuity of the vessels in the output image without medical expertise  (3D?) 







