---
title: 'VesselBoost: A Python package for small vessel segmentation in human magnetic resonance angiography data'
tags:
  - Python
  - MRI
  - small vessels
  - test-time adaptation
authors:
  - name: Ke (Marshall) Xu
    orcid: TODO
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Fernanda L. Ribeiro
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name:  
    affiliation: 1
  - name:  Saskia Bollmann
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: School of Electrical Engineering and Computer Science, The University of Queensland, Brisbane, Australia 
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 24 July 2023
bibliography: paper.bib

---

# Summary



# Statement of need


# Methodology

Our *VesselBoost* application comprises of three modules: 1) deep learning model training, 2) inference with our pre-trained models, and 3) test-time adaptation (TTA). At the core of *VesselBoost* is our test-time adaptation module (Module 3) that allows the user to use their initial segmentation or the one generated with our pre-trained model (Module 2) to drive further adaptation of the provided pre-trained model (Figure 1a). In brief, we found that TTA in combination with data augmentation can improve the segmentation results beyond the training data (i.e., proxies) and increase sensitivity to small vessels. Specifically, our pre-trained model consists of a 3D U-Net model [@cicek_2016] initially trained on the SMILE-UHURA challenge 'train' set and one sample of the 'validate' set (sub-007). We performed several modifications to the proposed 3D U-Net architecture [@cicek_2016], including increased depth (from 3 to 4 layers in both the encoder and decoder blocks), number of input and output channels equal to 1, and number of convolution filters equal to 16. These modifications were implemented to increase the model's ability to learn complex features, to classify vessels only, and to reduce training time. Our models were implemented using Python 3.9 and Pytorch 1.13 [REF]. Note, however, that our approach is flexible, allowing other developers to constribute with new model architectures and new pre-trained models. 

<p align="center">
<img src="./figure1_v1.png">
</p>


## Module 1: Deep learning model training
**Training data**: We pre-trained our models using the SMILE-UHURA challenge dataset (https://www.soumick.com/en/uhura/). This dataset was collected as part of the StudyForrest project [@forstmann_multi-modal_2014] and are 3D multi-slab time-of-flight magnetic resonance angiography data acquired at a 7T Siemens MAGNETOM magnetic resonance scanner [@hanke_high-resolution_2014] with an isotropic resolution of 300$\mu$. Twenty right-handed participants (21-38 years, 12 males) participanted in the study, but we used XXX samples for model training. 

**Pre-processing**: Input images are pre-processed using N4ITK for bias field correction [@tustison_n4itk_2010] and non-local means denoising [@manjon_adaptive_2010] to increase the SNR. 

**Data augmentation**: Our model was pre-trained using randomly cropped patches from the input images and their corresponding labels. Each patch was subjected to several augmentation steps to create six differently modified training patches: At each training epoch, the training data were cropped at a random location and size and then resized to 64×64×64 using nearest-neighbor interpolation (patch 1). This procedure is equivalent to zooming in or out for patches smaller or larger than 64×64×64, respectively. On patch 1, we also applied rotation by 90, 180, and 270° (patches 2 – 4) or applied blurring using two different Gaussian filters (patches 5 and 6). In total, 78,000 training patches were generated (6 patches x 13 subjects x 1,000 training epochs). 

**Training**: We pre-trained three distinct models, each using a specific set of labeled data: one using the ground-truth labels provided for the challenge and the two others using the OMELETTE 1 (O1) and OMELETTE 2 (O2) labels. The OMELLETE labels were generated in an automated fashion [REF] using two different sets of parameters. Each model was trained for 1000 epochs at an initial learning rate of 0.001, which was reduced when the loss reached a plateau (ReduceLROnPlateau [REF]). The Tversky loss [@salehi_tversky_2017] [@chatterjee_ds6_2022] determined the learning objective with α = 0.3 and β = 0.7. 

**Post-processing**: The model's output is then post-processed to appropriately convert the predicted probabilities to binary classes by setting the threshold to 0.1. Finally, any connected components with a size smaller than ten voxels are removed [@silversmith:2021]. 

 
## Module 2: Inference with pre-trained model
Our inference pipeline includes pre-processing the input image (Figure 1b, step i), generating an initial segmentation using our pre-trained model (Figure 1b, step ii), and post-processing the predictions (Figure 1b, step iii). This segmentation can be further adjusted with test-time adaptation by using it as a proxy. 

## Module 3: Test-time adaptation
Test-time adaptation consists of adapting the weights of our pre-trained models using a proxy segmentation as labels to guide parameter optimization (Figure 1a, step ii). The user can specify the number of epochs for the model adaptation. The initial learning rate is set to 0.001 using the Tversky loss (α = 0.3 and β = 0.7) and the learning rate scheduler ReduceLROnPlateau. Note that these parameters can be altered given the user's requirements.

# Code Availability
The VesselBoost tool is freely available at (https://osf.io/abk4p/) and via the
Neurodesk data analysis environment (https://neurodesk.github.io) or as a Docker/Singularity containers (https://github.com/KMarshallX/vessel_code)


# Acknowledgements

The authors acknowledge funding by NHMRC-NIH BRAIN Initiative Collaborative Research Grant APP1117020 and by the NIH NIMH BRAIN Initiative grant R01-MH111419. FLR acknowledges funding through an ARC Linkage grant (LP200301393). MB acknowledges funding from Australian Research Council Future Fellowship grant FT140100865.

# References