---
title: 'VesselBoost: A Python package for small vessel segmentation in human magnetic resonance angiography data'
tags:
  - Python
  - MRI
  - small vessels
  - test-time adaptation
authors:
  - name: Marshall Xu
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Fernanda L. Ribeiro
    orcid: 0000-0002-1620-4193
    equal-contrib: true 
    corresponding: true 
    affiliation: 1
  - name: Markus Barth
    orcid: 0000-0002-0520-1843
    affiliation: 1
  - name: Michaël Bernier
    affiliation: "2,3"
  - name: Steffen Bollmann
    orcid: 0000-0002-2909-0906
    affiliation: "1,4"
  - name: Soumick Chatterjee
    affiliation: "5,6,7"
  - name: Francesco Cognolato
    affiliation: "8,9"
  - name: Omer Faruk Gulban
    orcid: 0000-0001-7761-3727
    affiliation: "10,11"
  - name: Vaibhavi Itkyal
    affiliation: 12
  - name: Siyu Liu
    affiliation: "1,13"
  - name: Hendrik Mattern
    affiliation: "5,14,15"
  - name: Jonathan R. Polimeni
    orcid: 0000-0002-1348-1179
    affiliation: "2,3,16"
  - name: Thomas B. Shaw
    affiliation: 1
  - name: Oliver Speck
    affiliation: "5,14,15"
  - name: Saskia Bollmann
    orcid: 0000-0001-8242-8008
    corresponding: true 
    affiliation: 1
affiliations:
 - name: School of Electrical Engineering and Computer Science, The University of Queensland, Brisbane, Australia 
   index: 1
 - name: Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital, Charlestown, USA
   index: 2
 - name: Department of Radiology, Harvard Medical School, Boston, USA
   index: 3
 - name: Queensland Digital Health Centre, The University of Queensland, Brisbane, Australia
   index: 4
 - name: Department of Biomedical Magnetic Resonance, Institute of Experimental Physics, Otto-von-Guericke-University, Magdeburg, Germany
   index: 5
 - name: Data and Knowledge Engineering Group, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany
   index: 6
 - name: Genomics Research Centre, Human Technopole, Milan, Italy
   index: 7
 - name: Centre for Advanced Imaging, The University of Queensland, Brisbane, Australia
   index: 8
 - name: ARC Training Centre for Innovation in Biomedical Imaging Technology, The University of Queensland, Brisbane, Australia
   index: 9
 - name: Faculty of Psychology and Neuroscience, Maastricht University, Maastricht, Netherlands
   index: 10
 - name: Brain Innovation, Maastricht, Netherlands
   index: 11
 - name: Department of Biotechnology, Indian Institute of Technology, Madras, India
   index: 12
 - name: Australian eHealth Research Centre, CSIRO, Herston, Australia
   index: 13
 - name: German Center for Neurodegenerative Diseases, Magdeburg, Germany
   index: 14
 - name: Center for Behavioral Brain Sciences, Magdeburg, Germany
   index: 15
 - name: Division of Health Sciences and Technology, Massachusetts Institute of Technology, Cambridge, USA
   index: 16

date: 24 July 2023
bibliography: paper.bib
---

# Summary
*VesselBoost* is a Python-based software package utilizing deep learning techniques to segment high-resolution time-of-flight MRI angiography data with high sensitivity towards small vessels. The software suite encompasses three functional modules: (1) *predict*, (2) *test-time adaptation* (TTA), and (3) *boost*. By leveraging these modules, users can efficiently segment high-resolution time-of-flight data or conveniently 'boost' segmentations for other vascular MRI image contrasts.

One of the distinguishing features of *VesselBoost* lies in the idea of incorporating imperfect training labels for vessel segmentation. At the core of *VesselBoost* is a data augmentation strategy that leverages the self-similarity of large and small vessels, which sensitises a segmentation model towards the smallest vessels. This allows to 'boost' coarse segmentations and increase the number of segmented, small vessels. In summary, *VesselBoost* can provide detailed segmentations of the human brain vasculature from high-resolution MRI angiographic imaging, using either *predict* or *test-time adaptation*, or it can *boost* segmentations for other vascular MRI image contrasts.


# Statement of Need
Magnetic resonance angiography (MRA) performed at ultra-high field provides the unique opportunity to study the arteries of the living human brain at the mesoscopic level [@bollmann_imaging_2022]. From this, we can gain new insights into the brain's blood supply [@hirsch_topology_2012] and vascular disease affecting small vessels [@hetts_pial_2017; @mcconnell_cerebral_2016]. However, to gather quantitative data on human angioarchitecture to — for example — inform modern blood-flow simulations [@park_quantification_2020; @ii_multiscale_2020], detailed segmentations of the smallest vessels are required. 

Several challenges arise when segmenting high-resolution MRA data, most notably the difficulty of obtaining large data sets of correctly and comprehensively labeled data. Thus, *VesselBoost* implements the idea of imperfect training labels [@lucena_convolutional_2019] for vasculature segmentation. At the core of *VesselBoost* is a data augmentation strategy that leverages the self-similarity of large and small vessels, which sensitises a segmentation model towards the smallest vessels, enabling the 'boosting' of coarser, imperfect labels. 

# Methodology
## Overview
*VesselBoost* comprises three modules: 1) *predict*, 2) test-time adaptation (*TTA*), and 3) *boost*. These modules are designed to capture different levels of similarity between the original training data and the new data. If the properties of the new data are close to the original training data, *predict* can be directly applied to the new data. *TTA* will be useful if the new data is somewhat similar, but network adaptation is needed. *Boost* utilises the same pre-processing and data augmentation strategies but trains a new network from scratch. Thus, *boost* caters to cases where the new data is significantly different from the original training data, for example, when using a different vascular MRI contrast.

## Training Data
All models were trained on the SMILE-UHURA challenge dataset [@Chatterjee_Mattern_Dubost_Schreiber_Nürnberger_Speck_2023], which uses the data collected in the StudyForrest project [@hanke_high-resolution_2014]. It consists of 3D multi-slab time-of-flight magnetic resonance angiography (MRA) data acquired at a 7T Siemens MAGNETOM magnetic resonance scanner [@hanke_high-resolution_2014] with an isotropic resolution of 300$\mu$m. Twenty right-handed individuals (21-38 years, 12 males) participated in the original study, but we used the 14 samples for model training where corresponding segmentations were made available through the SMILE-UHURA challenge. 

## Model Architecture
The pre-trained model consists of a 3D U-Net model [@cicek_2016]. We performed several modifications to the proposed 3D U-Net architecture [@cicek_2016], including increased depth (from 3 to 4 layers in both the encoder and decoder blocks), number of input and output channels equal to 1, and number of convolution filters equal to 16. We implemented these modifications to increase the model's ability to learn complex features, classify vessels only, and reduce training time. The models were implemented using Python 3.9 and Pytorch 1.13 [@paszke_automatic_2017]. 

## Training Procedure
Before model training, the MRA data were pre-processed as described below. Data augmentation was performed to increase the amount of training data and to leverage the self-similarity of large and small vessels. The input data is cropped at random locations and sizes at each training epoch and then resized to 64×64×64 using nearest-neighbor interpolation (patch 1). The minimum size for each dimension of the cropped patch is 32, and the maximum is the dimension size of the original image. This procedure is equivalent to zooming in or out for patches smaller or larger than 64×64×64. We generated multiple copies (5 more copies per patch) of each of these patches and applied rotation by 90°, 180°, and 270° (copies 1-3) or blurring using two different Gaussian filters (copies 4 and 5), totalling six copies per patch at each epoch. Four unique patches are generated per training data and training epoch and, with data augmentation, that amounts to 24 images per training sample and epoch (4 patches x 6 copies). We found that by increasing the number of unique patches per training sample and epoch and setting the minimum size for each dimension of the cropped patch to 32, the pre-trained models were more stable across a range of random seeds used to initialize model weights.

We pre-trained three distinct models, each using a specific set of labels: manually corrected labels provided for the challenge and the others using the OMELETTE 1 (O1) and OMELETTE 2 (O2) labels. The OMELLETE labels were generated in an automated fashion [@mattern_2021] using two different sets of parameters. Each model was trained for 1000 epochs at an initial learning rate of 0.001, which was reduced when the loss reached a plateau using ReduceLROnPlateau. The Tversky loss [@salehi_tversky_2017; @chatterjee_ds6_2022] determined the learning objective, with α = 0.3 and β = 0.7.

## VesselBoost Modules

### Module 1: Predict
The prediction pipeline includes input image pre-processing (\autoref{fig:1}a, step i), image segmentation using a pre-trained model (\autoref{fig:1}b, step ii), and post-processing (\autoref{fig:1}b, step iii). This module provides extra flexibility for users to manipulate post-processing parameters to obtain a more suitable proxy before, for example, using it for TTA.

**Pre-processing**: Input images are pre-processed using N4ITK for bias field correction [@tustison_n4itk_2010] and non-local means denoising [@manjon_adaptive_2010] to increase the signal-to-noise ratio (SNR) (\autoref{fig:1}a, step i).

**Post-processing**: The model's output is post-processed to appropriately convert the predicted probabilities to binary classes by setting the threshold to 0.1. Finally, any connected components with a size smaller than ten voxels are removed [@silversmith:2021]. 

### Module 2: Test-time adaptation
Test-time adaptation consists of adapting the weights of pre-trained models using a proxy segmentation to guide parameter optimization (\autoref{fig:1}b, step ii). The user can specify the number of epochs for the model adaptation. The initial learning rate and the loss function have default configurations equal to 0.001 and the Tversky loss (α = 0.3 and β = 0.7), respectively. The default learning rate scheduler is the ReduceLROnPlateau (available through PyTorch [@paszke_automatic_2017]), which automatically reduces the learning rate when the loss reaches a plateau.

### Module 3: Boost
*Boost* (Module 3) allows users to train a segmentation model from scratch using imperfect training labels from a new data set. This module benefits users with access to a small number of labeled data who want to boost small vessel segmentation. This module shares the general training settings previously described, but the user can specify the number of training epochs.

![*VesselBoost* overview. (a) *Predict* allows users to segment high-resolution time-of-flight data using our pre-trained models. (b) The test-time adaptation module allows users to provide a proxy segmentation to drive further adaptation of the pre-trained models. (c) *Boost* allows users to train a segmentation model on a single or more data using existing imperfect training labels. \label{fig:1}](figure1.png)

## Summary
Using our pre-trained models, users can segment high-resolution time-of-flight data with *predict* (Module 1). The *TTA* module (Module 2) allows the user to provide a proxy segmentation, or generate a proxy with our pre-trained model (Module 1), to drive further adaptation of the pre-trained models. We found that TTA, in combination with data augmentation, can improve the segmentation results beyond the training data (i.e., proxies) and increase sensitivity to small vessels. Finally, the *boost* module (Module 3) allows users to train a segmentation model on a new data set using existing imperfect training labels. This module is particularly useful for users with access to a small number of labeled data who want to boost small vessel segmentation. The key difference between *TTA* and *boost* is that the latter allows users to train a model from scratch, whereas the former adapts a pre-trained model.

# Results

## Qualitative evaluation

To qualitatively evaluate our software, we used 3D MRA image slabs with a diverse range of image resolution (from 400 $\mu$m to 150 $\mu$m [@mattern_prospective_2018; @Chatterjee_Mattern_Dubost_Schreiber_Nürnberger_Speck_2023; @bollmann_imaging_2022]). \autoref{fig:2}a shows the maximum intensity projections (MIP) of the original input images, and \autoref{fig:2}b shows the predicted segmentation using *predict* (note that here we use the model trained on manually corrected labels from the SMILE-UHURA challenge). We found that even though our models were trained on lower resolution data, i.e., using 3D multi-slab time-of-flight MRA with an isotropic resolution of 300 $\mu$m, it can be generalized to MRA images with varying resolutions.

![Qualitative evaluation of *Module 1* (*predict*) from **VesselBoost**. (a) Maximum intensity projection of input images with a diverse range of image resolutions. (b) Predicted segmentations. \label{fig:2}](figure2.png)

In cases where no manually corrected data is available, \autoref{fig:3} demonstrates that *TTA* can offer an extra segmentation boost when pre-trained models are trained on automatically generated, imperfect training labels (here, we use the OMELLETE labels [@mattern_2021]). \autoref{fig:3}a shows the MIP of the original input images, and \autoref{fig:3}b the initial segmentation of an OMELETTE-based pre-trained model. Note how this pre-trained model cannot segment the smallest vessel shown in the 'zoomed in' patches. Despite being imperfect, these segmentations can be leveraged as proxy segmentations for *TTA*. Accordingly, we found improved segmentation of smallest vessels (see 400 $\mu$m and 300 $\mu$m images) and improved segmentation continuity (see 160 $\mu$m and 150 $\mu$m images) (\autoref{fig:3}c).

![Qualitative evaluation of *Module 2* (*TTA*) from **VesselBoost**. (a) Maximum intensity projection of input images with a diverse range of image resolutions. (b) Predicted segmentations of an OMELETTE-based pre-trained model. (c) *TTA* based segmentation using the segmentation shown in (b) as proxy segmentation to guide model adaptation. The 'zoomed in' patches show the missing small vessels in panel (b) that are recovered with *TTA*. \label{fig:3}](figure3.png)

\autoref{fig:4} shows the utility of *boost* to train a segmentation model from scratch using imperfect training labels (middle), which were generated simply by thresholding the original image. By leveraging the self-similarity of large and small vessels through data augmentation, it is possible to train a model from scratch using imperfect labels to improve the segmentation of smallest vessel, and hence, improve the segmentation results beyond the training data. Importantly, this occurs so long as the voxel intensity of the smallest vessels is sufficiently similar to the ones from the largest ones.

![Qualitative evaluation of *Module 3* (*boost*) from **VesselBoost**. Maximum intensity projection of input images (left), imperfect segmentation (middle), and predicted segmentation (right) for 160 $\mu$m (panel **a**) and 150 $\mu$m data (panel **b**) using models trained from scratch. The 'zoomed in' patches show the segmentation boost afforded with the *boost* module. \label{fig:4}](figure4.png)

## Quantitative evaluation

To quantitatively evaluate Module 2, *TTA*, we trained a model using 13 MRA images from the SMILE-UHURA challenge [@Chatterjee_Mattern_Dubost_Schreiber_Nürnberger_Speck_2023], leaving one sample out for evaluation. Specifically, we used the coarse segmentations (OMELETTE 2) for model training for 1000 epochs. Then, this pre-trained model was used for proxy generation (analogous to running *predict*) and adapted for 200 epochs using our TTA module and the holdout test sample. To remove unwanted false-positive voxels from outside the brain, a semi-automatically derived brain mask was applied. \autoref{fig:5} shows the coarse segmentation, the ground-truth (or manually annotated) segmentation, the initial segmentation (proxy), and the final segmentation after TTA for the test image. Using the manually annotated segmentation as a reference, the final segmentation after TTA showed an increase in Dice score of 0.04 compared to the initial proxy segmentation. This result demonstrates that imperfect segmentation can be leveraged as proxy segmentations for *TTA*, and *TTA* affords a segmentation boost. 

![Quantitative evaluation of *Module 2* (*TTA*) from **VesselBoost** on 3D MRA image slab with an isotropic resolution of 300 $\mu$m. Dice scores were estimated for the initial segmentation (proxy) and the final segmentation, with the ground truth image as a reference. In green, we show the boost in Dice score after test-time adaptation. \label{fig:5}](figure5.png)

# Discussion and Conclusion
We found that the network trained on low-resolution MRA images performed well on high-resolution data. We also show the potential to train networks on a single data set for vessel segmentation to generate segmentations better than the training data. In addition, test-time adaption allows training on larger data sets with imperfect labels and improves the initial prediction for a new input image.
Note that we always trained and predicted on unmasked images but applied a brain mask afterwards to remove unwanted false positive segmentations stemming from the skin or fat around the skull. When predicting directly on masked images, we noticed that the network would segment the rim of the masked volume as a vessel. Thus, we recommend predicting unmasked images first and then applying a brain mask. Moreover, we use a fixed threshold for removing small disconnected components during post-processing. This threshold can be adapted to cater for very high- or low-resolution images.
In conclusion, VesselBoost has the potential to facilitate the segmentation of small datasets with unique contrasts, such as ex-vivo MRI. VesselBoost can also be combined with existing, traditional segmentation techniques to provide a *boost* to their performance. 

# Code Availability

The VesselBoost tool is freely available at [Open Science Framework](https://osf.io/abk4p/), [GitHub](https://github.com/KMarshallX/vessel_code) and via the
Neurodesk [@renton_neurodesk_2023] data analysis environment (https://neurodesk.github.io) as Docker and Singularity containers. 

# Acknowledgments

The authors acknowledge funding by NHMRC-NIH BRAIN Initiative Collaborative Research Grant APP1117020 and by the NIH NIMH BRAIN Initiative grant R01-MH111419. FLR and Steffen Bollmann acknowledge funding through an ARC Linkage grant (LP200301393). HM acknowledges funding from the Deutsche Forschungsgemeinschaft (DFG) (501214112, MA 9235/3-1) and the Deutsche Alzheimer Gesellschaft (DAG) e.V. (MD-DARS project). MB acknowledges funding from the Australian Research Council Future Fellowship grant FT140100865. This work was initiated with the support of UQ AI Collaboratory.

An early prototype of this work was presented at the 12th Scientific Symposium on Clinical Needs, Research Promises and Technical Solutions in Ultrahigh Field Magnetic Resonance in Berlin in 2021. This work was also submitted to the SMILE-UHURA challenge [@Chatterjee_Mattern_Dubost_Schreiber_Nürnberger_Speck_2023].

# References
