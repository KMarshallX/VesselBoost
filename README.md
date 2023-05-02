# **Vessel Boost**
This repository contains functional modules and stand-alone scripts for Vessel Boost.
## **Table of Contents**
- [Current Version](https://github.com/KMarshallX/vessel_code#current-version)
- [Installation & Requirements](https://github.com/KMarshallX/vessel_code#installation--requirements)
- [Purpose](https://github.com/KMarshallX/vessel_code#purpose)
- [Feature Overview](https://github.com/KMarshallX/vessel_code#feature-overview)
- [Citation](https://github.com/KMarshallX/vessel_code#citation)
- [Contact](https://github.com/KMarshallX/vessel_code#contact)
## **Current Version**
VesselBoost 0.9.0
## **Installation & Requirements**
This is a Pytorch based project, for successfully running this project on your local machine, please follow the following steps to set up necessary sofware environment.
1. Install a Conda environment with python version of 3.9.15 or later version (find Conda installation guidance [here](https://docs.anaconda.com/free/anaconda/install/index.html)).
2. Install the Pytorch framework with version of 1.13.1 (you can also find guidance [here](https://pytorch.org/get-started/locally/)).
3. Clone this repository to your local machine
    ```
    $ git clone git@github.com:KMarshallX/vessel_code.git
    ```
    or 
    ```
    $ git clone https://github.com/KMarshallX/vessel_code.git
    ```
4. Then set your current working directory as the cloned repository, and install the remaining required packages
    ```
    $ cd ./vessel_code
    $ pip install -r requirements.txt
    ```
Done. Happy tuning! :innocent:

## **Purpose**
...
## **Feature Overview**
### *Pipeline overview*
<p align="center">
<img src="./readme_img/overall_flowchart_2.png">
</p>
The complete pipeline will firstly train an intial model on the provided high resolution MRAs, or you can directly use our pretrained models. The pretrained models will be used to infer intermediate segmentations (proxies) of the images you want to process. Lastly, a test-time-adaptation (TTA) process will conduct based on the chosen pretrained model and the proxies it generates.

### *Initial training module*
You can use this module to train your initial model for TTA with your own datasets. For detailed description of this module, please see [this page](https://github.com/KMarshallX/vessel_code/tree/master/train)

### *Inference module*
...
For detailed description of this module, please see [this page](https://github.com/KMarshallX/vessel_code/tree/master/infer)

### *Test-time-adaptation module*
...
For detailed description of this module, please see [this page](https://github.com/KMarshallX/vessel_code/tree/master/tta)

## **Citation**
...

## **Contact**
...

