#!/usr/bin/env bash
set -e

cp -r . /tmp/vessel_code

# test readme
echo "[DEBUG]: testing the clone command from the README:"
clone_command=`cat /tmp/vessel_code/README.md | grep https://github.com/KMarshallX/vessel_code.git`
echo $clone_command
$clone_command

echo "[DEBUG]: testing the miniconda installation from the README:"
get_command=`cat /tmp/vessel_code/README.md | grep miniconda-setup.sh`
echo $get_command
$get_command 

export PATH="/home/runner/miniconda3/bin:$PATH"
source ~/.bashrc

echo "[DEBUG]: testing the conda env build from the README:"
cd vessel_code
condaenv_command=`cat ./README.md | grep environment.yml`
echo $condaenv_command
$condaenv_command

# conda activate in a bash script
source /home/runner/miniconda3/bin/activate
conda init bash

echo "[DEBUG]: testing conda activate command from the README:"
condact_command=`cat ./README.md | grep activate`
echo $condact_command
$condact_command

# settings for data download
mkdir -p data/images/
mkdir -p data/labels/
mkdir -p data/preprocessed/
mkdir models
pip install osfclient
osf -p nr6gc fetch /osfstorage/twoEchoTOF/raw/GRE_3D_400um_TR20_FA18_TE7p5_14_sli52_FCY_GMP_BW200_32_e2.nii.gz ./dataosf/images/sub-001.nii.gz
osf -p nr6gc fetch /osfstorage/twoEchoTOF/seg/seg_GRE_3D_400um_TR20_FA18_TE7p5_14_sli52_FCY_GMP_BW200_32_biasCor_H75_L55_C10.nii dataosf/images/sub-001.nii.gz

path_to_images="data/images/"
path_to_labels="data/labels/"
path_to_model="models/model_test"
path_to_preprocessed="data/preprocessed/"
n_epochs=5

echo "[DEBUG]: testing train module:"
train_command1=`cat ./documentation/train_readme.md | grep --prep_mode 4`
echo $train_command1
$train_command1

train_command2=`cat ./documentation/train_readme.md | grep --prep_mode 1`
echo $train_command2
$train_command2