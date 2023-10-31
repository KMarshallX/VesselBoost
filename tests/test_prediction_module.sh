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
mkdir -p ./data/images/
mkdir -p ./data/predicted_labels/
mkdir -p ./data/preprocessed_imgs/
mkdir ./pretrained_models/

pip install osfclient
osf -p nr6gc fetch /osfstorage/twoEchoTOF/raw/GRE_3D_400um_TR20_FA18_TE7p5_14_sli52_FCY_GMP_BW200_32.nii ./data/images/sub-001.nii
#pretrained model download
osf -p abk4p fetch /osfstorage/pretrained_models/manual_ep1000_1029 ./pretrained_models/manual_ep1000_1029
osf -p abk4p fetch /osfstorage/pretrained_models/om1_ep1000_1029 ./pretrained_models/om1_ep1000_1029
osf -p abk4p fetch /osfstorage/pretrained_models/om2_ep1000_1029 ./pretrained_models/om2_ep1000_1029


path_to_images="./data/images/"
echo "Path to images: "$path_to_images""

path_to_output="./data/predicted_labels/"
echo "Path to output: "$path_to_output""

path_to_preprocessed_images="./data/preprocessed_imgs/"
echo "Path to preprocessed images: "$path_to_preprocessed_images""

path_to_pretrained_model="./pretrained_models/manual_ep1000_1029"
echo "Path to pretrained model: "$path_to_pretrained_model""

echo "[DEBUG]: testing prediction module without preprocessing:"
train_command1=`cat ./documentation/predict_readme.md | grep 'prep_mode 4'`
echo $train_command1
eval $train_command1

echo "[DEBUG]: testing prediction module with preprocessing:"
train_command2=`cat ./documentation/predict_readme.md | grep 'prep_mode 1'`
echo $train_command2
eval $train_command2

echo "[DEBUG]: osf setup"
export OSF_TOKEN=$OSF_TOKEN_
export OSF_USERNAME=$OSF_USERNAME_
export OSF_PROJECT_ID=$OSF_PROJECT_ID_
mkdir -p ~/.osfcli
echo -e "[osf]\nproject = $OSF_PROJECT_ID\nusername = \$OSF_USERNAME" > ~/.osfcli/osfcli.config
cd $path_to_output
for file in ./*; do
    echo $file
    osf -p abk4p remove /osfstorage/github_actions/prediction/predicted_labels/$file
done

echo "[DEBUG]: saving data to osf"
osf -p abk4p upload -r ./ /osfstorage/github_actions/prediction/predicted_labels/

echo "Testing done!"
