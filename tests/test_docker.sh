#!/usr/bin/env bash
set -e

cd /opt/vessel_code

# settings for data download
mkdir -p data/images/
mkdir -p data/labels/
mkdir -p data/preprocessed/
pip install osfclient
osf -p nr6gc fetch /osfstorage/twoEchoTOF/raw/GRE_3D_400um_TR20_FA18_TE7p5_14_sli52_FCY_GMP_BW200_32.nii ./data/images/sub-001.nii
osf -p nr6gc fetch /osfstorage/twoEchoTOF/seg/seg_GRE_3D_400um_TR20_FA18_TE7p5_14_sli52_FCY_GMP_BW200_32_biasCor_H75_L55_C10.nii ./data/labels/sub-001.nii

path_to_images="data/images/"
echo "Path to images: "$path_to_images""

path_to_labels="data/labels/"
echo "Path to labels: "$path_to_labels""

path_to_model="saved_models/model_test"
echo "Path to model: "$path_to_model""

path_to_preprocessed="data/preprocessed/"
echo "Path to preprocessed data: "$path_to_preprocessed""

n_epochs=5
echo "Number of epochs: "$n_epochs""

pwd

echo "[DEBUG]: testing train module:"
train_command1=`cat ./documentation/train_readme.md | grep 'prep_mode 4'`
echo $train_command1
eval $train_command1

echo "[DEBUG]: osf setup"
export OSF_TOKEN=$OSF_TOKEN_
export OSF_USERNAME=$OSF_USERNAME_
export OSF_PROJECT_ID=$OSF_PROJECT_ID_
mkdir -p ~/.osfcli
echo -e "[osf]\nproject = $OSF_PROJECT_ID\nusername = \$OSF_USERNAME" > ~/.osfcli/osfcli.config
cd saved_models
osf -p abk4p remove /osfstorage/github_actions/docker/saved_model/model_test


echo "[DEBUG]: saving data to osf"
osf -p abk4p upload ./model_test /osfstorage/github_actions/docker/saved_model/model_test

echo "Testing done!"
