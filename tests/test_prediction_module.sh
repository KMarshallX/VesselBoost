#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tests/ci_helpers.sh"
cd "$repo_root"

# test readme
echo "[DEBUG]: testing the clone command from the README:"
ci_verify_readme_clone_command "$repo_root"

echo "[DEBUG]: testing the miniconda installation from the README:"
get_command="$(grep -m1 'miniconda-setup.sh' README.md)"
echo "$get_command"
eval "$get_command"

export PATH="/home/runner/miniconda3/bin:$PATH"
source ~/.bashrc

echo "[DEBUG]: testing the conda env build from the README:"
condaenv_command="$(grep -m1 'conda env create -f environment-ci.yml' README.md)"
echo "$condaenv_command"
eval "$condaenv_command"

# conda activate in a bash script
source /home/runner/miniconda3/bin/activate
conda init bash

echo "[DEBUG]: testing conda activate command from the README:"
condact_command="$(grep -m1 'conda activate vessel_boost_ci' README.md)"
echo "$condact_command"
eval "$condact_command"

# settings for data download
mkdir -p ./data/images/
mkdir -p ./data/predicted_labels/
mkdir -p ./data/preprocessed_imgs/
mkdir -p ./pretrained_models/

ci_download_test_image ./data/images/sub-001.nii
ci_download_primary_checkpoint ./pretrained_models



path_to_images="./data/images/"
echo "Path to images: "$path_to_images""

path_to_output="./data/predicted_labels/"
echo "Path to output: "$path_to_output""

path_to_preprocessed_images="./data/preprocessed_imgs/"
echo "Path to preprocessed images: "$path_to_preprocessed_images""

path_to_pretrained_model="./pretrained_models/weights/BM_VB2_aug_all_ep2k_bat_10_0903"
echo "Path to pretrained model: "$path_to_pretrained_model""

echo "[DEBUG]: testing prediction module without preprocessing:"
train_command1=`cat ./documentation/predict_readme.md | grep 'prep_mode 4'`
echo $train_command1
eval $train_command1

echo "[DEBUG]: testing prediction module with preprocessing:"
train_command2=`cat ./documentation/predict_readme.md | grep 'prep_mode 1'`
echo $train_command2
eval $train_command2

echo "[DEBUG]: publishing current prediction outputs to Hugging Face"
ci_publish_directory \
    "$path_to_output" \
    "github_actions/prediction/predicted_labels"

echo "Testing done!"
