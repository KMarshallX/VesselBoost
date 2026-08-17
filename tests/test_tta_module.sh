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
mkdir -p ./data/proxy_labels/
mkdir -p ./data/preprocessed_imgs/
mkdir -p ./pretrained_models/

ci_download_test_image ./data/images/sub-001.nii
echo "[DEBUG]: testing model's weights download:"
grep -F 'hf download BrainVascuLab/VesselBoost' ./documentation/tta_readme.md
ci_download_primary_checkpoint ./pretrained_models

path_to_images="./data/images/"
echo "Path to images: "$path_to_images""

path_to_output="./data/predicted_labels/"
echo "Path to output: "$path_to_output""

path_to_proxy_labels="./data/proxy_labels/"
echo "Path to proxy labels: "$path_to_proxy_labels""

path_to_preprocessed_images="./data/preprocessed_imgs/"
echo "Path to preprocessed images: "$path_to_preprocessed_images""

path_to_pretrained_model="./pretrained_models/weights/BM_VB2_aug_all_ep2k_bat_10_0903"
echo "Path to pretrained model: "$path_to_pretrained_model""

n_epochs=5
echo "Number of epochs: "$n_epochs""

echo "[DEBUG]: testing tta without a proxy and no preprocessing:"
tta_command1="$(ci_extract_markdown_command ./documentation/tta_readme.md "python test_time_adaptation.py" 1)"
ci_assert_gaussian_blending_command "$tta_command1"
printf '%s\n' "$tta_command1"
eval "$tta_command1"

echo "[DEBUG]: testing tta without a proxy and including preprocessing:"
tta_command2="$(ci_extract_markdown_command ./documentation/tta_readme.md "python test_time_adaptation.py" 2)"
ci_assert_gaussian_blending_command "$tta_command2"
printf '%s\n' "$tta_command2"
eval "$tta_command2"


echo "[DEBUG]: testing tta with a proxy and no preprocessing:"
tta_command3="$(ci_extract_markdown_command ./documentation/tta_readme.md "python test_time_adaptation.py" 3)"
ci_assert_gaussian_blending_command "$tta_command3"
printf '%s\n' "$tta_command3"
eval "$tta_command3"

echo "[DEBUG]: testing tta with a proxy and including preprocessing:"
tta_command4="$(ci_extract_markdown_command ./documentation/tta_readme.md "python test_time_adaptation.py" 4)"
ci_assert_gaussian_blending_command "$tta_command4"
printf '%s\n' "$tta_command4"
eval "$tta_command4"

echo "[DEBUG]: publishing current TTA outputs to Hugging Face"
ci_publish_directory \
    "$path_to_output" \
    "github_actions/tta/predicted_labels"

echo "Testing done!"
