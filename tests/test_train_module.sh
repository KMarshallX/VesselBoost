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
mkdir -p data/images/
mkdir -p data/labels/
mkdir -p data/preprocessed/
mkdir -p data/saved_models/
ci_download_test_image ./data/images/sub-001.nii
ci_download_test_label ./data/labels/sub-001.nii

path_to_images="data/images/"
echo "Path to images: "$path_to_images""

path_to_labels="data/labels/"
echo "Path to labels: "$path_to_labels""

path_to_model="data/saved_models/model_test"
echo "Path to model: "$path_to_model""

path_to_preprocessed="data/preprocessed/"
echo "Path to preprocessed data: "$path_to_preprocessed""

n_epochs=5
echo "Number of epochs: "$n_epochs""

pwd

echo "[DEBUG]: testing train module:"
train_command1="$(ci_extract_markdown_command ./documentation/train_readme.md "python train.py" 1)"
printf '%s\n' "$train_command1"
eval "$train_command1"

train_command2="$(ci_extract_markdown_command ./documentation/train_readme.md "python train.py" 2)"
printf '%s\n' "$train_command2"
eval "$train_command2"

echo "[DEBUG]: publishing current training outputs to Hugging Face"
ci_publish_directory \
    "./data/saved_models" \
    "github_actions/train/saved_model"

echo "Testing done!"
