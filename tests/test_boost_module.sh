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
mkdir -p ./data/labels/
mkdir -p ./data/preprocessed/
mkdir -p ./data/predicted_labels/

ci_download_test_image ./data/images/sub-001.nii
ci_download_test_label ./data/labels/sub-001.nii

path_to_images="./data/images/"
echo "Path to images: "$path_to_images""

path_to_labels="./data/labels/"
echo "Path to labels: "$path_to_labels""

path_to_output="./data/predicted_labels/"
echo "Path to output: "$path_to_output""

path_to_scratch_model="./data/predicted_labels/model_test"
echo "Path to model: "$path_to_scratch_model""

path_to_preprocessed_images="./data/preprocessed/"
echo "Path to preprocessed data: "$path_to_preprocessed_images""

n_epochs=5
echo "Number of epochs: "$n_epochs""

echo "[DEBUG]: testing boost module:"
boost_command1="$(ci_extract_markdown_command ./documentation/boost_readme.md "python boost.py" 1)"
ci_assert_gaussian_blending_command "$boost_command1"
printf '%s\n' "$boost_command1"
eval "$boost_command1"

boost_command2="$(ci_extract_markdown_command ./documentation/boost_readme.md "python boost.py" 2)"
ci_assert_gaussian_blending_command "$boost_command2"
printf '%s\n' "$boost_command2"
eval "$boost_command2"

echo "[DEBUG]: publishing current boost outputs to Hugging Face"
ci_publish_directory \
    "$path_to_output" \
    "github_actions/boost/predicted_labels"

echo "Testing done!"
