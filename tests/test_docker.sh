#!/usr/bin/env bash
set -eo pipefail

checkout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${checkout_root}/tests/ci_helpers.sh"

cd /opt/VesselBoost

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
train_command1=`cat ./documentation/train_readme.md | grep 'prep_mode 4'`
echo $train_command1
eval $train_command1

if [[ "${PUBLISH_CI_RESULTS:-false}" == "true" ]]; then
    python -m pip install --quiet --upgrade "huggingface_hub==1.27.0"
fi

echo "[DEBUG]: publishing current Docker-test model to Hugging Face"
ci_publish_file \
    "$path_to_model" \
    "github_actions/docker/saved_model/model_test"

echo "Testing done!"
