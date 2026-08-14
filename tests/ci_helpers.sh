#!/usr/bin/env bash

# Shared constants and helpers for GitHub Actions integration tests.

CI_HF_MODEL_REPO="BrainVascuLab/VesselBoost"
CI_HF_MODEL_REVISION="2dfcb64056110d819b073ff82934cc54fe3dd773"
CI_HF_BUCKET="BrainVascuLab/vesselboost-ci"
CI_HF_PRIMARY_CHECKPOINT="weights/BM_VB2_aug_all_ep2k_bat_10_0903"
CI_HF_PRIMARY_CHECKPOINT_SHA256="ff51c8e6c79947f13bf7c24ac2ae2242eb4b23b7739e62dc3f8344944bde28f1"

CI_OSF_TEST_IMAGE_URL="https://osf.io/download/zhqbn/"
CI_OSF_TEST_IMAGE_SHA256="f027cc9014a82c36366e5cc837ac2753cf25a2b92178f3575c9e763af56b8212"
CI_OSF_TEST_LABEL_URL="https://osf.io/download/veag4/"
CI_OSF_TEST_LABEL_SHA256="f7d5fe9415b628561812a676877c3f7915e28debe9668e89ff595aee2e9583b1"

ci_assert_sha256() {
    local file_path="$1"
    local expected_sha256="$2"

    printf '%s  %s\n' "$expected_sha256" "$file_path" | sha256sum --check -
}

ci_download_verified_file() {
    local url="$1"
    local destination="$2"
    local expected_sha256="$3"
    local partial_file="${destination}.partial"

    mkdir -p "$(dirname "$destination")"
    if ! curl \
        --fail \
        --location \
        --retry 3 \
        --retry-all-errors \
        --silent \
        --show-error \
        --output "$partial_file" \
        "$url"; then
        rm -f -- "$partial_file"
        return 1
    fi
    mv "$partial_file" "$destination"
    if ! ci_assert_sha256 "$destination" "$expected_sha256"; then
        rm -f -- "$destination"
        return 1
    fi
}

ci_download_test_image() {
    ci_download_verified_file \
        "$CI_OSF_TEST_IMAGE_URL" \
        "$1" \
        "$CI_OSF_TEST_IMAGE_SHA256"
}

ci_download_test_label() {
    ci_download_verified_file \
        "$CI_OSF_TEST_LABEL_URL" \
        "$1" \
        "$CI_OSF_TEST_LABEL_SHA256"
}

ci_download_primary_checkpoint() {
    local local_directory="$1"
    local checkpoint_path="${local_directory}/${CI_HF_PRIMARY_CHECKPOINT}"

    mkdir -p "$local_directory"
    hf download \
        "$CI_HF_MODEL_REPO" \
        "$CI_HF_PRIMARY_CHECKPOINT" \
        --revision "$CI_HF_MODEL_REVISION" \
        --local-dir "$local_directory"
    ci_assert_sha256 "$checkpoint_path" "$CI_HF_PRIMARY_CHECKPOINT_SHA256"
}

ci_verify_readme_clone_command() {
    local repository_root="$1"
    local clone_parent
    local clone_command

    clone_parent="$(mktemp -d /tmp/vesselboost-readme-clone.XXXXXX)"
    if ! clone_command="$(grep -m1 -F 'git clone https://github.com/KMarshallX/VesselBoost.git' "${repository_root}/README.md")"; then
        rm -rf -- "$clone_parent"
        return 1
    fi

    if ! (cd "$clone_parent" && eval "$clone_command"); then
        rm -rf -- "$clone_parent"
        return 1
    fi

    if ! test -d "${clone_parent}/VesselBoost/.git"; then
        rm -rf -- "$clone_parent"
        return 1
    fi
    rm -rf -- "$clone_parent"
}

ci_publish_directory() {
    local source_directory="$1"
    local destination_prefix="$2"

    if [[ "${PUBLISH_CI_RESULTS:-false}" != "true" ]]; then
        echo "[DEBUG]: skipping Hugging Face publication for this event"
        return 0
    fi

    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is required when PUBLISH_CI_RESULTS=true" >&2
        return 1
    fi
    if [[ ! -d "$source_directory" ]]; then
        echo "CI result directory does not exist: $source_directory" >&2
        return 1
    fi
    if [[ "$destination_prefix" != github_actions/* ]]; then
        echo "Refusing unexpected bucket destination: $destination_prefix" >&2
        return 1
    fi

    hf buckets sync \
        "$source_directory" \
        "hf://buckets/${CI_HF_BUCKET}/${destination_prefix}" \
        --delete
}

ci_publish_file() {
    local source_file="$1"
    local destination_path="$2"

    if [[ "${PUBLISH_CI_RESULTS:-false}" != "true" ]]; then
        echo "[DEBUG]: skipping Hugging Face publication for this event"
        return 0
    fi

    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is required when PUBLISH_CI_RESULTS=true" >&2
        return 1
    fi
    if [[ ! -f "$source_file" ]]; then
        echo "CI result file does not exist: $source_file" >&2
        return 1
    fi
    if [[ "$destination_path" != github_actions/* ]]; then
        echo "Refusing unexpected bucket destination: $destination_path" >&2
        return 1
    fi

    hf buckets cp \
        "$source_file" \
        "hf://buckets/${CI_HF_BUCKET}/${destination_path}"
}
