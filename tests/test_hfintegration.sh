#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tests/ci_helpers.sh"
cd "$repo_root"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for the Hugging Face bucket integration test" >&2
    exit 1
fi

run_id="${GITHUB_RUN_ID:-manual}"
run_attempt="${GITHUB_RUN_ATTEMPT:-1}"
remote_path="github_actions/smoke/${run_id}-${run_attempt}-figure1.png"
remote_uri="hf://buckets/${CI_HF_BUCKET}/${remote_path}"
smoke_directory="$(mktemp -d /tmp/vesselboost-hf-smoke.XXXXXX)"
remote_uploaded=0

cleanup_smoke_test() {
    local cleanup_status=0

    if [[ "$remote_uploaded" -eq 1 ]]; then
        hf buckets rm "$remote_uri" --yes || cleanup_status=$?
    fi
    if [[ "$smoke_directory" == /tmp/vesselboost-hf-smoke.* && -d "$smoke_directory" ]]; then
        rm -rf -- "$smoke_directory"
    fi
    return "$cleanup_status"
}
trap cleanup_smoke_test EXIT

source_file="${repo_root}/figures/figure1.png"
downloaded_file="${smoke_directory}/anonymous-download.png"
anonymous_hf_home="${smoke_directory}/anonymous-hf-home"
mkdir -p "$anonymous_hf_home"

echo "[DEBUG]: uploading bucket smoke-test object"
hf buckets cp "$source_file" "$remote_uri"
remote_uploaded=1

echo "[DEBUG]: downloading bucket object anonymously"
env -u HF_TOKEN \
    HF_HOME="$anonymous_hf_home" \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    hf buckets cp "$remote_uri" "$downloaded_file"

ci_assert_sha256 \
    "$downloaded_file" \
    "$(sha256sum "$source_file" | cut -d ' ' -f 1)"
cmp --silent "$source_file" "$downloaded_file"

echo "Hugging Face public bucket integration test passed"
