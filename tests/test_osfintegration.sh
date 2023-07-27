#!/usr/bin/env bash
set -e

echo "[DEBUG]: osf setup"
pip install osfclient
export OSF_TOKEN=$OSF_TOKEN_
export OSF_USERNAME=$OSF_USERNAME_
export OSF_PROJECT_ID=$OSF_PROJECT_ID_
echo $OSF_PROJECT_ID
# mkdir -p ~/.osfcli
# echo -e "[osf]\nproject = $OSF_PROJECT_ID\nusername = $OSF_USERNAME" > ~/.osfcli/osfcli.config
# cat ~/.osfcli/osfcli.config
ls
export timestamp=$(date +%Y%m%d_%H%M%S)
echo $timestamp
export file="figure1_v1.png "
echo $timestamp_$file
osf -p abk4p upload ./paper/$file /osfstorage/github_actions/paper/"$timestamp"_"$file"
