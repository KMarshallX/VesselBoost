#!/usr/bin/env bash
set -e

pip install osfclient
echo "[DEBUG]: osf setup"
export OSF_TOKEN=$OSF_TOKEN_
export OSF_USERNAME=$OSF_USERNAME_
export OSF_PROJECT_ID=$OSF_PROJECT_ID_

mkdir -p ~/.osfcli
echo -e "[osf]\nproject = $OSF_PROJECT_ID\nusername = \$OSF_USERNAME" > ~/.osfcli/osfcli.config
cd ./paper/

echo "[DEBUG]: saving data to osf"
osf -p abk4p upload ./figure1_v1.png /osfstorage/test/file.png

echo "Testing done!"