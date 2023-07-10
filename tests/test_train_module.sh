#!/usr/bin/env bash
set -e

cp -r . ./tmp/vessel_code

# test readme
echo "[DEBUG]: testing the clone command from the README:"
clone_command=`cat ./tmp/vessel_code/README.md | grep https://github.com/KMarshallX/vessel_code.git`
echo $clone_command
$clone_command

echo "[DEBUG]: testing the miniconda installation from the README:"
get_command=`cat ./tmp/vessel_code/README.md | grep miniconda-setup.sh`
echo $get_command
$get_command > /dev/null 2>&1

echo "[DEBUG]: testing the conda env build from the README:"
cd vessel_code
condaenv_command=`cat ./README.md | grep environment.yml`
echo $condaenv_command
$condaenv_command

echo "[DEBUG]: testing conda activate command from the README:"
condaenv_command=`cat ./README.md | grep environment.yml`
echo $condaenv_command
$condaenv_command