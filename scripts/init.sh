#!/bin/bash

# Return to the source directory
# shellcheck disable=SC2046
cd $(dirname "$0") && cd ..

# Begin build image in source directory
docker build -t ml_transformer:v1.0 . -f ./docker/Dockerfile
