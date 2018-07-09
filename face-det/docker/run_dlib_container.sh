#!/bin/bash

# This script's absolute dir
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Assume this script is in the docker sub-directory.
projectDir=$( cd $SCRIPT_DIR/.. && pwd )

nvidia-docker run -ti --name gaze-tracking-dlib --network=host -v "$projectDir":/code gaze-tracking/dlib
