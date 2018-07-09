#!/bin/bash

# This script's absolute dir
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

docker build -t gaze-tracking/dlib --network=host $SCRIPT_DIR
