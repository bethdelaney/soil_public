#!/usr/bin/env bash

# activate environment with the required libraries
source /home/mattpayne/miniconda3/etc/profile.d/conda.sh
conda activate earthengine # too lazy to make another env

# error handling
set -eu
trap 'echo "An error occurred. Exiting..." >&2' ERR

###################### CUSTOM PARAMETERS
LOGS_DIR="$1"
IN_CSV_PATH="$2"
OUT_DIRECTORY="$3"
# FILTER="Savitzky-Golay" # currently not needed as only SG filter is available
##########################################

# make parent dir
mkdir -p "$LOGS_DIR" "$OUT_DIRECTORY" 

LOG_PATH="$LOGS_DIR/s2_smoothing.log"

python $HOME/forked/soil_public/scripts/smooth/apply_smoothing.py "$LOG_PATH" "$IN_CSV_PATH" "$OUT_DIRECTORY"