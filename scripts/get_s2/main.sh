#!/usr/bin/env bash

# activate environment with the required libraries
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate earthengine

# error handling
set -eu
trap 'echo "An error occurred. Exiting..." >&2' ERR

###################### CUSTOM PARAMETERS
LOGS_DIR="$1"
EE_PROJECT_NAME="$2"
AOI_PATH="$3"
START_DATE="$4"
END_DATE="$5"
OUT_DIRECTORY="$6"
##########################################

# make parent dir
mkdir -p "$OUT_DIRECTORY" "$LOGS_DIR"

LOG_PATH="$LOGS_DIR/ee_s2.log"

python $HOME/soil_public/scripts/get_s2/get_sentinel2_timeseries.py "$LOG_PATH" "$EE_PROJECT_NAME" "$AOI_PATH" "$START_DATE" "$END_DATE" "$OUT_DIRECTORY"