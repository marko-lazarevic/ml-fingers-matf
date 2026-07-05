#!/bin/bash
set -eo pipefail

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate fingers
exec python live-app.py
