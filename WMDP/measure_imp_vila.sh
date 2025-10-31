#!/bin/bash

set -e

echo "Starting Measure importances experiments..."

# measure importance for VILA
echo "Measure importance for VILA..."
CUDA_VISIBLE_DEVICES=0 python src/exec/measure_imp_model.py \
    --config-file configs/unlearn/wmdp_all/VILA.json \
    --unlearn.max_steps=125

echo "Measure importances completed successfully!"