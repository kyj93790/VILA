#!/bin/bash

set -e

echo "Starting Measure importances experiments..."

# measure importance for FILA
echo "Measure importance for FILA..."
CUDA_VISIBLE_DEVICES=0,1 python src/exec/measure_imp_model.py \
    --config-file configs/unlearn/wmdp_all/FILA.json \
    --unlearn.max_steps=125 \

echo "Measure importances completed successfully!"