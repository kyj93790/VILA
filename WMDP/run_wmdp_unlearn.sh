#!/bin/bash

set -e

## if you want to run other methods, please change the config file like, GA+FT+FILA, NPO+FT+VILA, IHL+VILA ... 
echo "Starting unlearning model experiments..."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0,1 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp_all/GA+FT+VILA.json \
    --unlearn.max_steps=125 \


echo "All experiments completed successfully!"
