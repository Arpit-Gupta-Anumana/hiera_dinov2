#!/bin/bash

# Set the number of GPUs to use
export NUM_GPUS=4

# Set the torch hub cache to a local directory
export TORCH_HOME=$(pwd)/.torch_cache

# Launch the distributed training process
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    src/train.py
