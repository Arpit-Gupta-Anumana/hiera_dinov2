#!/bin/bash

# Set the number of GPUs to use
export NUM_GPUS=4

# Set the torch hub cache to a local directory
export TORCH_HOME=$(pwd)/.torch_cache

# --- START: ADDED LOGGING ---
# Create a directory to store logs from each process
LOG_DIR="ddp_logs"
echo "Clearing old logs..."
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
echo "Logs will be saved in the '${LOG_DIR}' directory."
# --- END: ADDED LOGGING ---

# Launch the distributed training process with logging enabled
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --log_dir $LOG_DIR \
    -m src.train```
