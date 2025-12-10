#!/bin/bash

# --- CONFIGURATION ---
# Set the number of GPUs you want to use
export NUM_GPUS=4

# --- ENVIRONMENT SETUP ---
# Set a local torch hub cache to prevent permission errors
export TORCH_HOME=$(pwd)/.torch_cache
echo "Using TORCH_HOME: ${TORCH_HOME}"

# Create a directory to store logs from each process for debugging
LOG_DIR="ddp_logs"
echo "Clearing old logs from '${LOG_DIR}'..."
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
echo "Logs for this run will be saved in '${LOG_DIR}'."

# --- LAUNCH COMMAND ---
# Use torchrun with the -m flag to run src.train as a module
# This correctly handles relative imports within your project.
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    --log_dir $LOG_DIR \
    -m src.train

echo "--- Training script finished ---"
