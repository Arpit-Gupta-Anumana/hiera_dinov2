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

**What We've Changed:**
*   **`LOG_DIR="ddp_logs"`:** We define a variable for the log directory name.
*   **`rm -rf $LOG_DIR` and `mkdir -p $LOG_DIR`:** We add commands to clear out any old logs and create a fresh directory for our new run. This is important for clean debugging.
*   **`--log_dir $LOG_DIR`:** This is the crucial new argument. We are telling `torchrun` to redirect the `stdout` and `stderr` of each of its child processes into files inside this directory.

---

### **Next Steps: The Final Debugging Push**

This will be the final step to find the error.

1.  **Replace the content** of your `run_distributed.sh` script with the new version above.
2.  **Run the script** as before:
    ```bash
    ./run_distributed.sh
    ```
3.  The script **will fail again** with the same `ChildFailedError`. **This is expected.**
4.  After it fails, a new directory named `ddp_logs` will have been created in your project folder.
5.  This directory will contain subfolders for each process (`0`, `1`, `2`, `3`). The real error message is almost always in the log file for the first process (`rank 0`).
6.  **Run this command to display the error log for rank 0:**
    ```bash
    cat ddp_logs/0/stderr.log
    ```
7.  **Please paste the entire output of that `cat` command here.** This file will contain the true, underlying error that is causing the silent crash, and we will be able to fix it definitively.
