import os
import torch

print(f"Environment variable CUDA_HOME: {os.environ.get('CUDA_HOME')}")
print(f"Environment variable LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
print(f"torch version: {torch.__version__}")
print(f"PyTorch built with CUDA version: {torch.version.cuda}")

try:
    if torch.cuda.is_available():
        print("CUDA is available! PyTorch can use GPUs.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        device = torch.device("cuda")
        print(f"Using device: {device}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Try to create a tensor on the GPU
        x = torch.rand(2, 2, device=device)
        print(f"Successfully created tensor on GPU: {x}")
    else:
        print("CUDA is NOT available. PyTorch is running on CPU.")
        print("Reason for failure (if any):")
        # This might not always show the original error, but worth a try
        print(torch.cuda.current_device()) # This will likely error if no CUDA
except Exception as e:
    print(f"An error occurred during CUDA initialization or usage: {e}")

print("Script finished.")
