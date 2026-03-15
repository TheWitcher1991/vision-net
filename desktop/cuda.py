import subprocess
import sys

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("\nTrying to detect NVIDIA hardware...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout[:500])  # First 500 chars
        else:
            print("nvidia-smi not found or not working")
    except FileNotFoundError:
        print("nvidia-smi command not found - NVIDIA drivers not installed")

    # Check if on Windows
    if sys.platform == "win32":
        print("\nOn Windows, you can check Device Manager:")
        print("1. Open Device Manager")
        print("2. Look under 'Display adapters'")
        print("3. If you see an NVIDIA GPU, drivers need to be installed/updated")
