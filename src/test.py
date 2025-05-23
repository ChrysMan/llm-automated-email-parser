import os
import torch

#os.environ["LD_LIBRARY_PATH"] = "/home/chryssida/venv/lib/python3.12/site-packages/torch/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
#os.getenv("LD_LIBRARY_PATH", "")
#os.environ["CUDA_HOME"] = "/home/chryssida/venv/lib/python3.12/site-packages/torch"

print(f"1. PyTorch version: {torch.__version__}")
print(f"2. CUDA available: {torch.cuda.is_available()}")
print(f"3. CUDA runtime (PyTorch): {torch.version.cuda}")

try:
    print(f"5. CUDA paths:")
    print(f"   - torch: {os.path.join(os.path.dirname(torch.__file__), 'lib')}")
    print(f"   - system: {os.popen('which nvcc').read().strip()}")
except Exception as e:
    print(f"Diagnostic error: {str(e)}")

print("\n6. Environment variables:")
print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"   - LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
