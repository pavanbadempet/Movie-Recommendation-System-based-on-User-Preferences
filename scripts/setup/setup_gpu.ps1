# setup_gpu.ps1
# Script to enable NVIDIA GPU support for the Movie Recommendation System

Write-Host "==================================================="
Write-Host "       Setting up GPU Support (CUDA 12.1)          "
Write-Host "==================================================="

# 1. Uninstall CPU-only versions
Write-Host "1. Removing existing CPU-only PyTorch..."
pip uninstall -y torch torchvision torchaudio

# 2. Install CUDA-enabled PyTorch
# We use the stable index for Windows + CUDA 12.1
Write-Host "2. Installing GPU-enabled PyTorch (this involves a ~2.5GB download)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Verify
Write-Host "3. Verifying installation..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}')"

Write-Host "==================================================="
Write-Host "If 'CUDA Available' is True, you are ready to race! 🏎️"
Write-Host "==================================================="
Write-Host "NOTE: If this fails, please ensure you have installed the NVIDIA Drivers from:"
Write-Host "https://www.nvidia.com/Download/index.aspx"
