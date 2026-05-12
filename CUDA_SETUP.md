# CUDA Setup and Configuration Guide

## Overview

This guide covers CUDA installation and configuration for the Reinforcement Learning Snake project. CUDA provides GPU acceleration that can speed up neural network training by 5-10x compared to CPU-only training.

## System Requirements

### Hardware Requirements
- **NVIDIA GPU**: CUDA-capable GPU (Compute Capability 3.5+)
- **GPU Memory**: Minimum 2GB VRAM, 4GB+ recommended
- **System RAM**: 8GB+ recommended for large training runs

### Software Requirements
- **Linux**: Ubuntu 18.04+ / Debian 10+ / RHEL 8+
- **Kernel Version**: 3.10+ (check with `uname -r`)
- **GCC**: Compatible version for your CUDA version

## Step-by-Step CUDA Installation

### Step 1: Verify GPU Compatibility

#### Check NVIDIA GPU
```bash
# Check if you have an NVIDIA GPU
lspci | grep -i nvidia

# Check GPU details
nvidia-smi

# If nvidia-smi is not found, install drivers first
```

#### Check Compute Capability
```bash
# Check your GPU's compute capability
# Visit: https://developer.nvidia.com/cuda-gpus
# Common values:
# - GTX 10xx: 6.0-6.1
# - RTX 20xx: 7.5
# - RTX 30xx: 8.6
# - RTX 40xx: 8.9
```

### Step 2: Install NVIDIA Drivers

#### Ubuntu/Debian Method
```bash
# Update package list
sudo apt update

# Install recommended drivers
sudo ubuntu-drivers autoinstall

# Or install specific driver version
sudo apt install nvidia-driver-535

# Reboot system
sudo reboot
```

#### Verify Driver Installation
```bash
# After reboot, check driver status
nvidia-smi

# Expected output should show:
# - GPU name
# - Driver version
# - CUDA version (maximum supported)
```

### Step 3: Install CUDA Toolkit

#### Method 1: NVIDIA Repository (Recommended)
```bash
# Download CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

# Install repository
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt-get update

# Install CUDA toolkit
sudo apt-get -y install cuda-toolkit-11-8

# Alternative versions:
# sudo apt-get -y install cuda-toolkit-12-0
# sudo apt-get -y install cuda-toolkit-12-1
```

#### Method 2: NVIDIA Installer (Universal)
```bash
# Download CUDA installer (example for CUDA 11.8)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Make installer executable
chmod +x cuda_11.8.0_520.61.05_linux.run

# Run installer
sudo ./cuda_11.8.0_520.61.05_linux.run

# During installation:
# 1. Accept EULA
# 2. Select "CUDA Toolkit" only (deselect driver if already installed)
# 3. Choose installation location (default: /usr/local/cuda-11.8)
# 4. Create symbolic link when prompted
```

### Step 4: Configure Environment

#### Set Environment Variables
```bash
# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# For CUDA 12.x, you might need:
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc
```

#### Verify CUDA Installation
```bash
# Check CUDA compiler version
nvcc --version

# Check CUDA runtime
nvcc --version | grep "release"

# Check library paths
ls /usr/local/cuda/lib64/
```

### Step 5: Verify CUDA Functionality

#### Test CUDA Compilation
```bash
# Create test file
cat > test_cuda.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Running CUDA test...\n");
    
    hello_cuda<<<1, 5>>>();
    cudaDeviceSynchronize();
    
    printf("CUDA test completed successfully!\n");
    return 0;
}
EOF

# Compile and run
nvcc test_cuda.cu -o test_cuda
./test_cuda

# Clean up
rm test_cuda.cu test_cuda
```

#### Check GPU Information
```bash
# Detailed GPU information
nvidia-smi -q

# Check CUDA devices
nvidia-smi --list-gpus

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## CUDA Integration with the Project

### Step 6: Configure CMake for CUDA

#### Verify CMake CUDA Support
```bash
# Check if CMake supports CUDA
cmake --help | grep -i cuda

# Check for CUDA toolkit
cmake --find-package -DNAME=CUDA -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST
```

#### Update CMakeLists.txt
The project's CMakeLists.txt already includes CUDA configuration:

```cmake
# CUDA architecture configuration
set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)

# CUDA libraries linking
target_link_libraries(ReinforcementSnake PRIVATE
    "${CMAKE_SOURCE_DIR}/libtorch/lib/libtorch_cuda.so"
    "${CMAKE_SOURCE_DIR}/libtorch/lib/libc10_cuda.so"
    "/usr/local/cuda/lib64/libcudart.so"
    "/usr/local/cuda/lib64/libnvrtc.so"
)
```

### Step 7: Install cuDNN (Optional but Recommended)

#### Download cuDNN
```bash
# Visit: https://developer.nvidia.com/cudnn
# Download cuDNN that matches your CUDA version
# Example for CUDA 11.x with cuDNN 8.9:
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
```

#### Install cuDNN
```bash
# Extract cuDNN
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz

# Copy files to CUDA directory
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Clean up
rm -rf cudnn-*-archive cudnn-*-archive.tar.xz
```

#### Verify cuDNN Installation
```bash
# Check cuDNN version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Or check library
ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep libcudnn
```

## Project-Specific CUDA Configuration

### LibTorch CUDA Setup

#### Download CUDA-enabled LibTorch
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake

# Download CUDA version (example for CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-latest.zip

# Extract
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip
```

#### Verify LibTorch CUDA Libraries
```bash
# Check CUDA libraries in libtorch
ls -la libtorch/lib/ | grep cuda

# Expected files:
# libtorch_cuda.so
# libc10_cuda.so
# libcudart.so (if included)
```

### Build with CUDA Support

#### Configure CMake
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake/build

# Configure with CUDA support
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

# The project should automatically detect CUDA if properly installed
```

#### Build and Test
```bash
# Build the project
make -j$(nproc)

# Test CUDA functionality
./ReinforcementSnake

# Check if CUDA is being used (look for GPU-related output)
```

## Performance Optimization

### GPU Memory Management

#### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Process-specific GPU usage
nvidia-smi pmon -s u -o T
```

#### Optimize Memory Usage
```bash
# Set GPU memory fraction (if needed)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Clear GPU cache (if memory issues)
sudo nvidia-smi --gpu-reset
```

### Multi-GPU Setup

#### Check Multiple GPUs
```bash
# List all GPUs
nvidia-smi --list-gpus

# Check specific GPU
nvidia-smi -i 0
nvidia-smi -i 1
```

#### Configure for Multi-GPU
```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0,1

# For single GPU usage
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting CUDA Issues

### Common Problems and Solutions

#### Issue 1: CUDA Not Found by CMake
```bash
# Problem: CMake cannot find CUDA toolkit
# Solution:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reconfigure CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

#### Issue 2: GPU Out of Memory
```bash
# Problem: Training fails due to insufficient GPU memory
# Solutions:
# 1. Reduce batch size in training parameters
# 2. Use smaller network architecture
# 3. Clear GPU memory between runs
```

#### Issue 3: CUDA Version Mismatch
```bash
# Problem: LibTorch CUDA version doesn't match system CUDA
# Solution:
# Download matching LibTorch version
# Example: For CUDA 12.0
wget https://download.pytorch.org/libtorch/cu120/libtorch-shared-with-deps-latest.zip
```

#### Issue 4: Driver Installation Problems
```bash
# Problem: NVIDIA driver installation fails
# Solution:
# 1. Remove existing drivers
sudo apt purge nvidia-*

# 2. Reboot to safe mode
# 3. Install drivers again
sudo ubuntu-drivers autoinstall
```

#### Issue 5: Compilation Errors
```bash
# Problem: CUDA compilation fails
# Solution:
# Check GCC compatibility
gcc --version
nvcc --version

# Ensure GCC version is compatible with CUDA version
# CUDA 11.x supports GCC 7-9
# CUDA 12.x supports GCC 9-11
```

### Verification Commands

#### Complete CUDA Check
```bash
#!/bin/bash
echo "=== CUDA System Check ==="
echo "GPU Information:"
nvidia-smi
echo -e "\nCUDA Compiler:"
nvcc --version
echo -e "\nCUDA Libraries:"
ls /usr/local/cuda/lib64/ | grep -E "(cuda|cudnn)"
echo -e "\nEnvironment Variables:"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH" | grep -o cuda
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | grep -o cuda
echo -e "\nProject CUDA Libraries:"
ls -la /home/moinshaikh/CLionProjects/ReinforcementSnake/libtorch/lib/ | grep cuda
```

## Performance Benchmarks

### Expected Performance Gains
- **CPU-only**: ~10-50 episodes/second
- **GPU (CUDA)**: ~100-500 episodes/second
- **High-end GPU**: Up to 1000 episodes/second

### Memory Requirements
- **Small Network**: 1-2GB VRAM
- **Medium Network**: 2-4GB VRAM
- **Large Network**: 4-8GB VRAM

---

*This CUDA guide provides comprehensive setup instructions for GPU acceleration in the Reinforcement Learning Snake project.*
