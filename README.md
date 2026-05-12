# Reinforcement Learning Snake - Complete Installation Guide

## Overview

This project implements a Deep Q-Network (DQN) reinforcement learning agent that learns to play the classic Snake game. The implementation features:

- **Deep Q-Learning** with experience replay and target networks
- **Real-time visualization** of the game, neural network weights, and training statistics
- **CUDA acceleration** for fast training on NVIDIA GPUs
- **Interactive parameter tuning** during training
- **Modern C++17** implementation with PyTorch C++ API

## System Requirements

### Hardware Requirements
- **CPU**: 64-bit x86 processor (Intel/AMD)
- **GPU**: NVIDIA CUDA-compatible GPU (optional but recommended for faster training)
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: 2GB free disk space

### Software Requirements
- **Operating System**: Ubuntu 20.04+ / Debian 10+ / Other Linux distributions
- **Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **CMake**: Version 3.22 or higher
- **CUDA Toolkit**: Version 11.0+ (if using GPU acceleration)
- **Git**: For cloning repositories

## Step-by-Step Installation

### Step 1: Install System Dependencies

#### Update Package Manager
```bash
sudo apt update && sudo apt upgrade -y
```

#### Install Essential Build Tools
```bash
sudo apt install -y build-essential cmake git pkg-config
```

#### Install Graphics and Window System Libraries
```bash
sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

#### Install OpenGL and Vulkan Development Headers
```bash
sudo apt install -y libvulkan-dev vulkan-tools libglvnd-dev
```

### Step 2: Install CUDA Toolkit (Optional but Recommended)

#### Download CUDA Toolkit
Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and download the appropriate version for your system.

#### Install CUDA (Example for CUDA 11.8)
```bash
# Download CUDA 11.8 (adjust URL for your system)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Make installer executable
chmod +x cuda_11.8.0_520.61.05_linux.run

# Run installer (accept terms and select only CUDA Toolkit)
sudo ./cuda_11.8.0_520.61.05_linux.run
```

#### Configure CUDA Environment
```bash
# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload environment
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

### Step 3: Install vcpkg Package Manager

#### Clone vcpkg Repository
```bash
cd ~
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
```

#### Bootstrap vcpkg
```bash
./bootstrap-vcpkg.sh
```

#### Add vcpkg to System PATH
```bash
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Install Project Dependencies with vcpkg

#### Install SDL3 (Simple DirectMedia Layer 3)
```bash
cd ~/vcpkg
./vcpkg install sdl3:x64-linux
```

#### Install GLM (OpenGL Mathematics Library)
```bash
./vcpkg install glm:x64-linux
```

#### Verify vcpkg Installations
```bash
./vcpkg list
# You should see:
# sdl3:x64-linux
# glm:x64-linux
```

### Step 5: Install PyTorch C++ Distribution (LibTorch)

#### Download LibTorch with CUDA Support
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake

# Download LibTorch (CUDA 11.8 version - adjust if using different CUDA version)
wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-latest.zip

# Extract LibTorch
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip
```

#### Alternative: CPU-Only Version (if no CUDA GPU)
```bash
# Download CPU-only version
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip

# Extract
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip
```

### Step 6: Configure CMake for vcpkg Integration

#### Create CMake Presets File
```bash
# Create vcpkg toolchain file reference
echo 'set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")' >> vcpkg-toolchain.cmake
```

### Step 7: Build the Project

#### Create Build Directory
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake
mkdir build
cd build
```

#### Configure with CMake
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

#### Compile the Project
```bash
# Use all available CPU cores for faster compilation
make -j$(nproc)
```

## Troubleshooting Common Issues

### Issue 1: SDL3 Not Found
```bash
# Solution: Reinstall SDL3 with vcpkg
cd ~/vcpkg
./vcpkg remove sdl3:x64-linux
./vcpkg install sdl3:x64-linux
```

### Issue 2: CUDA Libraries Not Found
```bash
# Solution: Check CUDA installation and paths
which nvcc
ls /usr/local/cuda/lib64/
echo $LD_LIBRARY_PATH
```

### Issue 3: LibTorch Linking Errors
```bash
# Solution: Verify LibTorch directory structure
ls -la libtorch/
ls -la libtorch/lib/
ls -la libtorch/include/
```

### Issue 4: CMake Configuration Fails
```bash
# Solution: Clear CMake cache and reconfigure
cd build
rm -rf *
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

### Issue 5: Compilation Errors with GCC
```bash
# Solution: Ensure GCC version supports C++17
g++ --version
# If version < 9, update GCC:
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

## Running the Application

### Basic Execution
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake/build
./ReinforcementSnake
```

### Training Controls
During training, you can use these keyboard controls:
- **↑/↓**: Select parameter to adjust
- **←/→**: Adjust selected parameter value
- **R**: Reset all parameters to defaults
- **Space**: Reset exploration rate (epsilon=1)
- **Ctrl+C**: Force immediate rendering (doesn't stop training)

### Performance Tuning
- **Game Speed**: Use +/- keys to adjust rendering FPS (5-120)
- **Training Speed**: Adjust train_speed parameter (1=slow, 100=fast)

## Project Structure

```
ReinforcementSnake/
├── CMakeLists.txt          # CMake configuration
├── main.cpp                # Application entry point
├── libtorch/               # PyTorch C++ library
├── src/
│   ├── SnakeAI.hpp         # AI implementation header
│   ├── SnakeAI.cpp         # AI implementation
│   └── Utils.h             # Constants and utilities
├── build/                  # Build output directory
└── README.md               # This file
```

## Required Libraries Summary

| Library | Version | Purpose | Installation Method |
|---------|---------|---------|-------------------|
| **SDL3** | Latest | Graphics rendering & window management | vcpkg |
| **GLM** | Latest | OpenGL mathematics | vcpkg |
| **PyTorch** | Latest | Deep learning framework | Manual download |
| **CUDA Toolkit** | 11.0+ | GPU acceleration (optional) | NVIDIA installer |
| **CMake** | 3.22+ | Build system | apt |
| **GCC/Clang** | 9+/10+ | C++17 compiler | apt |

## Verification Commands

### Verify All Dependencies
```bash
# Check compiler
g++ --version

# Check CMake
cmake --version

# Check CUDA (if installed)
nvcc --version
nvidia-smi

# Check vcpkg packages
~/vcpkg/vcpkg list

# Check LibTorch
ls -la /home/moinshaikh/CLionProjects/ReinforcementSnake/libtorch/

# Check built executable
ls -la /home/moinshaikh/CLionProjects/ReinforcementSnake/build/ReinforcementSnake
```

### Test Run
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake/build
./ReinforcementSnake --help  # Should start the training interface
```

## Next Steps

After successful installation:

1. **Run Training**: Start with a few hundred episodes to test
2. **Monitor Progress**: Watch the score and epsilon graphs
3. **Adjust Parameters**: Use keyboard controls to tune hyperparameters
4. **Save Models**: Extend the code to save trained models
5. **Experiment**: Try different network architectures or reward functions

## Support

For issues related to:
- **vcpkg**: [vcpkg GitHub Issues](https://github.com/Microsoft/vcpkg/issues)
- **PyTorch**: [PyTorch Forums](https://discuss.pytorch.org/)
- **SDL3**: [SDL Discord/Forums](https://www.libsdl.org/)
- **CUDA**: [NVIDIA Developer Forums](https://developer.nvidia.com/forums)

---

*This installation guide covers all necessary dependencies and steps to get the Reinforcement Learning Snake project running on Linux systems.*
