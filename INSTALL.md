# Library Installation Guide for Reinforcement Snake

## Quick Installation Commands

This file contains all the essential commands to install required libraries for the Reinforcement Learning Snake project.

### 1. System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential cmake git pkg-config

# Install graphics libraries
sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# Install OpenGL/Vulkan headers
sudo apt install -y libvulkan-dev vulkan-tools libglvnd-dev
```

### 2. CUDA Toolkit (Optional but Recommended)
```bash
# Download CUDA 11.8 (adjust URL for your system)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Make executable and install
chmod +x cuda_11.8.0_520.61.05_linux.run
sudo ./cuda_11.8.0_520.61.05_linux.run

# Configure environment
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

### 3. vcpkg Package Manager
```bash
# Clone vcpkg
cd ~
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
./bootstrap-vcpkg.sh

# Add to PATH
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install Project Dependencies with vcpkg
```bash
cd ~/vcpkg

# Install SDL3
./vcpkg install sdl3:x64-linux

# Install GLM
./vcpkg install glm:x64-linux

# Verify installations
./vcpkg list
```

### 5. Install PyTorch C++ (LibTorch)
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake

# Download CUDA version (recommended)
wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-latest.zip

# Extract and cleanup
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip

# For CPU-only systems, use:
# wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
```

### 6. Build the Project
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

# Compile
make -j$(nproc)

# Run the application
./ReinforcementSnake
```

## Library Information

### SDL3 (Simple DirectMedia Layer 3)
- **Purpose**: Graphics rendering and window management
- **Installation**: vcpkg install sdl3:x64-linux
- **Used for**: Game visualization, real-time rendering

### GLM (OpenGL Mathematics)
- **Purpose**: Mathematics library for 3D graphics
- **Installation**: vcpkg install glm:x64-linux
- **Used for**: Mathematical operations in rendering

### PyTorch C++ (LibTorch)
- **Purpose**: Deep learning framework
- **Installation**: Manual download from pytorch.org
- **Used for**: Neural network implementation, GPU acceleration

### CUDA Toolkit
- **Purpose**: GPU acceleration for deep learning
- **Installation**: NVIDIA installer
- **Used for**: Speeding up neural network training

## Verification Commands

```bash
# Check all installations
g++ --version          # Should be 9+
cmake --version        # Should be 3.22+
nvcc --version         # CUDA version (if installed)
~/vcpkg/vcpkg list     # Should show sdl3 and glm
ls libtorch/           # Should show include/, lib/ directories
```

## Troubleshooting

### Common Issues and Solutions

1. **SDL3 not found during cmake**:
   ```bash
   cd ~/vcpkg
   ./vcpkg remove sdl3:x64-linux
   ./vcpkg install sdl3:x64-linux
   ```

2. **CUDA libraries not found**:
   ```bash
   echo $LD_LIBRARY_PATH
   ls /usr/local/cuda/lib64/
   ```

3. **Compilation errors**:
   ```bash
   # Clear build and reconfigure
   cd build
   rm -rf *
   cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

4. **Permission denied errors**:
   ```bash
   # Fix permissions for vcpkg
   sudo chown -R $USER:$USER ~/vcpkg
   ```

---

*These commands provide a complete setup for the Reinforcement Learning Snake project on Linux systems.*
