# Linux Commands Cheat Sheet for Reinforcement Snake Project

## System Setup Commands

### Update System
```bash
# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Clean up unnecessary packages
sudo apt autoremove -y
sudo apt autoclean
```

### Install Essential Tools
```bash
# Build tools
sudo apt install -y build-essential cmake git pkg-config

# Version checks
gcc --version
cmake --version
git --version
```

## vcpkg Commands

### vcpkg Setup
```bash
# Clone vcpkg
cd ~
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap
./bootstrap-vcpkg.sh

# Add to PATH
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Package Management
```bash
# Install dependencies
./vcpkg install sdl3:x64-linux
./vcpkg install glm:x64-linux

# List installed packages
./vcpkg list

# Search packages
./vcpkg search sdl
./vcpkg search math

# Remove packages
./vcpkg remove package-name
./vcpkg remove package-name --purge

# Update packages
./vcpkg upgrade
./vcpkg upgrade package-name
```

## CUDA Commands

### Check GPU Status
```bash
# Check NVIDIA GPU
lspci | grep -i nvidia
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check GPU details
nvidia-smi -q
```

### CUDA Installation
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8

# Configure environment
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### CUDA Verification
```bash
# Check CUDA installation
nvcc --version
ls /usr/local/cuda/lib64/

# Test CUDA compilation
nvcc --version | grep release
```

## Project Build Commands

### Navigate to Project
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake
```

### Download LibTorch
```bash
# CUDA version
wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-latest.zip

# CPU-only version
# wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip

# Extract
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip
```

### Build Project
```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Alternative: Single-threaded build
make

# Clean build
rm -rf *
```

### Run Application
```bash
# Basic run
./ReinforcementSnake

# Run with specific parameters
./ReinforcementSnake --epochs 1000
```

## File System Commands

### Project Structure
```bash
# View project structure
tree -a
ls -la

# Check source files
ls -la src/
cat src/SnakeAI.hpp | head -20

# Check CMakeLists.txt
cat CMakeLists.txt
```

### File Operations
```bash
# Copy files
cp source.txt destination.txt
cp -r source_dir/ destination_dir/

# Move files
mv old_name.txt new_name.txt
mv file.txt /path/to/destination/

# Remove files
rm file.txt
rm -rf directory/
```

## Process Management Commands

### Monitor Processes
```bash
# Show running processes
ps aux
ps aux | grep ReinforcementSnake

# Show resource usage
top
htop

# Monitor GPU processes
nvidia-smi pmon
```

### Kill Processes
```bash
# Kill by PID
kill 12345

# Kill by name
pkill ReinforcementSnake

# Force kill
kill -9 12345
pkill -9 ReinforcementSnake
```

### Background Processes
```bash
# Run in background
./ReinforcementSnake &

# Bring to foreground
fg

# List background jobs
jobs

# Kill background job
kill %1
```

## System Monitoring Commands

### CPU and Memory
```bash
# CPU usage
top
htop
iostat

# Memory usage
free -h
cat /proc/meminfo

# Disk usage
df -h
du -sh *
```

### GPU Monitoring
```bash
# GPU status
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# GPU processes
nvidia-smi pmon -s u -o T
```

## Network Commands

### Download Commands
```bash
# wget
wget https://example.com/file.zip
wget -O custom_name.zip https://example.com/file.zip

# curl
curl https://example.com/file.zip -o file.zip
curl -O https://example.com/file.zip
```

### Git Commands
```bash
# Clone repository
git clone https://github.com/user/repo.git

# Check status
git status

# Pull changes
git pull origin main

# Check remote
git remote -v
```

## Environment Commands

### Environment Variables
```bash
# View all variables
env
printenv

# View specific variable
echo $PATH
echo $LD_LIBRARY_PATH

# Set temporary variable
export VAR_NAME=value

# Set permanent variable
echo 'export VAR_NAME=value' >> ~/.bashrc
source ~/.bashrc
```

### Shell Configuration
```bash
# Reload shell configuration
source ~/.bashrc
source ~/.zshrc

# Edit configuration
nano ~/.bashrc
vim ~/.bashrc
```

## Troubleshooting Commands

### Check Dependencies
```bash
# Check library dependencies
ldd ./ReinforcementSnake

# Check shared libraries
ldconfig -p | grep sdl
ldconfig -p | grep torch

# Check system libraries
ls /usr/lib/x86_64-linux-gnu/ | grep -E "(sdl|cuda)"
```

### Debug Build Issues
```bash
# Verbose build output
make VERBOSE=1

# Check CMake configuration
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug

# Check compiler
which gcc
which g++
gcc --version
```

### Check Permissions
```bash
# File permissions
ls -la file.txt

# Change permissions
chmod +x script.sh
chmod 755 file.txt

# Change ownership
sudo chown user:group file.txt
```

## Archive and Compression Commands

### Create Archives
```bash
# Create tar.gz
tar -czf archive.tar.gz directory/

# Create zip
zip -r archive.zip directory/

# Create tar.bz2
tar -cjf archive.tar.bz2 directory/
```

### Extract Archives
```bash
# Extract tar.gz
tar -xzf archive.tar.gz

# Extract zip
unzip archive.zip

# Extract tar.bz2
tar -xjf archive.tar.bz2
```

## Text Processing Commands

### View Files
```bash
# View file contents
cat file.txt

# View with line numbers
cat -n file.txt

# View first/last lines
head file.txt
tail file.txt

# View specific lines
sed -n '10,20p' file.txt
```

### Search in Files
```bash
# Search for text
grep "search_term" file.txt

# Recursive search
grep -r "search_term" directory/

# Case insensitive search
grep -i "search_term" file.txt
```

## System Information Commands

### System Info
```bash
# System information
uname -a
lsb_release -a

# Hardware info
lscpu
lspci
lsusb

# Disk info
lsblk
df -h
```

### Kernel and Modules
```bash
# Kernel version
uname -r

# Loaded modules
lsmod

# Check for NVIDIA module
lsmod | grep nvidia
```

## Quick Reference Summary

### Essential Commands for This Project
```bash
# 1. Install system deps
sudo apt update && sudo apt install -y build-essential cmake git

# 2. Setup vcpkg
cd ~ && git clone https://github.com/Microsoft/vcpkg.git && cd vcpkg && ./bootstrap-vcpkg.sh

# 3. Install project deps
./vcpkg install sdl3:x64-linux glm:x64-linux

# 4. Download LibTorch
cd /home/moinshaikh/CLionProjects/ReinforcementSnake
wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-latest.zip && unzip libtorch-shared-with-deps-latest.zip && rm libtorch-shared-with-deps-latest.zip

# 5. Build project
mkdir build && cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# 6. Run application
./ReinforcementSnake
```

---

*This cheat sheet provides quick reference commands for setting up and working with the Reinforcement Learning Snake project on Linux.*
