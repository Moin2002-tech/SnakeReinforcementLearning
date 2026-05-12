# vcpkg Setup and Configuration Guide

## What is vcpkg?

vcpkg is a C++ package manager that simplifies the installation of third-party libraries. This project uses vcpkg to manage SDL3 and GLM dependencies.

## Complete vcpkg Setup Process

### Step 1: Install vcpkg

#### Clone the Repository
```bash
# Navigate to home directory
cd ~

# Clone vcpkg repository
git clone https://github.com/Microsoft/vcpkg.git

# Enter vcpkg directory
cd vcpkg
```

#### Bootstrap vcpkg
```bash
# Run the bootstrap script
./bootstrap-vcpkg.sh

# This will create the vcpkg executable
# The process may take a few minutes
```

#### Verify Installation
```bash
# Check vcpkg version
./vcpkg version

# List available commands
./vcpkg help
```

### Step 2: Configure Environment Variables

#### Add vcpkg to PATH
```bash
# Add to ~/.bashrc for persistent configuration
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc

# Reload the shell configuration
source ~/.bashrc

# Verify the setup
echo $VCPKG_ROOT
which vcpkg
```

#### Alternative: Temporary Setup
```bash
# For current session only
export VCPKG_ROOT=~/vcpkg
export PATH=$VCPKG_ROOT:$PATH
```

### Step 3: Install Project Dependencies

#### Install SDL3
```bash
cd ~/vcpkg

# Install SDL3 for x64 Linux
./vcpkg install sdl3:x64-linux

# This will download, build, and install SDL3
# Process may take 10-20 minutes
```

#### Install GLM
```bash
# Install GLM for x64 Linux
./vcpkg install glm:x64-linux

# GLM is header-only, installation is faster
```

#### Verify Installations
```bash
# List installed packages
./vcpkg list

# Expected output:
# sdl3:x64-linux
# glm:x64-linux
```

### Step 4: Configure CMake Integration

#### Method 1: Command Line (Recommended for this project)
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake/build

# Configure with vcpkg toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

#### Method 2: CMake Presets File
```bash
# Create a CMakePresets.json file in project root
cat > CMakePresets.json << 'EOF'
{
  "version": 3,
  "configurePresets": [
    {
      "name": "vcpkg",
      "generator": "Unix Makefiles",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "buildType": "Release"
    }
  ]
}
EOF

# Then use:
cmake --preset vcpkg
```

#### Method 3: Environment Variable
```bash
# Set environment variable
export CMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake

# Now cmake will automatically use vcpkg
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Step 5: Build the Project

#### Standard Build Process
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake
mkdir build
cd build

# Configure with vcpkg
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j$(nproc)
```

#### Verify Build Success
```bash
# Check if executable was created
ls -la ReinforcementSnake

# Test run
./ReinforcementSnake
```

## vcpkg Commands Reference

### Basic Package Management

#### Search for Packages
```bash
# Search for SDL packages
./vcpkg search sdl

# Search for math libraries
./vcpkg search math
```

#### Package Information
```bash
# Get details about SDL3
./vcpkg info sdl3

# Show supported features
./vcpkg info sdl3 --x-all-features
```

#### Install Packages
```bash
# Basic installation
./vcpkg install package-name

# With specific target triplet
./vcpkg install package-name:x64-linux

# With specific features
./vcpkg install package-name[feature]:x64-linux
```

#### Remove Packages
```bash
# Remove package
./vcpkg remove package-name

# Remove and purge (delete build artifacts)
./vcpkg remove package-name --purge
```

#### Update Packages
```bash
# Update all packages
./vcpkg upgrade

# Update specific package
./vcpkg upgrade package-name

# Reinstall package
./vcpkg reinstall package-name
```

### Advanced Commands

#### Export/Import
```bash
# Export installed packages
./vcpkg export package-name --output-dir=exported

# Create manifest file
./vcpkg format-manifest
```

#### Build Trees
```bash
# Show build tree
./vcpkg depend-info package-name

# Show reverse dependencies
./vcpkg depend-info package-name --sort
```

#### Configuration
```bash
# Set default triplet
./vcpkg integrate install

# Show integration status
./vcpkg integrate project
```

## Troubleshooting vcpkg Issues

### Common Problems and Solutions

#### Issue 1: Bootstrap Fails
```bash
# Problem: Permission denied or missing dependencies
# Solution:
sudo apt install curl zip unzip tar
chmod +x bootstrap-vcpkg.sh
./bootstrap-vcpkg.sh
```

#### Issue 2: Package Build Fails
```bash
# Problem: Compilation errors during package installation
# Solution:
# Clean and retry
./vcpkg remove package-name --purge
./vcpkg install package-name:x64-linux

# Or try with different triplet
./vcpkg install package-name:x64-linux-dynamic
```

#### Issue 3: CMake Can't Find Packages
```bash
# Problem: CMake configuration fails to find vcpkg packages
# Solution:
# Check toolchain file path
ls ~/vcpkg/scripts/buildsystems/vcpkg.cmake

# Verify package installation
./vcpkg list

# Try explicit configuration
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-linux
```

#### Issue 4: Permission Errors
```bash
# Problem: Permission denied accessing vcpkg directories
# Solution:
sudo chown -R $USER:$USER ~/vcpkg
chmod -R u+rwx ~/vcpkg
```

#### Issue 5: Network Issues
```bash
# Problem: Download failures due to network issues
# Solution:
# Use proxy if needed
export https_proxy=http://proxy.example.com:8080
export http_proxy=http://proxy.example.com:8080

# Or use git configuration
git config --global http.proxy http://proxy.example.com:8080
```

### Performance Optimization

#### Speed Up Package Installation
```bash
# Use more CPU cores
./vcpkg install package-name --triplet x64-linux --jobs 8

# Use binary caching
export VCPKG_BINARY_SOURCES=clear;default,readwrite

# Disable features you don't need
./vcpkg install package-name[core]:x64-linux
```

#### Reduce Disk Usage
```bash
# Clean build artifacts
./vcpkg remove --outdated --purge

# Use smaller triplets
./vcpkg install package-name:x64-linux-dynamic
```

## vcpkg Integration with IDEs

### Visual Studio Code
```bash
# Install C/C++ extension
code --install-extension ms-vscode.cpptools

# Configure CMake Tools extension
# Add to settings.json:
{
    "cmake.configureSettings": {
        "CMAKE_TOOLCHAIN_FILE": "~/vcpkg/scripts/buildsystems/vcpkg.cmake"
    }
}
```

### CLion
```bash
# In CLion settings:
# Settings → Build, Execution, Deployment → CMake
# Add CMake option: -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Qt Creator
```bash
# In Projects → Build & Run → CMake
# Add argument: -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
```

## Project-Specific vcpkg Configuration

### For This Project

#### Required Packages
```bash
./vcpkg install sdl3:x64-linux
./vcpkg install glm:x64-linux
```

#### Optional Development Packages
```bash
# For debugging
./vcpkg install sdl3[tools]:x64-linux

# For testing
./vcpkg install catch2:x64-linux
```

#### Manifest File (vcpkg.json)
```json
{
  "name": "reinforcement-snake",
  "version": "1.0.0",
  "dependencies": [
    "sdl3",
    "glm"
  ]
}
```

#### Using Manifest Mode
```bash
# Create vcpkg.json in project root
# Then simply run:
cmake .. -DCMAKE_BUILD_TYPE=Release

# vcpkg will automatically install dependencies
```

## Best Practices

### 1. Version Pinning
```bash
# Pin specific versions
./vcpkg install sdl3@3.1.2:x64-linux

# Create version file
echo "sdl3" > vcpkg.json
```

### 2. Clean Environment
```bash
# Regular cleanup
./vcpkg remove --outdated --purge
./vcpkg integrate remove
```

### 3. Backup Configuration
```bash
# Export installed packages
./vcpkg export --output-dir=vcpkg-backup

# Save triplet configuration
cp ~/vcpkg/triplets/x64-linux.cmake my-triplet.cmake
```

---

*This vcpkg guide provides comprehensive setup and configuration instructions for the Reinforcement Learning Snake project.*
