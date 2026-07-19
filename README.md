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

---

# Project Architecture and Implementation Details

## Overview

The Reinforcement Learning Snake project implements a sophisticated Deep Q-Network (DQN) agent that learns to play Snake through reinforcement learning. This document details the architecture, algorithms, and implementation choices.

## Core Components

### 1. Deep Q-Network Architecture

#### Neural Network Structure
```
Input Layer: 16 neurons (state representation)
    ↓
Hidden Layer 1: 128 neurons (ReLU activation)
    ↓
Hidden Layer 2: 128 neurons (ReLU activation)
    ↓
Output Layer: 4 neurons (Q-values for actions)
```

#### State Representation (16-dimensional vector)
The agent observes the game state through a carefully designed 16-dimensional feature vector:

1. **Danger Indicators (4 dimensions)**: Binary flags for immediate threats
   - `state[0]`: Danger straight ahead
   - `state[1]`: Danger to the right
   - `state[2]`: Danger to the left
   - `state[3]`: Danger behind

2. **Food Direction (4 dimensions)**: One-hot encoding of food direction
   - `state[4]`: Food is up
   - `state[5]`: Food is down
   - `state[6]`: Food is left
   - `state[7]`: Food is right

3. **Distance to Food (2 dimensions)**: Normalized coordinates
   - `state[8]`: Normalized x-distance to food
   - `state[9]`: Normalized y-distance to food

4. **Current Direction (4 dimensions)**: One-hot encoding of snake's movement
   - `state[10]`: Moving up
   - `state[11]`: Moving down
   - `state[12]`: Moving left
   - `state[13]`: Moving right

5. **Game Context (2 dimensions)**:
   - `state[14]`: Snake length normalized by grid area
   - `state[15]`: Steps without food normalized by 100

### 2. Deep Q-Learning Algorithm

#### Mathematical Foundation

The DQN algorithm approximates the optimal action-value function Q*(s,a) using the Bellman equation:

```
Q*(s,a) = E[R_t + γ * max_a' Q*(s_{t+1}, a') | s_t = s, a_t = a]
```

Where:
- `R_t` is the immediate reward
- `γ ∈ [0,1]` is the discount factor
- `s_t, a_t` are current state and action
- `s_{t+1}, a'` are next state and optimal next action

#### Loss Function
The network minimizes the temporal difference error:

```
L(θ) = E[(R_t + γ * max_a' Q(s_{t+1}, a'; θ^-) - Q(s_t, a_t; θ))^2]
```

Where:
- `θ` are current network parameters
- `θ^-` are target network parameters (updated periodically)

### 3. Training Algorithm

#### Main Training Loop
```cpp
for each episode:
    reset environment
    get initial state
    
    while not terminal:
        select action via ε-greedy policy
        execute action, observe reward and next_state
        store experience (s,a,r,s',done) in replay buffer
        
        if replay buffer has enough experiences:
            sample random minibatch
            perform gradient descent step
            
        if step % target_update_frequency == 0:
            update target network parameters
            
    decay exploration rate ε
```

#### Experience Replay
- **Buffer Size**: 50,000 experiences
- **Sampling**: Random minibatch of 128 experiences
- **Purpose**: Break temporal correlations, improve sample efficiency

#### Target Network
- **Update Frequency**: Every 50 training steps
- **Purpose**: Provide stable targets for TD-learning
- **Mechanism**: Copy weights from main network to target network

### 4. Reward Function Design

The reward function shapes the agent's behavior:

```cpp
float reward = 0.0f;

if (food_eaten) {
    reward += 10.0f;           // Primary reward
} else if (moved_closer_to_food) {
    reward += 0.1f;            // Shaping reward
} else if (moved_away_from_food) {
    reward -= 0.15f;           // Small penalty
}

if (game_over) {
    reward -= 10.0f;           // Strong penalty for death
}
```

### 5. Exploration Strategy

#### ε-Greedy Policy
```cpp
action = {
    random_action,     with probability ε
    argmax_a Q(s,a),   with probability 1-ε
}
```

#### Epsilon Decay
- **Start**: ε = 1.0 (100% exploration)
- **Decay**: ε ← ε * 0.998 per episode
- **Minimum**: ε = 0.01 (1% exploration)

### 6. Game Implementation

#### Grid System
- **Grid Size**: 12×12 cells
- **Cell Size**: 40×40 pixels
- **Total Game Area**: 500×500 pixels

#### Snake Representation
```cpp
std::deque<Point> snake;  // Front = head, Back = tail
Dir dir = Dir::RIGHT;     // Current movement direction
```

#### Collision Detection
- **Wall Collision**: Snake head outside grid bounds
- **Self Collision**: Head intersects with body segments
- **Timeout**: Too many steps without eating food

### 7. Rendering System

#### Four-Panel Layout
1. **Game Board** (500×500px): Main game visualization
2. **Statistics Panel** (700×500px): Training graphs and parameters
3. **Network Weights** (400×400px): Static network visualization
4. **Network Activity** (400×400px): Real-time forward pass visualization

#### Custom Bitmap Font
- **5×7 pixel characters** for all ASCII values
- **No external font dependencies**
- **Efficient SDL rendering**

#### Real-time Visualization Features
- **Neural Network Weights**: Color-coded connections (red=positive, green=negative)
- **Training Graphs**: Score history, average scores, epsilon decay
- **Live Network Activity**: Neuron activations during forward pass
- **Parameter Display**: Current hyperparameter values with adjustment hints

### 8. Interactive Parameter Tuning

#### Adjustable Parameters
1. **Learning Rate** (0.00001 - 0.1): Adam optimizer step size
2. **Gamma** (0.5 - 0.999): Discount factor for future rewards
3. **Epsilon Decay** (0.9 - 0.9999): Exploration rate decay
4. **Batch Size** (16 - 512): Mini-batch size
5. **Replay Buffer Size** (1000 - 100000): Experience storage
6. **Reward Food** (1.0 - 100.0): Food eating reward
7. **Reward Closer** (0.0 - 2.0): Moving closer reward
8. **Penalty Away** (-2.0 - 0.0): Moving away penalty
9. **Penalty Death** (-100.0 - -1.0): Death penalty
10. **Train Speed** (1 - 100): Training acceleration factor

#### Control Interface
- **↑/↓ Arrows**: Select parameter
- **←/→ Arrows**: Adjust selected parameter
- **R Key**: Reset all to defaults
- **Space**: Reset exploration (ε=1)
- **+/- Keys**: Adjust rendering FPS

### 9. Performance Optimizations

#### CUDA Acceleration
- **GPU Libraries**: libtorch_cuda.so, libc10_cuda.so
- **CUDA Runtime**: libcudart.so
- **NVRTC Compiler**: libnvrtc.so
- **Automatic GPU Selection**: Falls back to CPU if CUDA unavailable

#### Memory Management
- **Experience Replay**: Circular buffer with automatic overflow handling
- **Tensor Operations**: PyTorch automatic memory management
- **SDL Resources**: Proper cleanup in destructor

#### Training Speed Control
- **Render Skipping**: Adjust train_speed to skip expensive rendering
- **FPS Control**: Adjustable game_speed for visualization
- **Batch Processing**: Efficient mini-batch training

### 10. File Structure and Dependencies

#### Source Files
```
src/
├── SnakeAI.hpp     # Main AI class declaration (621 lines)
├── SnakeAI.cpp     # AI implementation
└── Utils.h         # Constants, structures, font data (577 lines)
```

#### Key Dependencies
- **PyTorch C++**: Deep learning framework
- **SDL3**: Graphics and window management
- **GLM**: Mathematics library
- **CUDA**: GPU acceleration (optional)

#### Build System
- **CMake 3.22+**: Build configuration
- **vcpkg**: Package management
- **GCC 9+/Clang 10+**: C++17 compilation

### 11. Advanced Features

#### Signal Handling
- **SIGINT Handler**: Non-destructive interruption for forced rendering
- **Graceful Shutdown**: Proper resource cleanup

#### Mathematical Precision
- **Float32**: Single precision for neural networks
- **Normalized Values**: All state features normalized to [0,1] or [-1,1]
- **Stable Training**: Target networks prevent divergence

#### Extensibility
- **Modular Design**: Easy to modify network architecture
- **Parameter System**: Runtime adjustment without recompilation
- **Visualization Framework**: Adaptable to different games

---

*This architecture document provides a comprehensive overview of the Reinforcement Learning Snake implementation, covering the mathematical foundations, algorithmic details, and engineering choices.*

#TODO 
load pre trained data.
