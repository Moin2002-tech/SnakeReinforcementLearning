# Libtorch Linux source file
https://16sxj-my.sharepoint.com/:f:/g/personal/moinshaikhofficial_16sxj_onmicrosoft_com/IgBP8joNcEsFSo682GyFu0e0AZ7SOoiWAm8axEll3sfQGnU?e=3afds2


# vcpkg 

https://16sxj-my.sharepoint.com/:u:/g/personal/moinshaikhofficial_16sxj_onmicrosoft_com/IQC_ku7OVWghRakE-SQ7fYItAeOHs3tJ2J2UlaXJx1KzfOM?e=3BPkRT

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
