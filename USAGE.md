# Usage Guide and Training Instructions

## Getting Started

### Quick Start
```bash
cd /home/moinshaikh/CLionProjects/ReinforcementSnake/build
./ReinforcementSnake
```

The application will start immediately with the Snake AI training. You'll see four panels:
1. **Game Board** (left): Snake playing in real-time
2. **Training Statistics** (right): Graphs and parameters
3. **Network Weights** (bottom-left): Neural network visualization
4. **Network Activity** (bottom-right): Live forward pass visualization

## Training Controls

### Keyboard Controls

| Key | Function | Description |
|-----|----------|-------------|
| **↑/↓** | Select Parameter | Navigate through adjustable parameters |
| **←/→** | Adjust Parameter | Decrease/increase selected parameter |
| **R** | Reset Parameters | Reset all parameters to default values |
| **Space** | Reset Exploration | Set epsilon back to 1.0 (100% exploration) |
| **+/-** | Adjust Game Speed | Control rendering FPS (5-120) |
| **Ctrl+C** | Force Render | Show current state without stopping training |

### Parameter Adjustment During Training

#### Selectable Parameters
1. **Learning Rate** (0.00001 - 0.1)
   - Controls how fast the network learns
   - Higher = faster learning but risk of instability
   - Lower = stable but slower convergence

2. **Gamma** (0.5 - 0.999)
   - Discount factor for future rewards
   - Higher values encourage long-term planning
   - Lower values focus on immediate rewards

3. **Epsilon Decay** (0.9 - 0.9999)
   - How quickly exploration decreases
   - Higher values = more exploration, slower exploitation
   - Lower values = faster transition to exploitation

4. **Batch Size** (16 - 512)
   - Number of experiences per training step
   - Larger batches = more stable gradients
   - Smaller batches = faster updates but noisier

5. **Replay Buffer Size** (1000 - 100000)
   - Memory for past experiences
   - Larger = more diverse training data
   - Smaller = less memory usage

6. **Reward Food** (1.0 - 100.0)
   - Reward for eating food
   - Higher values strongly encourage food-seeking

7. **Reward Closer** (0.0 - 2.0)
   - Reward for moving toward food
   - Helps guide learning in right direction

8. **Penalty Away** (-2.0 - 0.0)
   - Penalty for moving away from food
   - Discourages inefficient movement

9. **Penalty Death** (-100.0 - -1.0)
   - Penalty for game over
   - Strong negative signal encourages survival

10. **Train Speed** (1 - 100)
    - How often episodes are rendered
    - 1 = render every episode (slowest)
    - 100 = render every 100th episode (fastest)

## Understanding the Visualization

### Game Board Panel
- **Green Square**: Snake head
- **Light Green**: Snake body
- **Red Square**: Food
- **Black**: Empty space
- **White Grid**: Game boundaries

### Statistics Panel
- **Score Graph**: Individual episode scores over time
- **Average Score Graph**: Running average (10-episode windows)
- **Epsilon Graph**: Exploration rate decay over time
- **Current Parameters**: Real-time parameter display
- **Training Info**: Episode count, current score, max score

### Network Weights Panel
- **Nodes**: Neurons in each layer
- **Connections**: Lines between neurons
- **Red Lines**: Positive weights
- **Green Lines**: Negative weights
- **Line Thickness**: Weight magnitude

### Network Activity Panel
- **Blue Nodes**: Current neuron activations
- **Brighter Colors**: Higher activation values
- **Q-Values**: Output values for each action
- **Selected Action**: Highlighted in real-time

## Training Process

### Training Phases

#### Phase 1: Pure Exploration (Episodes 1-100)
- **Epsilon**: 1.0 → ~0.82
- **Behavior**: Completely random movement
- **Goal**: Explore all possible states and actions
- **Expected Score**: 0-2 points per episode

#### Phase 2: Mixed Exploration (Episodes 100-500)
- **Epsilon**: ~0.82 → ~0.37
- **Behavior**: Mix of random and learned actions
- **Goal**: Start using learned knowledge while exploring
- **Expected Score**: 2-8 points per episode

#### Phase 3: Exploitation (Episodes 500+)
- **Epsilon**: <0.37
- **Behavior**: Mostly learned actions with some exploration
- **Goal**: Refine strategy and achieve high scores
- **Expected Score**: 10+ points per episode

### Training Tips

#### For Beginners
1. **Start with Default Parameters**: The defaults work well for most cases
2. **Watch the First 100 Episodes**: Understand random behavior first
3. **Don't Adjust Too Early**: Let the agent explore before tuning
4. **Focus on Score Trends**: Look at averages, not individual episodes

#### For Advanced Users
1. **Experiment with Learning Rate**: Try 0.0005 or 0.005 for different convergence
2. **Adjust Gamma**: 0.95 for more immediate rewards, 0.99 for long-term planning
3. **Modify Reward Structure**: Change reward_food to 5.0 or 20.0 for different behaviors
4. **Batch Size Tuning**: 64 for faster updates, 256 for stability

#### Common Training Patterns

**Good Training Indicators:**
- Average score steadily increases after episode 200
- Epsilon decay is smooth and gradual
- Network weights show clear patterns (not random)
- Agent consistently avoids walls after episode 300

**Problem Indicators:**
- Average score stays at 0-2 for >500 episodes
- Agent consistently hits walls
- Network weights remain random-looking
- No improvement in maximum score

### Performance Optimization

#### Speed Up Training
1. **Increase Train Speed**: Set to 50-100 to skip rendering
2. **Reduce Game Speed**: Set to minimum (5 FPS) when rendering
3. **Use GPU**: CUDA acceleration provides 5-10x speedup
4. **Close Other Applications**: Free up CPU/GPU resources

#### Memory Optimization
1. **Reduce Replay Buffer**: 10000 instead of 50000 if memory is limited
2. **Smaller Batch Size**: 64 instead of 128 reduces memory usage
3. **Monitor System Resources**: Use `htop` or `top` to track usage

## Saving and Loading Models

### Current Limitations
The base implementation doesn't include model saving/loading, but you can extend it:

#### Adding Model Saving
```cpp
// In SnakeAI class
void save_model(const std::string& filename) {
    torch::save(QNetwork, filename);
}

void load_model(const std::string& filename) {
    torch::load(QNetwork, filename);
    // Update target network to match
    targetNetwork = QNetwork;
}
```

#### Integration Points
- Add save/load calls in `train()` method
- Add keyboard shortcuts for saving (e.g., 'S' key)
- Save training statistics along with model weights

## Troubleshooting Training Issues

### Agent Not Learning

**Symptoms:**
- Score stays at 0-1 for many episodes
- Agent consistently hits walls
- No improvement in average score

**Solutions:**
1. **Increase Learning Rate**: Try 0.005 or 0.01
2. **Adjust Rewards**: Increase reward_food to 20.0
3. **Check Epsilon**: Ensure it's not decaying too fast
4. **Verify State Representation**: Make sure features are normalized

### Training Too Slow

**Symptoms:**
- Very low FPS during visualization
- Long time between episodes

**Solutions:**
1. **Increase Train Speed**: Set to 50-100
2. **Reduce Game Speed**: Set to minimum FPS
3. **Use CUDA**: Ensure GPU acceleration is working
4. **Skip Rendering**: Train in headless mode

### Unstable Training

**Symptoms:**
- Score fluctuates wildly
- Network weights change dramatically
- Training diverges

**Solutions:**
1. **Decrease Learning Rate**: Try 0.0005 or 0.0001
2. **Increase Batch Size**: Try 256 or 512
3. **Check Gradient Clipping**: Add gradient clipping if needed
4. **Verify Reward Scaling**: Ensure rewards aren't too large

## Advanced Usage

### Custom Training Scenarios

#### Short Training Sessions
```cpp
// In main.cpp
g.train(1000);  // Train for 1000 episodes
```

#### Continuous Training
```cpp
// Train indefinitely until user stops
while(true) {
    g.train(1000);
    // Optionally save model here
}
```

#### Comparative Training
```cpp
// Train with different parameters
TrainingParams params1 = g.params;
params1.learning_rate = 0.001;
g.params = params1;
g.train(1000);

TrainingParams params2 = g.params;
params2.learning_rate = 0.005;
g.params = params2;
g.train(1000);
```

### Performance Analysis

#### Tracking Metrics
- **Score per Episode**: Basic performance measure
- **Average Score (10 episodes)**: Stability measure
- **Epsilon Decay**: Exploration progress
- **Training Time**: Efficiency measure
- **Memory Usage**: Resource consumption

#### Expected Benchmarks
- **Good Performance**: Average score >15 after 1000 episodes
- **Excellent Performance**: Average score >25 after 2000 episodes
- **Training Time**: ~1-2 hours for 10000 episodes (with GPU)

---

*This usage guide provides comprehensive instructions for training and using the Reinforcement Learning Snake agent effectively.*
