/**
 * @file SnakeAI.hpp
 * @brief Deep Q-Network (DQN) Reinforcement Learning Implementation for Snake Game
 * @author moinshaikh
 * @date 1/17/26
 * 
 * This file implements a sophisticated Deep Q-Learning algorithm with experience replay,
 * target networks, and epsilon-greedy exploration for training an AI agent to play Snake.
 * 
 * Mathematical Foundation:
 * The DQN algorithm approximates the optimal action-value function Q*(s,a) using a neural
 * network, following the Bellman equation:
 * 
 * Q*(s,a) = E[R_t + γ * max_a' Q*(s_{t+1}, a') | s_t = s, a_t = a]
 * 
 * Where:
 * - R_t is the immediate reward
 * - γ ∈ [0,1] is the discount factor
 * - s_t, a_t are current state and action
 * - s_{t+1}, a' are next state and optimal next action
 * 
 * The loss function minimized during training:
 * L(θ) = E[(R_t + γ * max_a' Q(s_{t+1}, a'; θ^-) - Q(s_t, a_t; θ))^2]
 * 
 * Where θ are current network parameters and θ^- are target network parameters.
 */

#ifndef REINFORCEMENTSNAKE_SNAKEAI_HPP
#define REINFORCEMENTSNAKE_SNAKEAI_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<SDL3/SDL.h>
#include<vector>
#include<deque>
#include<thread>
#include<random>
#include<cmath>

#include"Utils.h"

/**
 * @brief Deep Q-Network Implementation for Snake Game
 * 
 * This neural network approximates the Q-function Q(s,a) which represents the expected
 * future reward when taking action 'a' in state 's'. The network architecture consists
 * of three fully connected layers with ReLU activations.
 * 
 * Network Architecture:
 * Input Layer: 16 neurons (state representation)
 * Hidden Layer 1: 128 neurons with ReLU activation
 * Hidden Layer 2: 128 neurons with ReLU activation  
 * Output Layer: 4 neurons (Q-values for UP, DOWN, LEFT, RIGHT actions)
 * 
 * Mathematical Representation:
 * h1 = ReLU(W1 * x + b1), where W1 ∈ ℝ^(128×16), b1 ∈ ℝ^128
 * h2 = ReLU(W2 * h1 + b2), where W2 ∈ ℝ^(128×128), b2 ∈ ℝ^128
 * Q = W3 * h2 + b3, where W3 ∈ ℝ^(4×128), b3 ∈ ℝ^4
 * 
 * The forward pass computes Q-values for all actions, enabling action selection
 * via argmax(Q) during exploitation or epsilon-greedy during exploration.
 */
struct QNetImpl : public torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    
    /**
     * @brief Constructor for Q-Network
     * 
     * Initializes three fully connected layers:
     * - fc1: Input (16) → Hidden (128)
     * - fc2: Hidden (128) → Hidden (128)  
     * - fc3: Hidden (128) → Output (4)
     */
    QNetImpl();
    
    /**
     * @brief Forward pass through the neural network
     * 
     * @param input Input tensor representing current state [batch_size, 16]
     * @return Output tensor of Q-values [batch_size, 4]
     * 
     * Computes Q(s,a) for all actions using the forward propagation:
     * Q = f3(ReLU(f2(ReLU(f1(x)))))
     * where f1, f2, f3 are linear transformations
     */
    torch::Tensor forward(torch::Tensor input);
};
TORCH_MODULE(QNet);

namespace RenSnake
{

    /**
     * @brief Main Snake AI Class implementing Deep Q-Learning
     * 
     * This class encapsulates the complete DQN training system including:
     * - Snake game environment simulation
     * - Neural network training and inference
     * - Experience replay buffer management
     * - Real-time visualization and parameter tuning
     * 
     * The agent learns to maximize cumulative reward through temporal difference
     * learning with experience replay and target network stabilization.
     */
    class SnakeAI
    {
    public:
        /**
         * @brief Constructor for Snake AI
         * 
         * @param Render Enable/disable real-time visualization
         * 
         * Initializes neural networks, optimizer, and training parameters.
         * Creates both main Q-network and target network with identical weights.
         * Sets up Adam optimizer with learning rate α = 0.001.
         */
        SnakeAI(bool Render);
        
        /**
         * @brief Destructor - cleans up SDL resources
         */
        ~SnakeAI();
        
        /**
         * @brief Initialize SDL rendering system and game state
         * 
         * @return true if initialization successful, false otherwise
         * 
         * Sets up SDL3 window and renderer if rendering enabled.
         * Initializes snake position and spawns first food item.
         */
        bool init();
        
        /**
         * @brief Main training loop for DQN agent
         * 
         * @param epochs Number of training episodes to run
         * 
         * Implements the complete DQN training algorithm:
         * 1. For each episode: reset environment, get initial state
         * 2. While not terminal: select action via ε-greedy, execute step
         * 3. Store experience (s,a,r,s',done) in replay buffer
         * 4. Sample random minibatch and perform gradient descent step
         * 5. Update target network every N steps
         * 6. Decay exploration rate ε
         * 
         * Training objective: minimize TD-error loss:
         * L(θ) = E[(y_j - Q(s_j,a_j;θ))^2]
         * where y_j = r_j + γ * max_a' Q(s'_j,a';θ^-)
         */
        void train(int epochs);
        
    private:
        /**
         * @brief Reset game environment to initial state
         * 
         * Clears snake deque, sets initial position at center,
         * resets direction to RIGHT, score to 0, and spawns food.
         */
        void reset();
        
        /**
         * @brief Convert signed direction enum to unsigned action index
         * 
         * @param signedDir Signed direction value (-2,-1,1,2)
         * @return Unsigned action index (0,1,2,3)
         * 
         * Mapping: UP(-1)→0, DOWN(1)→1, LEFT(-2)→2, RIGHT(2)→3
         */
        int serializeDirection(const int signedDir) const;
        
        /**
         * @brief Convert unsigned action index back to signed direction
         * 
         * @param unsighnedDir Unsigned action index (0,1,2,3)
         * @return Signed direction value (-2,-1,1,2)
         * 
         * Reverse mapping of serializeDirection()
         */
        int upackDirection(const int unsighnedDir) const;
        
        /**
         * @brief Execute one game step with given action
         * 
         * @param action Selected action (0=UP,1=DOWN,2=LEFT,3=RIGHT)
         * @param shouldprint Enable debug output
         * @return Immediate reward for this step
         * 
         * Reward function R(s,a,s'):
         * - +10.0: Eating food
         * - +0.1: Moving closer to food  
         * - -0.15: Moving away from food
         * - -10.0: Death (wall collision, self-collision, timeout)
         * 
         * Updates game state, checks termination conditions,
         * and calculates reward based on action outcome.
         */
        float step(int action, bool shouldprint);
        
        /**
         * @brief Main rendering function - orchestrates all visual components
         * 
         * @param state Current game state vector
         * @param epochs Current episode number
         * @param score Current episode score
         * 
         * Renders 4 panels: game board, training statistics,
         * network weights, and live network activity.
         */
        void render_all(const std::vector<float> & state,int epochs,int score);
        
        /**
         * @brief Render the Snake game board
         * 
         * @param x_off X offset for rendering position
         * @param y_off Y offset for rendering position
         * 
         * Draws grid, snake segments, food item, and current score.
         * Uses different colors for snake head vs body segments.
         */
        void render_game(int x_off, int y_off);
        
        /**
         * @brief Visualize neural network weights and architecture
         * 
         * @param x_off X offset
         * @param y_off Y offset  
         * @param w Width of visualization area
         * @param h Height of visualization area
         * 
         * Renders network layers as nodes with connections colored
         * by weight magnitude (red=positive, green=negative).
         * Shows up to 20 neurons per layer for clarity.
         */
        void render_network(int x_off, int y_off, int w, int h);
        
        /**
         * @brief Visualize real-time neural network activity
         * 
         * @param state Current state vector for forward pass
         * @param x_off X offset
         * @param y_off Y offset
         * @param w Width of visualization area  
         * @param h Height of visualization area
         * 
         * Performs forward pass and visualizes neuron activations
         * and weighted signal flow. Shows Q-values for each action
         * with highlighting of selected action.
         */
        void render_network_dynamic(const std::vector<float>& state, int x_off, int y_off, int w, int h);
        
        /**
         * @brief Render training statistics and control panel
         * 
         * @param x_off X offset
         * @param y_off Y offset
         * @param w Width of stats panel
         * @param h Height of stats panel
         * @param episode Current episode number
         * @param ep_score Current episode score
         * 
         * Displays three graphs: score history, average score,
         * and epsilon decay. Shows adjustable training parameters
         * with keyboard controls.
         */
        void render_stats(int x_off, int y_off, int w, int h, int episode, int ep_score);
        
        /**
         * @brief Render text using custom bitmap font
         * 
         * @param text String to render
         * @param x X position
         * @param y Y position
         * @param color RGB color values
         * @param scale Text scaling factor
         * 
         * Uses 5x7 bitmap font data from Utils.h for efficient
         * text rendering without external font dependencies.
         */
        void draw_text(const std::string& text, int x, int y, SDL_Color color, int scale = 1);
        
        /**
         * @brief Generate new food position at random empty location
         * 
         * Uses uniform random distribution to select coordinates
         * within grid bounds, ensuring food doesn't spawn on snake.
         */
        void spawn_food();
        
        /**
         * @brief Check if point is within game grid boundaries
         * 
         * @param p Point to check
         * @return true if point is valid grid position
         * 
         * Boundary check: 0 ≤ x ≤ COLS, 0 ≤ y ≤ ROWS
         */
        bool inside_grid(const Point &p) const;
        
        /**
         * @brief Check if point is occupied by snake body
         * 
         * @param p Point to check
         * @return true if point contains snake segment
         * 
         * Linear search through snake deque for collision detection.
         */
        bool snake_contains(const Point &p) const;
        
        /**
         * @brief Extract 16-dimensional state representation
         * 
         * @return State vector [danger(4), food_dir(4), distance(2), 
         *         direction(4), length(1), urgency(1)]
         * 
         * State encoding s ∈ ℝ^16:
         * s[0-3]: Binary danger indicators (wall/body in each direction)
         * s[4-7]: Food direction (one-hot: UP,DOWN,LEFT,RIGHT)
         * s[8-9]: Normalized distance to food (dx/COLS, dy/ROWS)
         * s[10-13]: Current direction (one-hot encoding)
         * s[14]: Snake length normalized by grid area
         * s[15]: Steps without food normalized by 100
         */
        std::vector<float> get_state() const;
        
        /**
         * @brief Select action using ε-greedy policy
         * 
         * @param state Current state vector
         * @return Selected action index (0-3)
         * 
         * Action selection policy:
         * a = {
         *   random action, with probability ε
         *   argmax_a Q(s,a), with probability 1-ε
         * }
         * 
         * During exploration, uniform random action enables discovery.
         * During exploitation, neural network selects optimal action.
         */
        int select_action(const std::vector<float>& state);
        
        /**
         * @brief Perform one gradient descent training step
         * 
         * Implements mini-batch gradient descent on TD-error loss.
         * Samples batch_size experiences from replay buffer.
         * 
         * For each experience (s,a,r,s',done):
         * Target y = {
         *   r, if done
         *   r + γ * max_a' Q(s',a';θ^-), otherwise
         * }
         * 
         * Loss: L = (1/N) * Σ(y - Q(s,a;θ))^2
         * 
         * Updates network parameters via backpropagation using Adam optimizer.
         * Periodically updates target network parameters (every 50 steps).
         */
        void train_step();

        /**
         * @brief SDL window handle for rendering
         * 
         * Main application window for visualization.
         * Created during init() if rendermode = true.
         */
        SDL_Window* window = nullptr;
        
        /**
         * @brief SDL renderer for 2D graphics
         * 
         * Handles all drawing operations for game board,
         * neural network visualization, and statistics.
         */
        SDL_Renderer* renderer = nullptr;

        /**
         * @brief Snake body represented as deque of points
         * 
         * Front element is snake head, back is tail.
         * Direction of movement controlled by dir enum.
         * Snake grows by adding to front when food eaten.
         */
        std::deque<Point> snake;
        
        /**
         * @brief Current movement direction
         * 
         * One of: UP(-1), DOWN(1), LEFT(-2), RIGHT(2)
         * Prevents 180° turns (opposite directions).
         */
        Dir dir = Dir::RIGHT;
        
        /**
         * @brief Current food position
         * 
         * Randomly positioned within grid bounds,
         * never overlapping with snake body.
         */
        Point food{0,0};
        
        /**
         * @brief Game termination flag
         * 
         * Set true on wall collision, self-collision,
         * or timeout (too many steps without food).
         */
        bool game_over = false;
        
        /**
         * @brief Current episode score
         * 
         * Increments by 1 for each food item consumed.
         * Used for tracking training progress.
         */
        int score = 0;
        
        /**
         * @brief Counter for steps without eating food
         * 
         * Triggers game over if exceeds threshold:
         * timeout = 50 + 10 * snake_length
         * Prevents infinite loops and encourages exploration.
         */
        int stepWithotfood = 0;
        
        /**
         * @brief Mersenne Twister random number generator
         * 
         * High-quality PRNG for:
         * - Action selection during exploration
         * - Food spawning
         * - Experience replay sampling
         * Seeded with system clock time.
         */
        std::mt19937 randomEngine;
        
        /**
         * @brief Main Q-network for action selection
         * 
         * Neural network approximating Q(s,a).
         * Continuously updated during training via gradient descent.
         * Used for action selection during both exploration and exploitation.
         */
        QNetImpl QNetwork;
        
        /**
         * @brief Target Q-network for stable training
         * 
         * Provides stable targets for TD-learning.
         * Parameters updated periodically (every 50 steps) from main network.
         * Prevents divergence caused by moving target problem.
         */
        QNetImpl targetNetwork;

        /**
         * @brief Adam optimizer for neural network training
         * 
         * Adaptive moment estimation optimizer with:
         * - Learning rate α = 0.001
         * - Beta1 = 0.9 (default)
         * - Beta2 = 0.999 (default)
         * - Epsilon = 1e-8 (default)
         * 
         * Efficiently minimizes TD-error loss via backpropagation.
         */
        torch::optim::Adam optimizer;

        /**
         * @brief Experience replay buffer
         * 
         * Circular buffer storing past experiences:
         * Experience = (state, action, reward, next_state, done)
         * 
         * Capacity: 50,000 experiences
         * Enables breaking temporal correlations and sample efficiency.
         * Random sampling provides i.i.d. training data.
         */
        std::deque<Experience> replay_buffer;
        
        /**
         * @brief Exploration rate for ε-greedy policy
         * 
         * Current probability of selecting random action.
         * Starts at ε_start = 1.0 (100% exploration)
         * Decays exponentially: ε ← ε * ε_decay each episode
         * Minimum value: ε_end = 0.01 (1% exploration)
         * 
         * Balances exploration vs exploitation during training.
         */
        float epsilon = initialConstants::EPSILON_START;
        
        /**
         * @brief Enable/disable real-time visualization
         * 
         * true: Full rendering with SDL3 (slower training)
         * false: Headless mode for faster training
         * Affects SDL initialization and all render_* functions.
         */
        bool rendermode;

        /**
         * @brief Total episodes completed
         * 
         * Incremented each time train() loop completes an episode.
         * Used for statistics tracking and epsilon decay calculation.
         */
        int episodeCount = 0;
        
        /**
         * @brief Total steps taken across all episodes
         * 
         * Cumulative counter of individual game actions.
         * Used for target network update frequency (every 50 steps).
         */
        int totalStep= 0;

        /**
         * @brief History of episode scores
         * 
         * Vector storing score of each completed episode.
         * Limited to 200 most recent episodes (FIFO behavior).
         * Used for calculating running averages and progress tracking.
         */
        std::vector<float> score_history;
        
        /**
         * @brief History of average scores (10-episode windows)
         * 
         * Stores running averages every 10 episodes.
         * Limited to 200 most recent averages.
         * Smoother indicator of training progress than individual scores.
         */
        std::vector<float> avg_score_history;
        
        /**
         * @brief History of exploration rates
         * 
         * Tracks epsilon values over time.
         * Limited to 200 most recent values.
         * Visualizes exploration-exploitation balance evolution.
         */
        std::vector<float> epsilon_history;
        
        /**
         * @brief Maximum score achieved so far
         * 
         * Tracks best performance across all episodes.
         * Updated when current episode exceeds previous maximum.
         * Key metric for training success.
         */
        int current_max_score = 0;
        
        /**
         * @brief Current average score (last 10 episodes)
         * 
         * Calculated every 10 episodes from recent performance.
         * More stable indicator than individual episode scores.
         * Used for progress assessment and parameter tuning.
         */
        float current_avg = 0;

        /**
         * @brief Game rendering speed (frames per second)
         * 
         * Controls visualization speed during training.
         * Range: 5-120 FPS (adjustable with +/- keys)
         * Higher values = faster visualization, lower = slower.
         * Only affects rendering, not training computation speed.
         */
        int game_speed = 15;  // FPS (adjustable with +/-)
        
        /**
         * @brief Training acceleration factor
         * 
         * Controls how often episodes are rendered:
         * 1 = render every episode (slowest)
         * 10 = render every 10th episode (default)
         * 100 = render every 100th episode (fastest)
         * 
         * Higher values dramatically speed up training by skipping
         * expensive rendering operations while still providing
         * periodic visualization updates.
         */
        int train_speed = 10;  // 1 = normal, higher = faster (skip renders)

        /**
         * @brief Adjustable training hyperparameters
         * 
         * Struct containing all tunable training parameters:
         * - learning_rate: Adam optimizer step size
         * - gamma: Discount factor for future rewards
         * - epsilon_decay: Exploration rate decay
         * - batch_size: Mini-batch size for gradient descent
         * - replay_buffer_size: Experience storage capacity
         * - reward_*: Reward function parameters
         * - penalty_*: Negative reward parameters
         * 
         * Can be adjusted in real-time via keyboard controls.
         */
        TrainingParams params;
        
        /**
         * @brief Index of currently selected parameter for adjustment
         * 
         * Range: 0-9 (10 adjustable parameters)
         * Used with Up/Down arrow keys to select which parameter
         * to modify with Left/Right arrow keys during training.
         */
        int selected_param = 0;  // Which parameter is selected for adjustment
    };



}



#endif //REINFORCEMENTSNAKE_SNAKEAI_HPP