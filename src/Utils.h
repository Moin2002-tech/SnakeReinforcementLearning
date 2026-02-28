/**
 * @file Utils.h
 * @brief Utility Constants, Data Structures, and Font Definitions for Snake AI
 * @author moinshaikh
 * @date 1/13/26
 * 
 * This header file contains all essential utilities for the Snake AI implementation:
 * - Window and game layout constants
 * - Neural network architecture parameters
 * - Training hyperparameters and reward structures
 * - Game state data structures (Point, Direction, Experience)
 * - Custom bitmap font for efficient text rendering
 * 
 * Design Philosophy:
 * All constants are constexpr for compile-time optimization.
 * Namespaces organize related constants logically.
 * Structures provide type-safe data handling for the DQN algorithm.
 */

#ifndef REINFORCEMENTSNAKE_UTILS_H
#define REINFORCEMENTSNAKE_UTILS_H

#include<vector>
#include<csignal>
#include <cstdint>
#include<array>

/**
 * @namespace RenderCount
 * @brief Signal handling for forced rendering during training
 * 
 * Provides mechanism to interrupt training loops and force immediate rendering
 * when SIGINT (Ctrl+C) is received. This allows users to see current state
 * without terminating the entire training process.
 */
namespace RenderCount
{
    /**
     * @brief Volatile flag for forced rendering
     * 
     * Set to 1 by signal handler when SIGINT received.
     * Volatile ensures compiler doesn't optimize away checks.
     * sig_atomic_t guarantees atomic read/write even in signal handlers.
     */
    inline volatile std::sig_atomic_t force_To_Render = 0;
    
    /**
     * @brief Signal handler for SIGINT (Ctrl+C)
     * 
     * @param signum Signal number (should be SIGINT)
     * 
     * Instead of terminating, sets force_To_Render flag to trigger
     * immediate visualization of current training state.
     * Allows non-destructive interruption during long training runs.
     */
    inline void signal_handler(int signum)
    {
        force_To_Render = 1;
    }
}

/**
 * @namespace window
 * @brief Window layout and rendering dimensions
 * 
 * Defines the complete UI layout with four main panels:
 * 1. Game board (left, square)
 * 2. Statistics panel (right, tall)
 * 3. Network weights visualization (bottom, left half)
 * 4. Network activity visualization (bottom, right half)
 */
namespace window
{
    /**
     * @brief Game board dimensions in pixels
     * 
     * Square area for the Snake game visualization.
     * Also determines grid size via gameConstants::CELL.
     */
    inline constexpr const int GAME_SIZE = 500;
    
    /**
     * @brief Statistics panel width in pixels
     * 
     * Tall panel on the right side showing training graphs,
     * current parameters, and performance metrics.
     */
    inline constexpr const int STAT_W = 700;
    
    /**
     * @brief Network visualization height in pixels
     * 
     * Height for both network weight and activity visualizations
     * at the bottom of the window.
     */
    inline constexpr const int NET_VIS_H = 400;
    
    /**
     * @brief Total window width
     * 
     * Calculated as: game board + network visualizations
     * GAME_SIZE (500) + NET_VIS_H (400) = 900 pixels
     */
    inline constexpr const int WINDOW_W = GAME_SIZE + NET_VIS_H;
    
    /**
     * @brief Total window height
     * 
     * Calculated as: game board + statistics panel
     * GAME_SIZE (500) + STAT_W (700) = 1200 pixels
     */
    inline constexpr const int WINDOW_H = GAME_SIZE + STAT_W;
}

/**
 * @namespace gameConstants
 * @brief Game grid and spatial constants
 * 
 * Defines the discrete grid system for the Snake game.
 * All game logic operates on this grid coordinate system.
 */
namespace gameConstants
{
    /**
     * @brief Size of each grid cell in pixels
     * 
     * Each cell is a 40x40 pixel square.
     * Determines visual scale and grid resolution.
     * Larger cells = easier visibility, smaller = more grid space.
     */
    inline constexpr int CELL = 40;
    
    /**
     * @brief Number of columns in game grid
     * 
     * Calculated: GAME_SIZE / CELL = 500 / 40 = 12.5 → 12 columns
     * Uses integer division, ensuring whole cells fit in game area.
     */
    inline constexpr int COLS = window::GAME_SIZE / CELL;
    
    /**
     * @brief Number of rows in game grid
     * 
     * Calculated: GAME_SIZE / CELL = 500 / 40 = 12.5 → 12 rows
     * Square grid provides symmetric gameplay area.
     */
    inline constexpr int ROWS = window::GAME_SIZE / CELL;
}

/**
 * @namespace neuralConstants
 * @brief Neural network architecture parameters
 * 
 * Defines the complete DQN network structure.
 * These parameters determine network capacity and learning capability.
 */
namespace neuralConstants
{
    /**
     * @brief Input layer size (state representation dimension)
     * 
     * 16-dimensional state vector encoding:
     * - 4 danger indicators (immediate threats)
     * - 4 food direction indicators (one-hot)
     * - 2 normalized distance to food
     * - 4 current direction indicators (one-hot)
     * - 1 snake length (normalized)
     * - 1 urgency counter (steps without food)
     * 
     * This compact representation captures all essential game information
     * for effective decision making while keeping network manageable.
     */
    inline constexpr int inputSize = 16;
    
    /**
     * @brief Hidden layer sizes
     * 
     * 128 neurons in each hidden layer provides good balance:
     * - Sufficient capacity to learn complex strategies
     * - Not so large as to cause overfitting or slow training
     * - Common choice for medium-sized reinforcement learning problems
     * 
     * Two hidden layers enable learning hierarchical features:
     * Layer 1: Basic pattern recognition from raw state
     * Layer 2: Higher-level strategic concepts
     */
    inline constexpr int HIDDEN_SIZE = 128;
    
    /**
     * @brief Output layer size (action space)
     * 
     * 4 outputs corresponding to possible actions:
     * - Output 0: Move UP
     * - Output 1: Move DOWN  
     * - Output 2: Move LEFT
     * - Output 3: Move RIGHT
     * 
     * Each output represents Q(s,a) - expected future reward
     * for taking action 'a' in current state 's'.
     */
    inline constexpr int OUTPUT_SIZE = 4;
}

/**
 * @namespace initialConstants
 * @brief Initial training hyperparameters
 * 
 * Starting values for key training parameters.
 * These can be adjusted during runtime via keyboard controls.
 */
namespace initialConstants
{
    /**
     * @brief Initial exploration rate (ε)
     * 
     * Starts at 1.0 (100% random actions) to encourage
     * thorough exploration of the state-action space.
     * Prevents premature convergence to suboptimal policies.
     */
    inline constexpr float EPSILON_START = 1.0f;
    
    /**
     * @brief Initial Adam optimizer learning rate (α)
     * 
     * 0.001 is a standard starting point for deep learning.
     * Balances:
     * - Fast enough learning for reasonable training time
     * - Small enough to avoid divergence and instability
     * - Can be adjusted during training for fine-tuning
     */
    inline constexpr float INITIAL_LEARNING_RATE = 0.001f;
}

/**
 * @struct TrainingParams
 * @brief Complete set of adjustable training hyperparameters
 * 
 * Contains all parameters that can be modified during training
 * via keyboard controls. This enables real-time hyperparameter
 * tuning without restarting the training process.
 * 
 * Default values are chosen based on common DQN practices
 * and empirical testing for the Snake game environment.
 */
struct TrainingParams
{
    /**
     * @brief Learning rate for Adam optimizer
     * 
     * Controls step size in parameter updates.
     * Range: 0.00001 to 0.1
     * Default: 0.001
     * 
     * Higher values = faster learning but risk instability
     * Lower values = stable but slower convergence
     */
    float learning_rate = 0.001f;
    
    /**
     * @brief Discount factor for future rewards (γ)
     * 
     * Determines importance of future vs immediate rewards.
     * Range: 0.5 to 0.999
     * Default: 0.99
     * 
     * γ = 0: Only immediate rewards matter
     * γ = 1: Future rewards equally important as immediate
     * 
     * High values encourage long-term planning in Snake.
     */
    float gamma = 0.99f;
    
    /**
     * @brief Epsilon decay rate per episode
     * 
     * Controls exploration rate decay: ε ← ε * ε_decay
     * Range: 0.9 to 0.9999
     * Default: 0.998
     * 
     * Higher values = slower decay, more exploration
     * Lower values = faster decay, quicker exploitation
     */
    float epsilon_decay = 0.998f;
    
    /**
     * @brief Minimum exploration rate
     * 
     * Floor value for epsilon decay.
     * Ensures some exploration continues indefinitely.
     * Default: 0.01 (1% exploration)
     */
    float epsilon_end = 0.01f;
    
    /**
     * @brief Mini-batch size for gradient descent
     * 
     * Number of experiences sampled per training step.
     * Range: 16 to 512
     * Default: 128
     * 
     * Larger batches = more stable gradients but slower
     * Smaller batches = faster but noisier updates
     */
    int batch_size = 128;
    
    /**
     * @brief Experience replay buffer capacity
     * 
     * Maximum number of past experiences stored.
     * Default: 50,000
     * 
     * Larger buffers = more diverse training data
     * but increased memory usage and sampling time.
     */
    int replay_buffer_size = 50000;
    
    /**
     * @brief Reward for eating food
     * 
     * Primary positive reinforcement signal.
     * Range: 1.0 to 100.0
     * Default: 10.0
     * 
     * Higher values strongly encourage food-seeking behavior.
     */
    float reward_food = 10.0f;
    
    /**
     * @brief Reward for moving closer to food
     * 
     * Shaping reward to guide learning.
     * Range: 0.0 to 2.0
     * Default: 0.1
     * 
     * Positive value encourages efficient pathfinding.
     */
    float reward_closer = 0.1f;
    
    /**
     * @brief Penalty for moving away from food
     * 
     * Discourages inefficient movement patterns.
     * Range: -2.0 to 0.0
     * Default: -0.15
     * 
     * Negative value encourages purposeful movement.
     */
    float penalty_away = -0.15f;
    
    /**
     * @brief Penalty for game termination
     * 
     * Strong negative signal for death events.
     * Range: -100.0 to -1.0
     * Default: -10.0
     * 
     * Triggers on: wall collision, self-collision, timeout
     * Encourages survival behavior above all else.
     */
    float penalty_death = -10.0f;
};

/**
 * @enum class Dir
 * @brief Movement direction enumeration
 * 
 * Uses signed integers to enable mathematical operations:
 * - Opposite directions sum to 0 (enabling 180° turn detection)
 * - Positive values: vertical movement (UP= -1, DOWN= 1)
 * - Negative values: horizontal movement (LEFT= -2, RIGHT= 2)
 * 
 * This design allows simple collision detection:
 * if ((int)oldDir + (int)newDir == 0) // 180° turn
 */
enum class Dir
{
    UP = -1,     ///< Move up (decrease y)
    DOWN = 1,    ///< Move down (increase y)
    LEFT = -2,   ///< Move left (decrease x)
    RIGHT = 2    ///< Move right (increase x)
};

/**
 * @struct Point
 * @brief 2D integer coordinate for grid positions
 * 
 * Represents positions on the game grid.
 * Used for snake segments, food location, and collision detection.
 * 
 * Integer coordinates align with discrete grid system.
 * Equality operator enables efficient membership testing.
 */
struct Point
{
    int x, y; ///< Grid coordinates (0 ≤ x < COLS, 0 ≤ y < ROWS)
    
    /**
     * @brief Equality comparison operator
     * 
     * @param o Other point to compare against
     * @return true if coordinates match exactly
     * 
     * Enables efficient collision detection and food consumption checks.
     * Used extensively in game logic and state representation.
     */
    bool operator==(Point const & o) const
    {
        return x == o.x && y == o.y;
    }
};

/**
 * @struct Experience
 * @brief Single transition tuple for experience replay
 * 
 * Represents one step in the environment for DQN training.
 * Stored in replay buffer and sampled for mini-batch training.
 * 
 * Follows standard reinforcement learning notation:
 * (s, a, r, s', done) where:
 * - s: current state
 * - a: action taken
 * - r: reward received
 * - s': next state
 * - done: episode termination flag
 */
struct Experience
{
    std::vector<float> state;      ///< Current state vector (16-dimensional)
    int action;                    ///< Action taken (0-3: UP,DOWN,LEFT,RIGHT)
    float reward;                  ///< Immediate reward received
    std::vector<float> next_state; ///< Next state vector after action
    bool done;                     ///< True if episode terminated
};

/**
 * @struct FONT
 * @brief Custom bitmap font generator for efficient text rendering
 * 
 * Generates a 5x7 pixel bitmap font for ASCII characters 0-127.
 * Eliminates dependency on external font libraries and ensures
 * consistent rendering across different systems.
 * 
 * Each character is represented as 7 bytes, with each byte
 * representing one row of 5 pixels (using lower 5 bits).
 * Bit pattern: 0bXXXXX where X = pixel (1=on, 0=off)
 * 
 * This approach provides:
 * - Fast rendering without text layout complexity
 * - Consistent appearance across platforms
 * - Minimal memory footprint
 * - Easy integration with SDL rendering
 */
struct FONT
{
    /**
     * @brief Generate complete bitmap font data
     * 
     * @return 2D array containing 128 characters × 7 rows
     * 
     * Creates bitmap representations for:
     * - Digits (0-9)
     * - Uppercase letters (A-Z)
     * - Lowercase letters (a-z)
     * - Common punctuation and symbols
     * - Space character
     * 
     * Each character is carefully designed for readability
     * at small scales typical of game UI elements.
     */
    static constexpr std::array<std::array<uint8_t, 7>, 128> generate()
    {
        std::array<std::array<uint8_t, 7>, 128> FONT_DATA = {};
        
        // Digits 0-9: Clear, bold designs for score display
        FONT_DATA['0'] = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E};
        FONT_DATA['1'] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
        FONT_DATA['2'] = {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F};
        FONT_DATA['3'] = {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E};
        FONT_DATA['4'] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
        FONT_DATA['5'] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E};
        FONT_DATA['6'] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E};
        FONT_DATA['7'] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
        FONT_DATA['8'] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
        FONT_DATA['9'] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C};
        
        // Common punctuation for UI elements
        FONT_DATA['.'] = {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C};
        FONT_DATA[':'] = {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00};
        FONT_DATA['-'] = {0x00,0x00,0x00,0x1F,0x00,0x00,0x00};
        FONT_DATA['+'] = {0x00,0x04,0x04,0x1F,0x04,0x04,0x00};
        FONT_DATA['%'] = {0x18,0x19,0x02,0x04,0x08,0x13,0x03};
        FONT_DATA['/'] = {0x01,0x02,0x02,0x04,0x08,0x08,0x10};
        
        // Uppercase letters: Bold, clear designs for headers
        FONT_DATA['A'] = {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11};
        FONT_DATA['B'] = {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E};
        FONT_DATA['C'] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E};
        FONT_DATA['D'] = {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C};
        FONT_DATA['E'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F};
        FONT_DATA['F'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10};
        FONT_DATA['G'] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F};
        FONT_DATA['H'] = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11};
        FONT_DATA['I'] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E};
        FONT_DATA['J'] = {0x07,0x02,0x02,0x02,0x02,0x12,0x0C};
        FONT_DATA['K'] = {0x11,0x12,0x14,0x18,0x14,0x12,0x11};
        FONT_DATA['L'] = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F};
        FONT_DATA['M'] = {0x11,0x1B,0x15,0x15,0x11,0x11,0x11};
        FONT_DATA['N'] = {0x11,0x11,0x19,0x15,0x13,0x11,0x11};
        FONT_DATA['O'] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
        FONT_DATA['P'] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
        FONT_DATA['Q'] = {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D};
        FONT_DATA['R'] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
        FONT_DATA['S'] = {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E};
        FONT_DATA['T'] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
        FONT_DATA['U'] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E};
        FONT_DATA['V'] = {0x11,0x11,0x11,0x11,0x11,0x0A,0x04};
        FONT_DATA['W'] = {0x11,0x11,0x11,0x15,0x15,0x15,0x0A};
        FONT_DATA['X'] = {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11};
        FONT_DATA['Y'] = {0x11,0x11,0x11,0x0A,0x04,0x04,0x04};
        FONT_DATA['Z'] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F};
        
        // Lowercase letters: Compact designs for detailed text
        FONT_DATA['a'] = {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F};
        FONT_DATA['b'] = {0x10,0x10,0x16,0x19,0x11,0x11,0x1E};
        FONT_DATA['c'] = {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E};
        FONT_DATA['d'] = {0x01,0x01,0x0D,0x13,0x11,0x11,0x0F};
        FONT_DATA['e'] = {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E};
        FONT_DATA['f'] = {0x06,0x09,0x08,0x1C,0x08,0x08,0x08};
        FONT_DATA['g'] = {0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E};
        FONT_DATA['h'] = {0x10,0x10,0x16,0x19,0x11,0x11,0x11};
        FONT_DATA['i'] = {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E};
        FONT_DATA['j'] = {0x02,0x00,0x06,0x02,0x02,0x12,0x0C};
        FONT_DATA['k'] = {0x10,0x10,0x12,0x14,0x18,0x14,0x12};
        FONT_DATA['l'] = {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E};
        FONT_DATA['m'] = {0x00,0x00,0x1A,0x15,0x15,0x11,0x11};
        FONT_DATA['n'] = {0x00,0x00,0x16,0x19,0x11,0x11,0x11};
        FONT_DATA['o'] = {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E};
        FONT_DATA['p'] = {0x00,0x00,0x1E,0x11,0x1E,0x10,0x10};
        FONT_DATA['q'] = {0x00,0x00,0x0D,0x13,0x0F,0x01,0x01};
        FONT_DATA['r'] = {0x00,0x00,0x16,0x19,0x10,0x10,0x10};
        FONT_DATA['s'] = {0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E};
        FONT_DATA['t'] = {0x08,0x08,0x1C,0x08,0x08,0x09,0x06};
        FONT_DATA['u'] = {0x00,0x00,0x11,0x11,0x11,0x13,0x0D};
        FONT_DATA['v'] = {0x00,0x00,0x11,0x11,0x11,0x0A,0x04};
        FONT_DATA['w'] = {0x00,0x00,0x11,0x11,0x15,0x15,0x0A};
        FONT_DATA['x'] = {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11};
        FONT_DATA['y'] = {0x00,0x00,0x11,0x11,0x0F,0x01,0x0E};
        FONT_DATA['z'] = {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F};
        
        // Space and special characters
        FONT_DATA[' '] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00};
        FONT_DATA['['] = {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E};
        FONT_DATA[']'] = {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E};
        FONT_DATA['('] = {0x02,0x04,0x08,0x08,0x08,0x04,0x02};
        FONT_DATA[')'] = {0x08,0x04,0x02,0x02,0x02,0x04,0x08};
        FONT_DATA['<'] = {0x02,0x04,0x08,0x10,0x08,0x04,0x02};
        FONT_DATA['>'] = {0x08,0x04,0x02,0x01,0x02,0x04,0x08};
        FONT_DATA['='] = {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00};
        FONT_DATA['|'] = {0x04,0x04,0x04,0x04,0x04,0x04,0x04};

        return FONT_DATA;
    }
};

/**
 * @brief Global font data constant
 * 
 * Compile-time generated bitmap font data.
 * Accessible throughout the application for text rendering.
 * 
 * Usage: FONT_DATA[char][row] returns byte representing
 * one row of 5 pixels for the specified character.
 */
inline constexpr auto FONT_DATA = FONT::generate();

#endif //REINFORCEMENTSNAKE_UTILS_H