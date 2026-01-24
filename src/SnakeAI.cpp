//
// Created by moinshaikh on 1/19/26.
//
#include"SnakeAI.hpp"
#include<SDL3/SDL.h>
#include<SDL3/SDL_init.h>
#include<algorithm>


QNetImpl::QNetImpl()
{
    fc1 =  register_module("fc1",torch::nn::Linear(neuralConstants::inputSize,neuralConstants::HIDDEN_SIZE));
    fc2 = register_module("fc2",torch::nn::Linear(neuralConstants::HIDDEN_SIZE,neuralConstants::HIDDEN_SIZE));
    fc3 = register_module("fc3",torch::nn::Linear(neuralConstants::HIDDEN_SIZE,neuralConstants::OUTPUT_SIZE));
}

torch::Tensor QNetImpl::forward(torch::Tensor input)
{
    input =  torch::relu(fc1->forward(input));
    input =  torch::relu(fc2->forward(input));
    input = fc3->forward(input);
    return input;

}


namespace RenSnake
{
    SnakeAI::SnakeAI(bool Render) :
    rendermode(Render) ,
    QNetwork(QNetImpl()),
    targetNetwork(QNetImpl()), optimizer(QNetwork.parameters(),torch::optim::AdamOptions(initialConstants::INITIAL_LEARNING_RATE))
    {
        randomEngine.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
        torch::NoGradGuard noGradGuard;
        for (size_t i=0;i<QNetwork.parameters().size();++i)
            {
                targetNetwork.parameters()[i].copy_(QNetwork.parameters()[i]);
            }
    }

    SnakeAI::~SnakeAI()
    {
        if(renderer) SDL_DestroyRenderer(renderer);
        if(window) SDL_DestroyWindow(window);
        if(rendermode) SDL_Quit();
    }

    bool SnakeAI::init()
    {
        if (rendermode)
        {
            // 1. SDL_Init now returns true on success, false on failure
            if (!SDL_Init(SDL_INIT_VIDEO))
            {
                std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
                return false;
            }

            // 2. SDL_CreateWindow now takes title, width, height, and flags.
            // Note: The window position is handled automatically or via flags in SDL3.
            window = SDL_CreateWindow("Snake AI - Neural Network Training",
                                       window::WINDOW_W, window::WINDOW_H, 0);

            if (!window) {
                std::cerr << "Window creation failed: " << SDL_GetError() << "\n";
                return false;
            }

            // 3. SDL_CreateRenderer logic:
            // Passing nullptr as the second argument uses the default driver.
            renderer = SDL_CreateRenderer(window, nullptr);

            if (!renderer) {
                std::cerr << "Renderer creation failed: " << SDL_GetError() << "\n";
                return false;
            }

            // 4. Set blend mode to enable the transparency you use in your network lines
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        }

        SnakeAI::reset();
        return true;
    }
    void SnakeAI::reset()
    {
        snake.clear();
        int x =gameConstants::COLS/2, y = gameConstants::ROWS/2;
        snake.push_back({x,y});
        snake.push_back({x-1,y});
        snake.push_back({x-2,y});
        dir = Dir::RIGHT;
        score = 0;
        game_over = false;
        stepWithotfood = 0;
        spawn_food();
    }

    void SnakeAI::spawn_food()
    {
        std::uniform_int_distribution<int>dx(0,gameConstants::COLS-1),dy(0,gameConstants::ROWS-1);
        Point p;
        do
        {
            p.x = dx(randomEngine);
            p.y = dy(randomEngine);
        }
        while (snake_contains(p));
        food = p;
    }

    int SnakeAI::serializeDirection(const int signedDir) const
    {
        switch (signedDir)
        {
            case -1: return 0;
            case 1: return 1;
            case -2: return 2;
            case 2: return 3;
        }
        return 0;
    }

    int SnakeAI::upackDirection(const int unsignedDir) const
    {
        switch(unsignedDir)
        {
            case 0: return -1;
            case 1: return 1;
            case 2: return -2;
            case 3: return 2;
        }
        return 0;
    }

    bool SnakeAI::inside_grid(const Point &p) const
    {
        return p.x >= 0 &&p.x<=gameConstants::COLS && p.y >= 0 &&p.y<=gameConstants::ROWS;
    }

    bool SnakeAI::snake_contains(const Point &p) const
    {
        for (auto&s: snake)
           {
            if (s==p)
               return true;
           }
        return false;
    }

    std::vector<float> SnakeAI::get_state() const
    {
        std::vector<float> state(neuralConstants::inputSize,0.0f);
        Point history = snake.front();

        // Danger detection (immediate) - is there wall or body in each direction?

        //UP
        Point up{history.x,history.y-1};
        state[0] = (!inside_grid(up) || snake_contains(up)) ? 1.0f : 0.0f;
        // DOWN
        Point down{history.x, history.y + 1};
        state[1] = (!inside_grid(down) || snake_contains(down)) ? 1.0f : 0.0f;
        // LEFT
        Point left{history.x - 1, history.y};
        state[2] = (!inside_grid(left) || snake_contains(left)) ? 1.0f : 0.0f;
        // RIGHT
        Point right{history.x + 1, history.y};
        state[3] = (!inside_grid(right) || snake_contains(right)) ? 1.0f : 0.0f;

        // Food direction (relative to head) - ALWAYS visible
        state[4] = (food.y < history.y) ? 1.0f : 0.0f;  // Food is UP
        state[5] = (food.y > history.y) ? 1.0f : 0.0f;  // Food is DOWN
        state[6] = (food.x < history.x) ? 1.0f : 0.0f;  // Food is LEFT
        state[7] = (food.x > history.x) ? 1.0f : 0.0f;  // Food is RIGHT

        // Normalized distance to food
        state[8] = (history.x - food.x) / (float)gameConstants::COLS;  // X distance (-1 to 1)
        state[9] = (history.y - food.y) / (float)gameConstants::ROWS;  // Y distance (-1 to 1)

        // Current direction (one-hot)
        state[10] = (dir == Dir::UP) ? 1.0f : 0.0f;
        state[11] = (dir == Dir::DOWN) ? 1.0f : 0.0f;
        state[12] = (dir == Dir::LEFT) ? 1.0f : 0.0f;
        state[13] = (dir == Dir::RIGHT) ? 1.0f : 0.0f;

        // Snake length normalized
        state[14] = snake.size() / (float)(gameConstants::COLS * gameConstants::ROWS);

        // Steps without food (urgency)
        state[15] = std::min(1.0f, stepWithotfood / 100.0f);

        return state;
    }


    int SnakeAI::select_action(const std::vector<float> &state)
    {
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        if (distribution(randomEngine) < epsilon)
        {
        std::uniform_int_distribution<int> ad(0, 3);
            return ad(randomEngine);
        }
        torch::Tensor input = torch::tensor(state).reshape({1,neuralConstants::inputSize});
        torch::Tensor out = QNetwork.forward(input);
        return out.argmax(1).item<int> ();
    }
float SnakeAI::step(int action, bool shouldPrint)
    {
     if (game_over) {
         return 0;
     }
        Dir oldDir = dir;
        dir =  (Dir)SnakeAI::upackDirection(action);
        Point h = snake.front();
        Point nh = h;
        if ((int)oldDir + (int)dir == 0)
        {
            dir = oldDir;
        }
        if (dir == Dir::UP)
        {
            nh.y--;
        }
        else if (dir == Dir::DOWN)
        {
            nh.y++;
        }
        else if (dir == Dir::LEFT)
        {
        nh.x--;
        }
        else {
            nh.x++;
        }
        // Calculate distance to food before and after move
        int old_dist = std::abs(h.x - food.x) + std::abs(h.y - food.y);
        int new_dist = std::abs(nh.x - food.x) + std::abs(nh.y - food.y);

        float reward = 0.0f;
        stepWithotfood++;

        if(!inside_grid(nh)){
            game_over = true;
            return params.penalty_death;
        }
        if(snake_contains(nh)){
            game_over = true;
            return params.penalty_death;
        }
        if(stepWithotfood > 50 + 10 * (int)snake.size()){  // Shorter timeout for smaller grid
            game_over = true;
            return params.penalty_death * 0.5f;
        }

        bool ate = (nh == food);
        snake.push_front(nh);
        if(!ate)
            {
            snake.pop_back();
            // Reward for moving toward food, penalty for moving away
            if(new_dist < old_dist)
            {
                reward = params.reward_closer;
            } else

            {
                reward = params.penalty_away;
            }
        }
        else
        {
            reward = params.reward_food;
            score++;
            stepWithotfood = 0;
            spawn_food();
        }
        return reward;
    }


    void SnakeAI::train_step()
    {
        if((int)replay_buffer.size() < params.batch_size) return;
        std::uniform_int_distribution<size_t> d(0, replay_buffer.size()-1);

        std::vector<torch::Tensor> states, targets;
        for(int i = 0; i < params.batch_size; i++){
            const auto& e = replay_buffer[d(randomEngine)];
            torch::Tensor s = torch::tensor(e.state);
            torch::Tensor q = QNetwork.forward(s);
            torch::Tensor t = q.detach().clone();

            if(e.done) t[e.action] = e.reward;
            else {
                torch::Tensor ns = torch::tensor(e.next_state);
                auto nq = targetNetwork.forward(ns);
                float mx = nq.max().item<float>();
                t[e.action] = e.reward + params.gamma * mx;
            }
            states.push_back(s);
            targets.push_back(t);
        }

        torch::Tensor batch_s = torch::stack(states);
        torch::Tensor batch_t = torch::stack(targets);

        optimizer.zero_grad();
        torch::Tensor out = QNetwork.forward(batch_s);
        torch::Tensor loss = torch::mse_loss(out, batch_t);
        loss.backward();
        optimizer.step();

        static int c = 0; c++;
        if(c % 50 == 0)
        {
                torch::NoGradGuard no_grad;
                for(size_t i = 0; i < QNetwork.parameters().size(); i++)
                {
                    targetNetwork.parameters()[i].copy_(QNetwork.parameters()[i]);
                }
        }
    }

    void SnakeAI::draw_text(const std::string &text, int x, int y, SDL_Color color, int scale) {
        SDL_SetRenderDrawColor(renderer,color.r, color.g, color.b, color.a);
        int cursor_x = x;
        for (char c : text)
        {
            if (c>=0 && c < 128)
            {
                const uint8_t* glyph = FONT_DATA[static_cast<size_t>(c)].data();
                for(int row = 0; row < 7; row++)
                {
                    for(int col = 0; col < 5; col++)
                    {
                        if(glyph[row] & (0x10 >> col))
                        {
                            // Change SDL_Rect
                            SDL_FRect r = { (float)cursor_x + col * scale, (float)y + row * scale, (float)scale, (float)scale };
                            SDL_RenderFillRect(renderer, &r);
                        }
                    }
                }
            }
            cursor_x += 6 * scale;
        }
    }

   void SnakeAI::render_game(int x_off, int y_off)
    {
        SDL_SetRenderDrawColor(renderer, 25, 25, 25, 255);

        // Fix 1: Cast ints to floats to avoid narrowing conversion error
        SDL_FRect bg = { (float)x_off, (float)y_off, (float)window::GAME_SIZE, (float)window::GAME_SIZE };
        SDL_RenderFillRect(renderer, &bg);

        // Grid
        SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
        for(int gx = 0; gx <= gameConstants::COLS; gx++){
            float x_pos = (float)(x_off + gx * gameConstants::CELL);
            // Fix 2: SDL_RenderDrawLine is now SDL_RenderLine
            SDL_RenderLine(renderer, x_pos, (float)y_off, x_pos, (float)(y_off + window::GAME_SIZE));
        }
        for(int gy = 0; gy <= gameConstants::ROWS; gy++){
            float y_pos = (float)(y_off + gy * gameConstants::CELL);
            SDL_RenderLine(renderer, (float)x_off, y_pos, (float)(x_off + window::GAME_SIZE), y_pos);
        }

        // Food
        SDL_SetRenderDrawColor(renderer, 200, 50, 50, 255);
        // Fix 3: Use SDL_FRect instead of SDL_Rect
        SDL_FRect fr = {
            (float)(x_off + food.x * gameConstants::CELL + 2),
            (float)(y_off + food.y * gameConstants::CELL + 2),
            (float)(gameConstants::CELL - 4),
            (float)(gameConstants::CELL - 4)
        };
        SDL_RenderFillRect(renderer, &fr);

        // Snake
        bool first = true;
        for(auto& s : snake){
            SDL_FRect r = {
                (float)(x_off + s.x * gameConstants::CELL + 1),
                (float)(y_off + s.y * gameConstants::CELL + 1),
                (float)(gameConstants::CELL - 2),
                (float)(gameConstants::CELL - 2)
            };

            if(first){
                SDL_SetRenderDrawColor(renderer, 90, 200, 90, 255);
                first = false;
            } else {
                SDL_SetRenderDrawColor(renderer, 30, 160, 30, 255);
            }
            SDL_RenderFillRect(renderer, &r);
        }

        // Score on game
        std::string score_text = "Score: " + std::to_string(score);
        draw_text(score_text, x_off + 10, y_off + 10, {255, 255, 255, 255}, 2);
    }

   void SnakeAI::render_stats(int x_off, int y_off, int w, int h, int episode, int ep_score)
    {
        // 1. Background (Use SDL_FRect and explicit float casts)
        SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255);
        SDL_FRect bg = { (float)x_off, (float)y_off, (float)w, (float)h };
        SDL_RenderFillRect(renderer, &bg);

        int margin = 15;
        int graph_h = 65;
        int label_h = 15;

        // Title
        draw_text("TRAINING STATISTICS", x_off + margin, y_off + 8, {255, 255, 255, 255}, 2);

        // Current stats row
        int stats_y = y_off + 35;
        std::ostringstream oss;
        oss << "Episode: " << episode << "  Max: " << current_max_score << "  Avg: " << std::fixed << std::setprecision(1) << current_avg;
        draw_text(oss.str(), x_off + margin, stats_y, {200, 200, 200, 255}, 1);

        oss.str(""); oss.clear();
        oss << "Epsilon: " << std::fixed << std::setprecision(3) << epsilon << "  Speed: " << game_speed << " FPS  Train: " << train_speed << "x";
        draw_text(oss.str(), x_off + margin, stats_y + 12, {150, 150, 150, 255}, 1);

        // Parameter controls panel
        int ctrl_x = x_off + w/2 + 20;
        int ctrl_y = y_off + 8;
        draw_text("CONTROLS [Up/Down=select, Left/Right=adjust]", ctrl_x, ctrl_y, {255, 200, 100, 255}, 1);
        ctrl_y += 18;

        const char* param_names[] =
        {
            "Game Speed (FPS)", "Train Speed (x)", "Learning Rate", "Gamma (discount)",
            "Epsilon Decay", "Batch Size", "Reward: Food", "Reward: Closer",
            "Penalty: Away", "Penalty: Death"
        };
        float* param_values[] = {nullptr, nullptr, &params.learning_rate, &params.gamma, &params.epsilon_decay, nullptr, &params.reward_food,
            &params.reward_closer, &params.penalty_away, &params.penalty_death};
        int* param_ints[] = {&game_speed, &train_speed, nullptr, nullptr, nullptr, &params.batch_size, nullptr, nullptr, nullptr, nullptr};

        for(int i = 0; i < 10; i++){
            SDL_Color col = (i == selected_param) ? SDL_Color{100, 255, 100, 255} : SDL_Color{150, 150, 150, 255};
            std::string prefix = (i == selected_param) ? "> " : "  ";
            oss.str(""); oss.clear();
            oss << prefix << param_names[i] << ": ";
            if(param_values[i]) oss << std::fixed << std::setprecision(4) << *param_values[i];
            else if(param_ints[i]) oss << *param_ints[i];
            draw_text(oss.str(), ctrl_x, ctrl_y + i * 12, col, 1);
        }

        // Helper to draw graph
        auto draw_graph = [&](const std::vector<float>& data, int gy, SDL_Color line_color, float max_val, const std::string& label)
        {
            float gx = (float)(x_off + margin);
            float gw = (float)(w - 2 * margin);
            float fgy = (float)gy;
            float fgh = (float)graph_h;

            // Label
            draw_text(label, (int)gx, (int)fgy, {180, 180, 180, 255}, 1);
            fgy += label_h;

            // Graph background
            SDL_SetRenderDrawColor(renderer, 40, 40, 55, 255);
            SDL_FRect gbg = {gx, fgy, gw, fgh};
            SDL_RenderFillRect(renderer, &gbg);

            // Border (Fix: SDL_RenderDrawRect -> SDL_RenderRect)
            SDL_SetRenderDrawColor(renderer, 70, 70, 90, 255);
            SDL_RenderRect(renderer, &gbg);

            // Grid lines (Fix: SDL_RenderDrawLine -> SDL_RenderLine)
            SDL_SetRenderDrawColor(renderer, 50, 50, 65, 255);
            for(int i = 1; i < 4; i++)
            {
                float ly = fgy + (fgh * i / 4.0f);
                SDL_RenderLine(renderer, gx, ly, gx + gw, ly);
            }

            // Data line
            if(data.size() > 1)
            {
                SDL_SetRenderDrawColor(renderer, line_color.r, line_color.g, line_color.b, 255);
                float x_step = gw / (data.size() - 1);
                for(size_t i = 1; i < data.size(); i++)
                {
                    float v1 = std::min(data[i-1] / max_val, 1.0f);
                    float v2 = std::min(data[i] / max_val, 1.0f);
                    float x1 = gx + ((float)(i-1) * x_step);
                    float x2 = gx + ((float)i * x_step);
                    float y1 = fgy + fgh - (v1 * (fgh - 4.0f)) - 2.0f;
                    float y2 = fgy + fgh - (v2 * (fgh - 4.0f)) - 2.0f;
                    SDL_RenderLine(renderer, x1, y1, x2, y2);
                }
            }

            // Current value indicator
            if(!data.empty())
            {
                float v = std::min(data.back() / max_val, 1.0f);
                float dot_y = fgy + fgh - (v * (fgh - 4.0f)) - 2.0f;
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                SDL_FRect dot = {gx + gw - 4.0f, dot_y - 2.0f, 5.0f, 5.0f};
                SDL_RenderFillRect(renderer, &dot);

                oss.str(""); oss.clear();
                oss << std::fixed << std::setprecision(1) << data.back();
                draw_text(oss.str(), (int)(gx + gw - 40), (int)(fgy + 5), {255, 255, 255, 255}, 1);
            }
        };

        // Render the three graphs
        float max_score_val = std::max(50.0f, (float)current_max_score * 1.2f);
        int g1_y = y_off + 75;
        int g2_y = g1_y + graph_h + label_h + 15;
        int g3_y = g2_y + graph_h + label_h + 15;

        draw_graph(score_history, g1_y, {100, 220, 100, 255}, max_score_val, "SCORE PER EPISODE (green)");
        draw_graph(avg_score_history, g2_y, {100, 150, 255, 255}, max_score_val, "AVERAGE SCORE /10 eps (blue)");
        draw_graph(epsilon_history, g3_y, {255, 200, 100, 255}, 1.0f, "EXPLORATION RATE (orange)");
    }


    void SnakeAI::render_network(int x_off, int y_off, int w, int h) {
        // 1. Background - Use SDL_FRect and cast to float to avoid narrowing
        SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
        SDL_FRect bg = { (float)x_off, (float)y_off, (float)w, (float)h };
        SDL_RenderFillRect(renderer, &bg);

        draw_text("Q-NETWORK WEIGHTS", x_off + 10, y_off + 10, {255, 255, 255, 255}, 1);

        const int padding_x = 30;
        const int padding_y = 35;
        const int layers[] = {neuralConstants::inputSize, neuralConstants::HIDDEN_SIZE, neuralConstants::HIDDEN_SIZE, neuralConstants::OUTPUT_SIZE};
        const int num_layers = 4;

        // 2. Use SDL_FPoint for sub-pixel precision in SDL3
        std::vector<std::vector<SDL_FPoint>> neuron_pos(num_layers);
        for(int l = 0; l < num_layers; l++){
            int n = layers[l];
            int max_display = std::min(n, 20);
            neuron_pos[l].resize(max_display);
            for(int i = 0; i < max_display; i++){
                neuron_pos[l][i] = {
                    (float)(x_off + padding_x + l * (w - 2*padding_x) / (num_layers-1)),
                    (float)(y_off + padding_y + i * (h - 2*padding_y) / std::max(1, max_display-1))
                };
            }
        }

        auto draw_layer_connections = [&](torch::nn::Linear layer, int l)
        {
            auto wt = layer->weight.detach();
            int in_n = std::min((int)wt.size(1), (int)neuron_pos[l].size());
            int out_n = std::min((int)wt.size(0), (int)neuron_pos[l+1].size());
            for(int i = 0; i < out_n; i++){
                for(int j = 0; j < in_n; j++){
                    float val = wt[i][j].item<float>();
                    uint8_t r = val > 0 ? (uint8_t)std::min(255, int(val * 400)) : 0;
                    uint8_t g = val < 0 ? (uint8_t)std::min(255, int(-val * 400)) : 0;

                    SDL_SetRenderDrawColor(renderer, r, g, 30, 100);
                    // 3. SDL_RenderDrawLine is now SDL_RenderLine
                    SDL_RenderLine(renderer,
                        neuron_pos[l][j].x, neuron_pos[l][j].y,
                        neuron_pos[l+1][i].x, neuron_pos[l+1][i].y);
                }
            }
        };

        draw_layer_connections(QNetwork.fc1, 0);
        draw_layer_connections(QNetwork.fc2, 1);
        draw_layer_connections(QNetwork.fc3, 2);

        // 4. Neurons - Use SDL_FRect
        for(int l = 0; l < num_layers; l++)
        {
            for(auto& p : neuron_pos[l])
            {
                SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
                SDL_FRect r = {p.x - 3.0f, p.y - 3.0f, 6.0f, 6.0f};
                SDL_RenderFillRect(renderer, &r);
            }
        }

        // Layer labels
        const char* layer_names[] = {"Input", "Hidden1", "Hidden2", "Output"};
        for(int l = 0; l < num_layers; l++)
        {
            // Cast to int for your draw_text function which still takes ints
            draw_text(layer_names[l], (int)(neuron_pos[l][0].x - 15), y_off + h - 20, {150, 150, 150, 255}, 1);
        }
    }
    void SnakeAI::render_network_dynamic(const std::vector<float>& state, int x_off, int y_off, int w, int h){
    // 1. Background (SDL_FRect)
    SDL_SetRenderDrawColor(renderer, 15, 15, 25, 255);
    SDL_FRect bg = { (float)x_off, (float)y_off, (float)w, (float)h };
    SDL_RenderFillRect(renderer, &bg);

    draw_text("LIVE NETWORK ACTIVITY", x_off + 10, y_off + 10, {255, 255, 255, 255}, 1);

    const int padding_x = 30;
    const int padding_y = 35;
    const int layers[] = {neuralConstants::inputSize, neuralConstants::HIDDEN_SIZE, neuralConstants::HIDDEN_SIZE, neuralConstants::OUTPUT_SIZE};
    const int num_layers = 4;

    // 2. Use SDL_FPoint for high-precision positions
    std::vector<std::vector<SDL_FPoint>> neuron_pos(num_layers);
    for(int l = 0; l < num_layers; l++){
        int n = layers[l];
        int max_display = std::min(n, 20);
        neuron_pos[l].resize(max_display);
        for(int i = 0; i < max_display; i++){
            neuron_pos[l][i] = {
                (float)(x_off + padding_x + l * (w - 2 * padding_x) / (num_layers - 1)),
                (float)(y_off + padding_y + i * (h - 2 * padding_y) / std::max(1, max_display - 1))
            };
        }
    }

    // Neural Network Logic (Remains mostly same)
    torch::Tensor inp = torch::tensor(state).reshape({1, neuralConstants::inputSize});
    torch::Tensor h1 = torch::relu(QNetwork.fc1->forward(inp));
    torch::Tensor h2 = torch::relu(QNetwork.fc2->forward(h1));
    torch::Tensor out = QNetwork.fc3->forward(h2);
    std::vector<torch::Tensor> activations = {inp.flatten(), h1.flatten(), h2.flatten(), out.flatten()};

    // 3. Draw connections with SDL_RenderLine
    auto draw_connections = [&](torch::nn::Linear layer, int l){
        auto wt = layer->weight.detach();
        int in_n = std::min((int)wt.size(1), (int)neuron_pos[l].size());
        int out_n = std::min((int)wt.size(0), (int)neuron_pos[l+1].size());
        for(int i = 0; i < out_n; i++){
            for(int j = 0; j < in_n; j++){
                float val = wt[i][j].item<float>() * activations[l][j].item<float>();
                float intensity = std::tanh(std::abs(val)) * 255.0f;
                SDL_SetRenderDrawColor(renderer, 0, 0, (uint8_t)std::min(255, (int)(intensity * 2)), 150);

                // SDL_RenderDrawLine -> SDL_RenderLine
                SDL_RenderLine(renderer,
                    neuron_pos[l][j].x, neuron_pos[l][j].y,
                    neuron_pos[l+1][i].x, neuron_pos[l+1][i].y);
            }
        }
    };

    draw_connections(QNetwork.fc1, 0);
    draw_connections(QNetwork.fc2, 1);
    draw_connections(QNetwork.fc3, 2);

    // 4. Neurons with SDL_FRect
    for(int l = 0; l < num_layers; l++){
        int max_display = std::min(layers[l], 20);
        for(int i = 0; i < max_display; i++){
            float act = activations[l][i].item<float>();
            uint8_t intensity = (uint8_t)std::min(255, std::max(0, int(std::abs(act) * 200)));
            SDL_SetRenderDrawColor(renderer, intensity, intensity, 255, 255);

            SDL_FRect r = { neuron_pos[l][i].x - 4.0f, neuron_pos[l][i].y - 4.0f, 8.0f, 8.0f };
            SDL_RenderFillRect(renderer, &r);
        }
    }

    // 5. Output labels
    const char* actions[] = {"UP", "DOWN", "LEFT", "RIGHT"};
    for(int i = 0; i < 4; i++){
        float q_val = out[0][i].item<float>();
        std::ostringstream oss;
        oss << actions[i] << ": " << std::fixed << std::setprecision(2) << q_val;

        float ly = neuron_pos[3][i].y - 3.0f;
        SDL_Color col = (i == out.argmax(1).item<int>()) ? SDL_Color{100, 255, 100, 255} : SDL_Color{150, 150, 150, 255};

        // Cast back to int for your custom text function
        draw_text(oss.str(), (int)(neuron_pos[3][i].x + 10), (int)ly, col, 1);
    }
}

    void SnakeAI::render_all(const std::vector<float>& state, int episode, int ep_score){
        if(!rendermode) return;

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Layout Logic (Assuming these have been updated to SDL_FRect internally)
        render_game(0, 0);
        render_stats(window::GAME_SIZE, 0, window::STAT_W, window::GAME_SIZE, episode, ep_score);
        render_network(0, window::GAME_SIZE, window::WINDOW_W/2, window::NET_VIS_H);
        render_network_dynamic(state, window::WINDOW_W/2, window::GAME_SIZE, window::WINDOW_W/2, window::NET_VIS_H);

        // Border lines
        SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255);

        // Fix: SDL_RenderDrawLine -> SDL_RenderLine
        // Note: Cast to float to satisfy the new parameter types
        SDL_RenderLine(renderer, (float)window::GAME_SIZE, 0.0f, (float)window::GAME_SIZE, (float)window::GAME_SIZE);
        SDL_RenderLine(renderer, 0.0f, (float)window::GAME_SIZE, (float)window::WINDOW_W, (float)window::GAME_SIZE);
        SDL_RenderLine(renderer, (float)(window::WINDOW_W/2), (float)window::GAME_SIZE, (float)(window::WINDOW_W/2), (float)window::WINDOW_H);

        SDL_RenderPresent(renderer);
    }
void SnakeAI::train(int episodes){
int total_score = 0;
    int max_score = 0;

    for(int ep = 0; ep < episodes; ep++){
        episodeCount = ep;
        reset();
        auto state = get_state();
        int ep_score = 0;
        int steps = 0;

        while(!game_over && steps < 1000){
            int a = select_action(state);
            float r = step(a, false);
            auto ns = get_state();

            replay_buffer.push_back({state, a, r, ns, game_over});
            if((int)replay_buffer.size() > params.replay_buffer_size) replay_buffer.pop_front();

            if((int)replay_buffer.size() >= params.batch_size)
                train_step();

            state = ns;
            ep_score = score;
            steps++;
            totalStep++;

            // Handle events
            SDL_Event e;
            while(SDL_PollEvent(&e)){
                if(e.type == SDL_EVENT_QUIT) return; // SDL2: SDL_QUIT -> SDL3: SDL_EVENT_QUIT

                if(e.type == SDL_EVENT_KEY_DOWN){ // SDL2: SDL_KEYDOWN -> SDL3: SDL_EVENT_KEY_DOWN
                    // SDL3 Fix: e.key.keysym.sym is now just e.key.key
                    switch(e.key.key){
                        case SDLK_UP:
                            selected_param = (selected_param + 9) % 10;
                            break;
                        case SDLK_DOWN:
                            selected_param = (selected_param + 1) % 10;
                            break;
                        case SDLK_LEFT:
                        case SDLK_RIGHT: {
                            float mult = (e.key.key == SDLK_RIGHT) ? 1.1f : 0.9f;
                            int delta = (e.key.key == SDLK_RIGHT) ? 1 : -1;
                            switch(selected_param){
                                case 0: game_speed = std::clamp(game_speed + delta * 5, 5, 120); break;
                                case 1: train_speed = std::clamp(train_speed + delta, 1, 100); break;
                                case 2: params.learning_rate = std::clamp(params.learning_rate * mult, 0.00001f, 0.1f); break;
                                case 3: params.gamma = std::clamp(params.gamma + delta * 0.01f, 0.5f, 0.999f); break;
                                case 4: params.epsilon_decay = std::clamp(params.epsilon_decay + delta * 0.001f, 0.9f, 0.9999f); break;
                                case 5: params.batch_size = std::clamp(params.batch_size + delta * 16, 16, 512); break;
                                case 6: params.reward_food = std::clamp(params.reward_food + delta * 1.0f, 1.0f, 100.0f); break;
                                case 7: params.reward_closer = std::clamp(params.reward_closer + delta * 0.05f, 0.0f, 2.0f); break;
                                case 8: params.penalty_away = std::clamp(params.penalty_away + delta * 0.05f, -2.0f, 0.0f); break;
                                case 9: params.penalty_death = std::clamp(params.penalty_death + delta * 1.0f, -100.0f, -1.0f); break;
                            }
                            break;
                        }
                        case SDLK_R: // SDL3: Note that SDLK names are generally uppercase now
                            train_speed = 1;
                            game_speed = 15;
                            params = TrainingParams();
                            break;
                        case SDLK_SPACE:
                            epsilon = 1.0f;
                            break;
                    }
                }
            }

            // Render based on train_speed
            bool should_render = (ep % train_speed == 0) || RenderCount::force_To_Render;
            if(should_render && rendermode){
                // SDL3 Fix: SDL_GetTicks() returns Uint64
                Uint64 frame_start = SDL_GetTicks();
                render_all(state, ep, ep_score);
                Uint64 frame_time = SDL_GetTicks() - frame_start;

                Uint64 frame_delay = 1000 / game_speed;
                if(frame_time < frame_delay){
                    SDL_Delay((Uint32)(frame_delay - frame_time));
                }
            }
        }

        // Quick transition - no delay between episodes
        total_score += ep_score;
        max_score = std::max(max_score, ep_score);
        current_max_score = max_score;
        epsilon = std::max(params.epsilon_end, epsilon * params.epsilon_decay);

        // Track score history
        score_history.push_back((float)ep_score);
        if(score_history.size() > 200) score_history.erase(score_history.begin());

        if(ep % 10 == 0){
            float avg = total_score / 10.0f;
            current_avg = avg;
            avg_score_history.push_back(avg);
            epsilon_history.push_back(epsilon);
            if(avg_score_history.size() > 200) avg_score_history.erase(avg_score_history.begin());
            if(epsilon_history.size() > 200) epsilon_history.erase(epsilon_history.begin());

            std::cout << "Episode " << ep
                      << " | Avg: " << avg
                      << " | Max: " << max_score
                      << " | Eps: " << std::fixed << std::setprecision(3) << epsilon
                      << " | Speed: " << train_speed << "x\n";
            total_score = 0;
        }
    }
}


}
