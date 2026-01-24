//
// Created by moinshaikh on 1/17/26.
//

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



struct QNetImpl : public torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    QNetImpl();
    torch::Tensor forward(torch::Tensor input);
};
TORCH_MODULE(QNet);

namespace RenSnake
{

    class SnakeAI
    {
    public:
        SnakeAI(bool Render);
        ~SnakeAI();
        bool init();
        void train(int epochs);
    private:
        void reset();
        int serializeDirection(const int signedDir) const;
        int upackDirection(const int unsighnedDir) const;
        float step(int action, bool shouldprint);
        void render_all(const std::vector<float> & state,int epochs,int score);
        void render_game(int x_off, int y_off);
        void render_network(int x_off, int y_off, int w, int h);
        void render_network_dynamic(const std::vector<float>& state, int x_off, int y_off, int w, int h);
        void render_stats(int x_off, int y_off, int w, int h, int episode, int ep_score);
        void draw_text(const std::string& text, int x, int y, SDL_Color color, int scale = 1);
        void spawn_food();
        bool inside_grid(const Point &p) const;
        bool snake_contains(const Point &p) const;
        std::vector<float> get_state() const;
        int select_action(const std::vector<float>& state);
        void train_step();

        SDL_Window* window = nullptr;
        SDL_Renderer* renderer = nullptr;

        std::deque<Point> snake;
        Dir dir = Dir::RIGHT;
        Point food{0,0};
        bool game_over = false;
        int score = 0;
        int stepWithotfood = 0;
        std::mt19937 randomEngine;
        QNetImpl QNetwork;
        QNetImpl targetNetwork;

        torch::optim::Adam optimizer;


        std::deque<Experience> replay_buffer;
        float epsilon = initialConstants::EPSILON_START;
        bool rendermode;

        int episodeCount = 0;
        int totalStep= 0;

        // Stats tracking
        std::vector<float> score_history;
        std::vector<float> avg_score_history;
        std::vector<float> epsilon_history;
        int current_max_score = 0;
        float current_avg = 0;

        // Speed control
        int game_speed = 15;  // FPS (adjustable with +/-)
        int train_speed = 10;  // 1 = normal, higher = faster (skip renders)

        // Adjustable training parameters
        TrainingParams params;
        int selected_param = 0;  // Which parameter is selected for adjustment
    };



}



#endif //REINFORCEMENTSNAKE_SNAKEAI_HPP