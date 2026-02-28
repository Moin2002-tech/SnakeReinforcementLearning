
#include <iostream>
#include"src/SnakeAI.hpp"
#include<torch/torch.h>

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main(int argc, char** argv) {

   std::signal(SIGINT,RenderCount::signal_handler);
    bool render = true;
    RenSnake::SnakeAI g(render);

    if(!g.init()) return 1;
    std::cout << "Snake AI Training - Controls:\n";
    std::cout << "  Up/Down   : Select parameter\n";
    std::cout << "  Left/Right: Adjust selected parameter\n";
    std::cout << "  R         : Reset all to defaults\n";
    std::cout << "  Space     : Reset exploration (epsilon=1)\n\n";
    std::cout << "Starting training...\n";

    g.train(100000);
}