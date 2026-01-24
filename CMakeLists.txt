cmake_minimum_required(VERSION 3.22)

# 2. MUST define the project before calling find_package
project(ReinforcementSnake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)

find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# 2. Define the Executable
project(ReinforcementSnake)


add_executable(ReinforcementSnake main.cpp
        src/Utils.h
        src/SnakeAI.hpp
        src/SnakeAI.cpp)


find_package(SDL3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)




find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)


target_include_directories(ReinforcementSnake PRIVATE
        "${CMAKE_SOURCE_DIR}/libtorch/include"
        "${CMAKE_SOURCE_DIR}/libtorch/include/torch/csrc/api/include"
)



target_link_libraries(ReinforcementSnake PRIVATE


        -Wl,--no-as-needed
        -Wl,--whole-archive
        "${CMAKE_SOURCE_DIR}/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_SOURCE_DIR}/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_SOURCE_DIR}/libtorch/lib/libtorch.so"
        -Wl,--no-whole-archive

        "${CMAKE_SOURCE_DIR}/libtorch/lib/libc10_cuda.so"
        "${CMAKE_SOURCE_DIR}/libtorch/lib/libc10.so"

        # 1. CUDA Runtime (from your toolkit)
        "/usr/local/cuda/lib64/libcudart.so"

        # 2. CUDA Driver (The one usually missing - found in system libs)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

        # 3. NVRTC (Compiler library)
        "/usr/local/cuda/lib64/libnvrtc.so"

        ${OpenCV_LIBS}
        # System base libs
        pthread
        dl
        m
)


set_target_properties(ReinforcementSnake PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)