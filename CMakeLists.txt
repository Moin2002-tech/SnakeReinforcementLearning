cmake_minimum_required(VERSION 3.22)

# 2. MUST define the project before calling find_package
project(ReinforcementSnake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
   add_compile_options(/utf-8)
 endif()

 if (MSVC)
  # Force the MSVC runtime library to MultiThreadedDLL (/MD) for all configs.
  # Change to "MultiThreadedDebugDLL" to force /MDd instead.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
endif()

# Force /MD in Debug (example) to match prebuilt libs


set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")

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



        
        
        "${CMAKE_SOURCE_DIR}/libtorch/lib/torch.lib"
        "${CMAKE_SOURCE_DIR}/libtorch/lib/torch_cpu.lib"
        "${CMAKE_SOURCE_DIR}/libtorch/lib/c10.lib"
        "${CMAKE_SOURCE_DIR}/libtorch/lib/fmt.lib"

      
   



  

)


set_target_properties(ReinforcementSnake PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)