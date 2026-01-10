cmake_minimum_required(VERSION 3.16)
project(vibe_check LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# Build Python extension module: vibe_core
pybind11_add_module(vibe_core vibeCheckCore.cpp)

target_include_directories(vibe_core PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vibe_core PRIVATE ${OpenCV_LIBS})
