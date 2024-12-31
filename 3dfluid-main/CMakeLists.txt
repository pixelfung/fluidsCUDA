cmake_minimum_required(VERSION 3.5)
project(fluid)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fluid_sim main.cpp EventManager.cpp fluid_solver.cpp)