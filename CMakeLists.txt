cmake_minimum_required(VERSION 3.12)

project(ekf_gpu VERSION 2.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES src/ekf.cpp src/demo.cpp)

find_package(ArrayFire)

add_executable(ekf ${SOURCES})

target_include_directories(ekf PRIVATE inc)
target_link_libraries(ekf ArrayFire::afopencl) #afcuda, afopencl, afcpu


