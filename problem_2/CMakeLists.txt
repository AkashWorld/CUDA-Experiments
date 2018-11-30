enable_testing()

if (UNIX)
cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
project(MATRIX_MULTIPLICATION LANGUAGES CXX CUDA)
endif(UNIX)

if (WIN32) #Temporary
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MATRIX_MULTIPLICATION CXX CUDA)
endif(WIN32)

set (CMAKE_CXX_STANDARD 11)
include_directories(src libs/catch2 libs/logger)

add_executable(main src/main.cpp)
add_executable(catch_test test/test.cpp)
target_link_libraries(catch_test)
target_link_libraries(main)

add_test(test catch_test)