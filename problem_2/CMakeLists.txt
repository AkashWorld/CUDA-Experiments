#If the NCUDA option is passed to CMAKE via -DNCUDA=TRUE,
#then the build files are generated without compiling CUDA code
#This is due to development limitations on my (Khalid) machine
enable_testing()

if (NCUDA)
cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
project(MATRIX_MULTIPLICATION LANGUAGES CXX)
else (NCUDA)
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MATRIX_MULTIPLICATION LANGUAGES CXX CUDA)
include_directories($ENV{CUDA_PATH}/include)
endif(NCUDA)

set (CMAKE_CXX_STANDARD 11)
include_directories(src libs/catch2 libs/logger)
add_executable(main src/main.cpp)
add_executable(catch_test test/test.cpp)

if (NCUDA)
target_link_libraries(catch_test)
target_link_libraries(main)
else (NCUDA)
add_library(matrix src/matrix.cu)
target_link_libraries(catch_test matrix)
target_link_libraries(main matrix)
endif(NCUDA)

add_test(test catch_test)