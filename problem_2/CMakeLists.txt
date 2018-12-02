cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MATRIX_MULTIPLICATION LANGUAGES C CXX CUDA)

set (CMAKE_CXX_STANDARD 11)
include_directories(src libs/catch2 libs/logger)
include_directories($ENV{CUDA_PATH}/include)

add_library(cu_matrix src/matrix.cu)
set_property(TARGET cu_matrix PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(main src/main.cpp)
add_executable(catch_test test/test.cpp)

set(LIBS cu_matrix cublas)

target_link_libraries(catch_test ${LIBS})
target_link_libraries(main ${LIBS})

add_test(test catch_test)
