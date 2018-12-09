cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MATRIX_MULTIPLICATION LANGUAGES C CXX CUDA)
set (CMAKE_CXX_STANDARD 11)
link_directories(src)
include_directories(src libs/catch2 libs/logger)
include_directories($ENV{CUDA_PATH}/include)

add_library(cu_matrix STATIC src/matrix.cu)
set_target_properties( cu_matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cu_matrix PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(main src/main.cpp)
add_executable(catch_test test/test.cpp)

set(LIBS cu_matrix cublas)
target_link_libraries(catch_test cu_matrix cublas)
target_link_libraries(main cu_matrix cublas)

add_test(test catch_test)
