cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MATRIX_MULTIPLICATION LANGUAGES C CXX CUDA)
set (CMAKE_CXX_STANDARD 11)

set(SRC "./part 1 and 2")
set(SRC_2 "./part 3 and 4")

link_directories(${SRC})
include_directories(${SRC} libs/catch2 libs/logger)
include_directories($ENV{CUDA_PATH}/include)

add_library(cu_matrix STATIC ${SRC}/matrix.cu)
set_target_properties( cu_matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cu_matrix PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(part_1_2 ${SRC}/main.cpp)
add_executable(catch_test test/test.cpp)

add_executable(part_3_4 ${SRC_2}/mm_main.cu ${SRC_2}/mmm_unrolled.cuh)
set_property(TARGET part_3_4 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


set(LIBS cu_matrix cublas)
target_link_libraries(catch_test cu_matrix cublas)
target_link_libraries(part_1_2 cu_matrix cublas)

add_test(test catch_test)
