if (UNIX)
cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
project(A_STAR_SEARCH LANGUAGES CXX)
endif(UNIX)

if (WIN32) #Temporary
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(A_STAR_SEARCH LANGUAGES CXX CUDA)
endif(WIN32)

set (CMAKE_CXX_STANDARD 11)
include_directories(src libs/Catch2 libs/logger)

set(TEST TRUE)

if (TEST)
add_executable(main test/test.cpp)
else()
add_executable(main src/main.cpp)
endif (TEST)
target_link_libraries(main graph a_star ${CMAKE_THREAD_LIBS_INIT})
