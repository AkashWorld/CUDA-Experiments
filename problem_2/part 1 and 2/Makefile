NVCC = ${CUDA_PATH}/bin/nvcc
SRC = ./
CULFLAGS = -lcudart -lcublas
CFLAGS = -std=c++11 -O3
ARCH = -arch=sm_35
CUDA_INC_PATH = ${CUDA_PATH}/include
LIB_PATH = ../libs
CATCH_PATH = $(LIB_PATH)/catch2
LOGGER_PATH = $(LIB_PATH)/logger
CU_LDIR = ${CUDA_PATH}/lib64
CPP_FILES = main.cpp matrix.h matrix.cuh

all: main

main: matrix_cu $(CPP_FILES)
	$(CXX) $(CFLAGS) -I$(LOGGER_PATH) main.cpp -o main.run
matrix_cu: matrix.cu
	$(NVCC) $(CFLAGS) $(CULFLAGS) $(ARCH) -I$(LOGGER_PATH) -I$(CUDA_INC_PATH) matrix.cu -c -o matrix.o -L$(CU_LDIR)
test: ../test/test.cpp $(CPP_FILES) matrix_cu
	$(CXX) $(CFLAGS) -I$(LOGGER_ATH) -I$(SRC) ../test/test.cpp -o catch_test
clean:
	rm *.o *.run
