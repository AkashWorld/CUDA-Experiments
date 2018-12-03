CXX = nvcc
CXX_FLAGS = -g -O3
CXX_OPTS = -Wall

all: brute_force

brute_force: brute_forcer.cu
	$(CXX) $(CXX_FLAGS) --compiler-options $(CXX_OPTS) brute_forcer.cu -o brute_forcer $(CXX_LIBS)

clean:
	rm brute_forcer