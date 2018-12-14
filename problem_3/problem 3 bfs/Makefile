all: bfs_seq create_graph cuda_bfs

cuda_bfs: bfs_cuda.cu
	nvcc bfs_cuda.cu -o bfs_cuda

create_graph: create_graph.cpp
	$(CXX)  -std=c++11 create_graph.cpp -o create_graph 
bfs_seq: bfs_seq.cpp
	$(CXX)  -std=c++11 bfs_seq.cpp -o bfs_seq 
clean:
	rm create_graph bfs_seq bfs_cuda
