all: prims_seq create_graph_weighted

create_graph_weighted: create_graph_weighted.cpp
	$(CXX)  -std=c++11 create_graph_weighted.cpp -o create_graph_weighted 
prims_seq: prims_seq.cpp
	$(CXX)  -std=c++11 prims_seq.cpp -o prims_seq 
clean:
	rm create_graph_weighted prims_seq