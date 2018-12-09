
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <time.h>
#include <cstring>

#include <cstdlib>
#include <iostream>
#include <time.h>


using namespace std;




__global__
void cuda_bfs(int v,int idx, int * dmat, bool * d_visited, int * d_push) {
	
	int index = idx * v;

	if (threadIdx.x < v) {
	
		if (dmat[index+ threadIdx.x ] && !d_visited[threadIdx.x]) {
			//printf("%d-----%d\n", index + threadIdx.x, threadIdx.x);
		//	printf("%d\n", threadIdx.x);
			d_visited[threadIdx.x] = true;
			d_push[threadIdx.x] = 1;
		}
	}


	/*if (threadIdx.x < v) {
		if (visited[threadIdx.x] == true) {
			printf("%d\n", threadIdx.x);

		}

	}
	/*if (threadIdx.x < v*v) {
		if (dmat[threadIdx.x] == 1) {

			printf("%d\n", threadIdx.x);
		}
	}
	*/
}

int main(int arg, char** argv) {
	int* mat;
	int i;
	int v;

	char single;

	if (arg != 3)
	{
		printf("usage: ./out size, starting_index example: ./t 5 2 \n");

		return -1;
	}
	v = atoi(argv[1]);
	int start = atoi(argv[2]);

	if (start >= v || start<0) {
		printf("start index is out of bounds \n");
		return -1;

	}
	FILE *pToFile = fopen("graph.txt", "r");

	 
	i = 0;



	//


	 mat = (int*)malloc(v *v* sizeof(int));

	 while ((single = fgetc(pToFile)) != EOF) {


		 if (single != '\n') {

			 if (single == '1') {
				// cout << i << endl;
				 mat[i] = 1;
				 //cout << mat[i] << endl;
			 }
			 else {
				 mat[i] = 0;
			 }
			 i++;
		 }

	 }


	 fclose(pToFile);


	//return 0;
	//cout << h_f[1] << h_f[1] << h_f[0];
	

	int * dmat;


	cudaMalloc((void**)&dmat, sizeof(int) * v*v);
	cudaMemcpy((void*)dmat, (void*)mat, sizeof(int)*v*v, cudaMemcpyHostToDevice);



	bool* visited = (bool*)malloc(v * sizeof(bool));   //visited
	bool* d_visited;

	for (int i = 0; i < v; i++) {
		visited[i] = false;
	}
	visited[start] = true;


	cudaMalloc((void**)&d_visited, sizeof(bool) * v);
	cudaMemcpy((void*)d_visited, (void*)visited, sizeof(bool)*v, cudaMemcpyHostToDevice);




	queue<int> q;						//queue
	q.push(start);

	int* h_push = (int*)malloc(v * sizeof(int));			//h push
	int* d_push;
	for (i = 0; i < v; i++) {
		h_push[i] = 0;
	}

	//h_push[start] = 1;
	
	cudaMalloc((void**)&d_push, sizeof(int) * v);
	cudaMemcpy((void*)d_push, (void*)h_push, sizeof(int)*v, cudaMemcpyHostToDevice);


	int j = 0;;
	while (!q.empty()) {
		for (i = 0; i < v; i++) {
			h_push[i] = 0;
		}

		cudaMemcpy((void*)d_push, (void*)h_push, sizeof(int)*v, cudaMemcpyHostToDevice);
	//	cudaMemcpy((void*)d_visited, (void*)visited, sizeof(bool)*v, cudaMemcpyHostToDevice);
		i = q.front();
		cout << q.front() << " ";
		q.pop();

		cuda_bfs << <1, v >> >(v, i, dmat, d_visited, d_push);

		cudaMemcpy((void*)h_push, (void*)d_push, sizeof(int) * v, cudaMemcpyDeviceToHost);
	//	cudaMemcpy((void*)d_visited, (void*)visited, sizeof(bool) * v, cudaMemcpyDeviceToHost);
	//	cout << h_push[1];
		for (j = 0; j < v; j++) {
			if (h_push[j] == 1) {
				//cout << j;

				q.push(j);
			}

		}

	}

	//cuda_bfs << <1, v >> >(v,i, dmat,d_visited, d_push);
	return 0;
}