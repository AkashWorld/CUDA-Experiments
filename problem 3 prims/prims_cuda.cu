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

/* created by Jonathan Garner */

using namespace std;




__global__
void cudaFindNeighbors(int v, int idx, int ** dmat, int * mins){
	//printf("called\n");
	int index = idx * v;
	int spot = index + threadIdx.x;
	if(threadIdx.x < v){
		if (dmat[index + threadIdx.x]){
			printf("Vertex: %d Parent: %d\n",index+threadIdx.x,mins[spot]);
		}
	}
	
	
	
	return;
}

int main(int arg, char** argv) {
	int ** mat;
	int v;
	
	if (arg != 3){
		printf("usage: ./out size, starting index example: ./t 5 2 \n");
		return -1;
	}
	v = atoi(argv[1]);
	int start = atoi(argv[2]);
	
	if (start >= v || start<0) {
		printf("start index is out of bounds \n");
		return -1;

	}
	FILE *pToFile = fopen("graph.txt", "r");

	int single;
	int i = 0;
	int j = 0;
	
	mat = (int**)malloc(v * sizeof(int*));
	for (i = 0; i < v; i++) {
		mat[i] = (int*)malloc(v * sizeof(int));
	}
	
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
			mat[i][j] = 0;
		}
	}
	
	i=0;
	j=0;
	while ((single = fgetc(pToFile)) != EOF) {
		// single = char being read
		if (j == v) {
			j = 0;
			i++;
			continue;
		}	// increment i if end of line is reached
		if (single != '0') {
			//cout << i << " " << j << endl;
			//cout << endl<< "hi" << i<< j<< endl;
			mat[i][j] = single-48;
		}
		j++;
	}
	
	fclose(pToFile);
	
	// debug: display mat
	/*
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
		cout << mat[i][j];
		}
		cout << endl;
	}
	*/
	
	
	
	int ** dmat;
	cudaMalloc((void**)&dmat, sizeof(int*)*v);
	cudaMemcpy((void*)dmat, (void*)mat, sizeof(int*)*v, cudaMemcpyHostToDevice);
	
	cudaEvent_t st1, stop;
	cudaEventCreate(&st1);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(st1);
	
	/*
	int * Q = (int*)malloc(v * sizeof(int));
	key = (int*)malloc(v * sizeof(int));
	parent = (int*)malloc(v * sizeof(int));
	int * dQ, * dkey, * dparent;
	*/

	//cudaMalloc((void**)&Q, sizeof(int) * v);

	/*
	cudaMalloc((void**)&dQ, sizeof(int) * v);
	cudaMemcpy((void*)dQ, (void*)Q, sizeof(int)*v, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&dkey, sizeof(int) * v);
	cudaMemcpy((void*)dkey, (void*)key, sizeof(int)*v, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&dparent, sizeof(int) * v);
	cudaMemcpy((void*)dparent, (void*)parent, sizeof(int)*v,cudaMemcpyHostToDevice);
	*/
	/*
	for(i = 0; i < v; i++){
		Q[i] = i;
	}
	*/

	// find nearest neighbor for each vertex
	int k;
	int * mins;
	mins = (int*)malloc(v * sizeof(int));

	//cudaMalloc((void**)&mins, sizeof(int) * v);
	// mins array stores the location of the nearest neighbor for each vertex
	for(i = 0; i < v; i ++){
		k = INT_MAX;
		for(j = 0; j < v; j++){
			if(mat[i][j] <= k && mat[i][j] != 0){
				k = mat[i][j];
				mins[i] = j;
			}
		}
		if(k == INT_MAX){
			mins[i] = 0;	// no neighbors found
		}
	}
	
	// debug: display mins
	/*
	printf("mins array: \n");
	for (i = 0; i < v; i++) {
		printf("%d ",mins[i]);
	}
	cout << endl;
	*/
	
	
	int * dmins;
	cudaMalloc((void**)&dmins, sizeof(int) * v);
	cudaMemcpy((void*)dmins, (void*)mins, sizeof(int)*v, cudaMemcpyHostToDevice);
	
	
	i=0;

	cudaFindNeighbors<<<1, v>>>(v, i, dmat, dmins);
	
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, st1, stop);
	cout << endl << milliseconds << " ms" <<endl;
	
	cudaFree(dmat);
	cudaFree(dmins);
	//cudaFree(Q);
	
	
	
	return 0;
}















