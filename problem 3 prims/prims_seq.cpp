#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <queue>
#include <time.h>
#include <cstring>
#include <chrono>
#include <climits>

/* Created by Jonathan Garner */

using namespace std;
using namespace std::chrono;
//void loop(bool* visited, queue<int> q);
//void recursive(bool* visited, queue<int> q);
int v, e, u, current, i;
int * key;
int * parent;
int** mat;
queue<string> edges;
queue<int> adjacent;
queue<int> weight;


// used to find the index of the smallest value in key array
int minIndex(int * key, int * Q){
    int index = 1;
    for(int i = 1; i < v; i++){
		if(Q[i] != -1){		
		// vertex must still be in Q to be considered
			if(key[i] <= key[index])
				index = i;      
		}
    }
	if(Q[index] == -1)return -2;
    return index;
}

// used to check if there are still verticies in Q
bool QValid(int * Q){
	int i;
	for(i = 0; i < v; i++){
		if(Q[i] != -1)return true;
	}
	return false;
}

void loop(int start, int * Q){
	int i = 0;
	key[start] = 0;
	int iterations = 0;
	while(QValid(Q)){
		if(iterations == 0){
			u = 0;
		}else{
			u = minIndex(key, Q);
		}
		iterations++;
		if(u == -2)break;
		
		/*
		cout << "min index found: ";
		cout << u;
		cout << endl;
		*/
		
		Q[u] = -1;			// remove vertex with min key from Q
			
		/*	
		cout << "Removed vertex ";
		cout << u;
		cout << " from ID list Q. Current Q:";
		cout << endl;		
		for (i = 0; i < v; i++) {
			cout << Q[i];
		}
		cout << endl;
		*/
		
		
		if(parent[u] != -1){
			//cout << "Adding an edge to final array!";
			//cout << endl;
			edges.push("Vertex: " + to_string(u) + ", Parent: " + to_string(parent[u]));
		}
		
		// find all verticies adjacent to the vertex at index u (every nonzero entry in mat[u])
		for(i = 0; i < v; i++){
			if(mat[u][i] != 0){
				/*
				cout << "Found adjacent vertex at index ";
				cout << i;
				cout << " with weight  ";
				cout << mat[u][i];
				cout << endl; 
				*/
				adjacent.push(i);		// ID of adjacent vertex is queued
				weight.push(mat[u][i]);	// weight of edge to this vertex is queued
			}
		}
		while(!adjacent.empty()){
			current = adjacent.front();
			if(Q[current] != -1 && weight.front() < key[current]){
				parent[current] = u;
				
				/*
				cout << "Set parent of ";
				cout << current;
				cout << " to  ";
				cout << u;
				cout << endl;
				*/
				
				key[current] = weight.front();
				
				/*
				cout << "Set key of ";
				cout << current;
				cout << " to  ";
				cout << weight.front();
				cout << endl;
				*/
			}
			adjacent.pop();
			weight.pop();
		}		
		
		/*
		cout << "Current Keys:";
		cout << endl;
		for (i = 0; i < v; i++) {
			cout << key[i];
			cout << endl;
		}
		cout << endl;
		*/
	}
	
	cout << "All verticies have been found: \n";
	while(!edges.empty()) {
		cout << edges.front() << "\n";
		edges.pop();
	}
	cout << endl;
	
	return;
}

void recursive(int start, int * Q){
	if(!QValid(Q))return;
	cout << "Q found to be valid";
	key[start] = 0;
	
	u = minIndex(key, Q);
	if(u == -2)return;
	
	Q[u] = -1;

	loop(u, Q);

	if(parent[u] != -1){
			//cout << "Adding an edge to final array!";
			//cout << endl;
			edges.push("Vertex: " + to_string(u) + ", Parent: " + to_string(parent[u]));
	}
	for(i = 0; i < v; i++){
			if(mat[u][i] != 0){
				/*
				cout << "Found adjacent vertex at index ";
				cout << i;
				cout << " with weight  ";
				cout << mat[u][i];
				cout << endl; 
				*/
				adjacent.push(i);		// ID of adjacent vertex is queued
				weight.push(mat[u][i]);	// weight of edge to this vertex is queued
			}
		}
		while(!adjacent.empty()){
			current = adjacent.front();
			if(Q[current] != -1 && weight.front() < key[current]){
				parent[current] = u;
				
				/*
				cout << "Set parent of ";
				cout << current;
				cout << " to  ";
				cout << u;
				cout << endl;
				*/
				
				key[current] = weight.front();
			}
			adjacent.pop();
			weight.pop();
		}		
	return;
}

int main(int arg, char** argv){
	int flag = 0;
	if (arg != 4)
	{
		cout << "usage: ./out size, starting_index, flag -l or -r. example: ./t 5 2 -r " << endl;
		return -1;
	}
	v = atoi(argv[1]);	// size
	int start = atoi(argv[2]);

	if (!strcmp(argv[3], "-l")) {
		flag = 1;
	}
	else if (!strcmp(argv[3], "-r")) {
		flag = 2;
	}
	else {
		cout << "Incorrect flag: -l for loop, -r for recursive" << endl;
		return -1;
	}

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
	
	i = 0, j=0;
	//cout << v << e << endl;
	
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
	// graph has now been copied into mat with 0s appended if user inputted size > graph size
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
	
	clock_t st, end;
	auto st1 = high_resolution_clock::now();
	
	int * Q = (int*)malloc(v * sizeof(int));
	key = (int*)malloc(v * sizeof(int));
	parent = (int*)malloc(v * sizeof(int));
	
	// Q stores an ID for each vertex
	// each entry of key is initiatlized to max int
	// each parent is initialized to null (-1)
	for(i = 0; i < v; i++){
		Q[i] = i;
		key[i] = INT_MAX;
		parent[i] = -1;
	}
	
	if(flag == 1){
		loop(start, Q);
	}else if(flag == 2){
		recursive(start, Q);
	}
	
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - st1);
	cout << endl << duration.count() << " microseconds"  << endl;
	return 0;
	

	
	

	
}

