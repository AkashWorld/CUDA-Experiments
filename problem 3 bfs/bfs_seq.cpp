#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <queue>
#include <time.h>
#include <cstring>

using namespace std;


void loop(bool* visited, queue<int> q);

int** mat;
int v;
int e;

void loop(bool* visited, queue<int> q) {
	int i;
	while (!q.empty()) {
		int i = q.front();
		cout << q.front() << " ";
		q.pop();

		
		for (int j = 0; j<v; j++) {
			if (mat[i][j] && !visited[j]) {
				//cout << endl << i << j << endl;
				visited[j] = true;
				q.push(j);
			}
		}
	}

}

int main(int arg, char** argv) {
	
	int flag = 0;
	if (arg != 4)
	{
		cout << "usage: ./out size, starting_index, flag -l or -r. example: ./t 5 2 -r " << endl;
		return -1;
	}
	v = atoi(argv[1]);
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

	int line = 0;
	char single;
	int i = 0;



	int j = 0;
	i = 0;

	mat = (int**)malloc(v * sizeof(int*));
	for (i = 0; i < v; i++) {
		mat[i] = (int*)malloc(v * sizeof(int));
	}

	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
			mat[i][j] = 0;
		}

	}

	i = 0;
	j = 0;
	//cout << v << e << endl;
	while ((single = fgetc(pToFile)) != EOF) {

		//cout << single;

		if (j == v) {
			j = 0;
			i++;
			continue;
		}

		if (single == '1') {
		//	cout << i << " " << j << endl;
		//	cout << endl<< "hi" << i<< j<< endl;
			mat[i][j] = 1;
		}
		j++;

	}


	fclose(pToFile);

/*
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
		cout << mat[i][j];
		}
		cout << endl;
	}
	*/


	bool* visited = (bool*)malloc(v * sizeof(bool));

	for (int i = 0; i < v; i++) {
		visited[i] = false;
	}
	visited[start] = true;

	queue<int> q;
	q.push(start);

	cout << "visiting " ;

	if (flag == 1) {
		loop(visited, q);
	}
	
	if (flag == 2) {

	}
	return 0;
	
}