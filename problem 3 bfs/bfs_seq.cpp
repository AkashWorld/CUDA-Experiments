#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <queue>
#include <time.h>

using namespace std;

int** mat;
int v;
int e;

int main() {

	FILE *pToFile = fopen("graph.txt", "r");

	int line = 0;
	char single;
	int i = 0;
	while (i < 39) {
		single = fgetc(pToFile);
		
		i++;
	}
	
 v = single - '0';

	single = fgetc(pToFile);
	single = fgetc(pToFile);

	 e = single - '0';
	
	single = fgetc(pToFile);

	int j = 0;
	i = 0;
	
	 mat = (int**)malloc(v * sizeof(int*));
	for (i = 0; i < v; i++ ) {
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
		


		if (j == v) {
			j = 0;
			i++;
			continue;
		}
		
		if (single == '1') {
			mat[i][j] = 1;
		}
		j++;
		
	}


	fclose(pToFile);

	queue<int> q;

/*	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
			cout << mat[i][j];
		}
		cout << endl;
	}
	*/
	return 0;

}