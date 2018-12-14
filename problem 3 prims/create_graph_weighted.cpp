#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>


/* Created by Suva Shahriaa, modified by Jonathan Garner */

using namespace std;



int main(int arg, char** argv)
{
	if (arg != 2)
	{
		cout << "usage: ./out size" << endl;
		return -1;
	}


	ofstream file_;
	file_.open("graph.txt");

	if (!file_.is_open()) {

		cout << "graph.txt was not accessible";
		return -1;

	}
	srand(time(NULL));

	int x = atoi(argv[1]);

	int** arr = (int**)malloc(x * sizeof(int*));
	int i = 0, j = 0;;

	for (i = 0; i < x; i++) {

		arr[i] = (int*)malloc(x * sizeof(int));
	}

	for (i = 0; i < x; i++) {

		for (j = 0; j < x; j++) {
			arr[i][j] = 0;
		}

	}
	int max = x*x - x - 1;
	int e = rand() % max + 1;  // random edge generate
	//e = 7000; // set to 7000 to test
	
	//cout << e;
	cout << "creating graph with: " << x << " vertices " << e-1 << " edges in graph.txt" << endl;
	
	for (int k = 0; k < e; k++) {
		i = rand() % x;
		j = rand() % x;

		if (arr[i][j] == -1) {
			k--;
		}
		else
		{
			if(arr[j][i] != 0){
				arr[i][j] = arr[j][i];
			}
			else if (i != j)
			{
				arr[i][j] = (rand() % 8) + 2;
				arr[j][i] = arr[i][j];
			}
			else {
				k--;
			}
		}

	}

	for (int i = 0; i < x; i++) {

		for (int j = 0; j < x; j++) {

			file_ << arr[i][j];
		}

		file_ << endl;
	}




	file_.close();


	return 0;

}