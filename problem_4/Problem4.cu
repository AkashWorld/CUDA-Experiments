#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

using namespace std::chrono;

int N = 1000000;

#define THREADS_PER_BLK 128

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

void get_random_nums(int* randoms, int length) {
  std::srand(std::time(0));
  for (int i = 0; i < length; i++) {
    randoms[i] = std::rand() % 101;
  }
}

__global__ void exclusive_scan_gpu(int* input, int* output, int n) {
  __shared__ int temp[4 * THREADS_PER_BLK];
  int thid_global = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  int thid = threadIdx.x;

  {  
    int offset = 1;
    
    int ai = thid;
    int bi = thid + n / 2;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = input[thid_global];
    temp[bi + bankOffsetB] = input[thid_global + n / 2];  
     

    for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) { 
      //temp[n - 1] = 0;
      temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        int t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }

    __syncthreads();

    output[thid_global] = temp[ai + bankOffsetA];
    output[thid_global + n / 2] = temp[bi + bankOffsetB];
  }

}

__global__ void find_repeats(int* input, int* length){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int fl = 0;
	int arrB[1000000];
	int arrC[1000000];
	
	
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			if(input[i] == input[i+1]){
				fl = 1;
				arrB[j] = i;
				break;
			}
		}
		if(fl == 0){
			arrC[i] = input[i];
		}
		fl = 0;
	}
}
/* writing out the general idea in c
void find_repeats(int *a, int size)
{
  int i, j, k;
  int fl = 0;
  int arrB[4];
  int arrC[6];

  for( i = 0; i < size; i++){
    for(j = 0; j < size; j++){
      if(a[i] == a[i+1]){
	fl = 1;
	arrB[j] = i;
	//	printf("%d\n", arrB[j]);
	break;
      }
	   }//end j for loop
    if(fl == 0){
      arrC[i] = a[i];
      printf("%d\n", arrC[i]);
    }
    fl = 0;
  }//end i for loop
    
}
main(){

		int size = 10;
		int arr[10] = { 1, 2, 2, 1, 1, 1, 3, 5, 3, 3};
		int sum[10];
		
		find_repeats(arr, size);
		//	exclusive_prefix_sum(arr, sum, size);
		
		
}
*/
