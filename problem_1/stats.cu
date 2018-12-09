#include <iostream>
#include <math.h>
#include <ctime>
#include <cstdlib>

#define THREADS_PER_BLK 256

using namespace std;

struct stats {
    double mean;
    double min;
    double max;
    double stddev;
};

// Kernel function to find the maximum element of an array
__global__ void get_gpu_max(int n, double *x, double *results) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    double max = x[index];
    for (int i = index + 1; i < n; i += stride) {
        max = (max < x[i]) ? x[i] : max;
    }
    results[threadIdx.x] = max;
}

// Kernel function to find the minimum element of an array
__global__ void get_gpu_min(int n, double *x, double *results) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    double min = x[index];
    for (int i = index + 1; i < n; i += stride) {
        min = (x[i] < min) ? x[i] : min;
    }
    results[threadIdx.x] = min;
}
    
// CPU function to find mean of an array
double cpu_get_mean(int n, double *x) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum/n;
}

// use CPU to calculate std deviation (Welford's algorithm)
double cpu_get_stddev(int n, double *x){
    double mean = x[0];
    double M2 = 0;
    double delta;
    double delta2;
    for (int i = 1; i < n; i++){
        delta = x[i] - mean;
        mean += delta/(i+1);
        delta2 = x[i] - mean;
        M2 += delta * delta2;
    }
    return sqrt(M2/n);
}

// CPU function to find max element of an array
double cpu_get_max(int n, double *x) {
    double max = x[0];
    for (int i = 1; i < n; i++) {
        max = (max < x[i]) ? x[i] : max;
    }
    return max;
}

// CPU function to find min element of an array
double cpu_get_min(int n, double *x) {
    double min = x[0];
    for (int i = 1; i < n; i++) {
        min = (x[i] < min) ? x[i] : min;
    }
    return min;
}

// use CPU to calculate min, mean, max, std deviation (Welford's algorithm)
stats cpu_get_all(int n, double *x){
    stats myStats; 
    double mean = x[0];
    double min = x[0];
    double max = x[0];
    double M2 = 0;
    double delta;
    double delta2;
    for (int i = 1; i < n; i++){
        max = (max < x[i]) ? x[i] : max;
        min = (x[i] < min) ? x[i] : min;
        delta = x[i] - mean;
        mean += delta/(i+1);
        delta2 = x[i] - mean;
        M2 += delta * delta2;
    }
    myStats.mean = mean;
    myStats.min = min;
    myStats.max = max;
    myStats.stddev = sqrt(M2/n);
    return myStats;
}

int main(void) {
    int N = 195312*THREADS_PER_BLK; // ~50M
    cout.precision(17);
   
    // Allocate memory and initialize x
    cout << "Allocating memory and initializing...";
    double *x;
    cudaMallocManaged(&x, N*sizeof(double));
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
      x[i] = ((double) rand()) / ((double) RAND_MAX);
    }
    double *results;
    cudaMallocManaged(&results, THREADS_PER_BLK*sizeof(double));
    cout << "Done\n";

    // use CPU to calculate max
    double cpu_max = cpu_get_max(N, x);
    cout << "CPU calculated max:" << fixed << cpu_max << endl;

    // use GPU to calculate max
    get_gpu_max<<<1, THREADS_PER_BLK>>>(N, x, results);
    cudaDeviceSynchronize();
    double gpu_max = results[0];
    for (int i = 1; i < THREADS_PER_BLK; i++) {
        gpu_max = (gpu_max < results[i]) ? results[i] : gpu_max;
    }
    cout << "GPU calculated max:" << fixed << gpu_max << endl;

    // use CPU to calculate min
    double cpu_min = cpu_get_min(N, x);
    cout << "CPU calculated min:" << fixed << cpu_min << endl;

    // use GPU to calculate min
    get_gpu_min<<<1, THREADS_PER_BLK>>>(N, x, results);
    cudaDeviceSynchronize();
    double gpu_min = results[0];
    for (int i = 1; i < THREADS_PER_BLK; i++) {
        gpu_min = (results[i] < gpu_min) ? results[i] : gpu_min;
    }
    cout << "GPU calculated min:" << fixed << gpu_min << endl;

    // use CPU to calculate mean
    double cpu_mean = cpu_get_mean(N, x);
    cout << "CPU calculated mean:" << fixed << cpu_mean << endl;

    // use CPU to calculate std dev
    double cpu_stddev = cpu_get_stddev(N, x);
    cout << "CPU calculated std dev:" << fixed << cpu_stddev << endl;

    // use CPU to calculate all stats
    stats my_stats = cpu_get_all(N, x);
    cout << "Concurrent: CPU calculated max:" << fixed << my_stats.max << endl;
    cout << "Concurrent: CPU calculated min:" << fixed << my_stats.min << endl;
    cout << "Concurrent: CPU calculated mean:" << fixed << my_stats.mean << endl;
    cout << "Concurrent: CPU calculated std dev:" << fixed << my_stats.stddev << endl;

    // Free memory
    cudaFree(x);
    cudaFree(results);

    return 0;
}
