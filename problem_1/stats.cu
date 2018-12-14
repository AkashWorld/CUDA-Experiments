#include <iostream>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

struct stats {
    double mean;
    double min;
    double max;
    double stddev;
};

    
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
    double m2 = 0;
    double delta;
    double delta2;
    for (int i = 1; i < n; i++){
        delta = x[i] - mean;
        mean += delta/(i+1);
        delta2 = x[i] - mean;
        m2 += delta * delta2;
    }
    return sqrt(m2/n);
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
    double m2 = 0;
    double delta;
    double delta2;
    for (int i = 1; i < n; i++){
        max = (max < x[i]) ? x[i] : max;
        min = (x[i] < min) ? x[i] : min;
        delta = x[i] - mean;
        mean += delta/(i+1);
        delta2 = x[i] - mean;
        m2 += delta * delta2;
    }
    myStats.mean = mean;
    myStats.min = min;
    myStats.max = max;
    myStats.stddev = sqrt(m2/n);
    return myStats;
}

// Kernel function to find the maximum element of an array
__global__ void get_gpu_max(int n, double *x, double *results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double max = x[index];
    for (int i = index + stride; i < n; i += stride) {
        max = (max < x[i]) ? x[i] : max;
    }
    results[index] = max;
}

// Kernel function to find the minimum element of an array
__global__ void get_gpu_min(int n, double *x, double *results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double min = x[index];
    for (int i = index + stride; i < n; i += stride) {
        min = (x[i] < min) ? x[i] : min;
    }
    results[index] = min;
}

// kernel to calculate the mean on the GPU
__global__ void get_gpu_mean(int n, double *x, double *results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double mean = x[index];
    int count = 1;
    for (int i = index + stride; i < n; i += stride){
        count++;
        mean += (x[i] - mean)/count;
    }
    results[index] = mean;
}

// Calculate std deviation on the GPU
__global__ void get_gpu_stddev(int n, double *x, double *results){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double mean = x[index];
    double m2 = 0;
    double delta;
    double delta2;
    int count = 1;
    for (int i = index + stride; i < n; i += stride){
        count++;
        delta = x[i] - mean;
        mean += delta/count;
        delta2 = x[i] - mean;
        m2 += delta * delta2;
    }
    results[index] = m2;
}



// caluclate all stats on the GPU
__global__ void get_gpu_all(int n, double *x, stats *all_results){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double mean = x[index];
    double min = x[index];
    double max = x[index];
    double m2 = 0;
    double delta;
    double delta2;
    int count = 1;
    for (int i = index + stride; i < n; i += stride){
        max = (max < x[i]) ? x[i] : max;
        min = (x[i] < min) ? x[i] : min;
        count++;
        delta = x[i] - mean;
        mean += delta/count;
        delta2 = x[i] - mean;
        m2 += delta * delta2;
    }
    all_results[index].mean = mean;
    all_results[index].min = min;
    all_results[index].max = max;
    all_results[index].stddev = m2; // m2 not actually std dev
}

void print_diff(double x, double y){
    cout << "Difference: " << 100*(y - x)/x << "%\n";
}


void run_tests(int N_pre, int N_BLOCKS, int THREADS_PER_BLK) {

    // We need N to be a multiple of N_THREADS
    int N = N_BLOCKS * THREADS_PER_BLK * floor(N_pre / (THREADS_PER_BLK * N_BLOCKS));
    
    /**  
    cout << "N = " << N << endl;
    cout << "N_BLOCKS = " << N_BLOCKS << endl;
    cout << "THREADS_PER_BLK = " << THREADS_PER_BLK << endl;
    cout << "Allocating memory and initializing...";
    **/
    double *x;
    cudaMallocManaged(&x, N*sizeof(double));
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
      x[i] = ((double) rand()) / ((double) RAND_MAX);
    }
    double *results;
    cudaMallocManaged(&results, N_BLOCKS*THREADS_PER_BLK*sizeof(double));

    // use CPU to calculate max
    auto start = std::chrono::high_resolution_clock::now();
    double cpu_max = cpu_get_max(N, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    // cout << "CPU calculated max:" << fixed << cpu_max << endl;
    // fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,max,cpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());


    // use GPU to calculate max
    start = std::chrono::high_resolution_clock::now();
    get_gpu_max<<<N_BLOCKS, THREADS_PER_BLK>>>(N, x, results);
    cudaDeviceSynchronize();
    double gpu_max = results[0];
    for (int i = 1; i < N_BLOCKS*THREADS_PER_BLK; i++) {
        gpu_max = (gpu_max < results[i]) ? results[i] : gpu_max;
    }
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "GPU calculated max:" << fixed << gpu_max << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,max,gpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());

    //print_diff(cpu_max, gpu_max);
    //cout << endl;

    // use CPU to calculate min
    start = std::chrono::high_resolution_clock::now();
    double cpu_min = cpu_get_min(N, x);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "CPU calculated min:" << fixed << cpu_min << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,min,cpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());

    // use GPU to calculate min
    start = std::chrono::high_resolution_clock::now();
    get_gpu_min<<<N_BLOCKS, THREADS_PER_BLK>>>(N, x, results);
    cudaDeviceSynchronize();
    double gpu_min = results[0];
    for (int i = 1; i < N_BLOCKS*THREADS_PER_BLK; i++) {
        gpu_min = (results[i] < gpu_min) ? results[i] : gpu_min;
    }
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "GPU calculated min:" << fixed << gpu_min << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,min,gpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());

    //print_diff(cpu_min, gpu_min);
    //cout << endl;

    // use CPU to calculate mean
    start = std::chrono::high_resolution_clock::now();
    double cpu_mean = cpu_get_mean(N, x);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "CPU calculated mean:" << fixed << cpu_mean << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,avg,cpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());

    // use GPU to calculate mean
    start = std::chrono::high_resolution_clock::now();
    get_gpu_mean<<<N_BLOCKS, THREADS_PER_BLK>>>(N, x, results);
    cudaDeviceSynchronize();
    double gpu_mean_sum = 0;
    for (int i = 0; i < N_BLOCKS*THREADS_PER_BLK; i++) {
        gpu_mean_sum += results[i];
    }
    double gpu_mean = gpu_mean_sum/(N_BLOCKS*THREADS_PER_BLK);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "GPU calculated mean:" << fixed << gpu_mean << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,avg,gpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());

    //print_diff(cpu_mean, gpu_mean);
    //cout << endl;

    // use CPU to calculate std dev
    start = std::chrono::high_resolution_clock::now();
    double cpu_stddev = cpu_get_stddev(N, x);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "CPU calculated std dev:" << fixed << cpu_stddev << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,dev,cpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());

    // use GPU to calculate std dev
    start = std::chrono::high_resolution_clock::now();
    get_gpu_stddev<<<N_BLOCKS, THREADS_PER_BLK>>>(N, x, results);
    cudaDeviceSynchronize();
    double gpu_m2 = 0;
    for (int i = 0; i < N_BLOCKS*THREADS_PER_BLK; i++) {
        gpu_m2 += results[i];
    }
    double gpu_stddev = sqrt(gpu_m2/N);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    //cout << "GPU calculated std dev:" << fixed << gpu_stddev << endl;
    //fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    fprintf(stdout,"%d,%d,%d,dev,gpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());
    /**
    print_diff(cpu_stddev, gpu_stddev);
    cout << endl;
    **/

    // use CPU to calculate all stats
    start = std::chrono::high_resolution_clock::now();
    stats my_stats = cpu_get_all(N, x);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    /**
    cout << "Concurrent: CPU calculated max:" << fixed << my_stats.max << endl;
    cout << "Concurrent: CPU calculated min:" << fixed << my_stats.min << endl;
    cout << "Concurrent: CPU calculated mean:" << fixed << my_stats.mean << endl;
    cout << "Concurrent: CPU calculated std dev:" << fixed << my_stats.stddev << endl;
    fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    **/
    fprintf(stdout,"%d,%d,%d,all,cpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());
    

    cudaFree(results);

    // use GPU to calculate all stats
    stats* all_results;
    cudaMallocManaged(&all_results, N_BLOCKS*THREADS_PER_BLK*sizeof(stats));
    
    // start the timer
    start = std::chrono::high_resolution_clock::now();

    // run calculations on the GPU
    get_gpu_all<<<N_BLOCKS, THREADS_PER_BLK>>>(N, x, all_results);

    // synchrnonize 
    cudaDeviceSynchronize();

    // We now need to accumulate results from all threads
    double m2 = all_results[0].stddev;
    double mean = all_results[0].mean;
    double delta;
    double new_mean;
    int n_a = N / (N_BLOCKS*THREADS_PER_BLK); 
    int n_b = n_a;
    double max = all_results[0].max;
    double min = all_results[0].min;
    for (int i = 1; i < N_BLOCKS*THREADS_PER_BLK; i++) {
        new_mean = all_results[i].mean;
        delta = new_mean - mean;

        // we update our running mean value
        mean = (n_a*mean + n_b*new_mean)/(n_a + n_b);

        m2 += all_results[i].stddev + delta * delta * n_a * n_b / (n_a + n_b);

        n_a += n_b;

        min = (all_results[i].min < min) ? all_results[i].min : min;
        max = (all_results[i].max > max) ? all_results[i].max : max;
    }
    double stddev = sqrt(m2/N);
    end = std::chrono::high_resolution_clock::now();
    dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    /**
    cout << "Concurrent: GPU calculated max:" << fixed << max << endl;
    cout << "Concurrent: GPU calculated min:" << fixed << min << endl;
    cout << "Concurrent: GPU calculated mean:" << fixed << mean << endl;
    cout << "Concurrent: GPU calculated std dev:" << fixed << stddev << endl;
    fprintf(stdout, "Elapsed time %lld ns\n", dur_ns.count());
    **/
    fprintf(stdout,"%d,%d,%d,all,gpu,%lld\n",N,N_BLOCKS,THREADS_PER_BLK,dur_ns.count());
    
    // Free memory
    cudaFree(x);
    cudaFree(all_results);
}


int main(void) {

    // We want to display floats with max precision
    cout.precision(17);

    int Ns[] = {50000000,100000000,150000000,200000000};
    int TPBs[] = {128, 256, 512, 1024};
    int NBs[] = {1, 4, 16, 28};

    for (int n : Ns) {
        for (int threads_per_block : TPBs) {
            for (int n_blocks : NBs) {
                run_tests(n, n_blocks, threads_per_block); 
            }
        }
    }
   
}
