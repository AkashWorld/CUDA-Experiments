#ifndef MMM_UNROLLED_CUH
#define MMM_UNROLLED_CUH

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std::chrono;

//Unrolled matrix multiplication in the GPU (performs C := C + A*B)
template <typename T>
__global__ void matrix_multiply_unrolled(T *A, T *B, T *C, size_t m, size_t n, size_t p) {
    //extern __shared__ T temp_arr[];
    //temp_arr[threadIdx.x + blockDim.x*threadIdx.y] = 0;
    int tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    T temp = 0;
    for (int k = 0; k < n; k++) {
        if (tid1 < m && tid2 < p) {
           /* temp_arr[threadIdx.x + blockDim.x*threadIdx.y]*/temp += A[n*tid1 + k]*B[k*p + tid2];
        }
    }
    if (tid1 < m && tid2 < p) {
        C[tid1*p + tid2] = temp;
    }
}

template <typename T>
class mmm_unrolled {

public:
    mmm_unrolled() {}
    ~mmm_unrolled() {}

    //Performs normal matrix multiplication in the CPU
  //  template <typename T>
    int matrix_multiply_normal(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t, bool v) {
        if (t == true) {
            start = high_resolution_clock::now();
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                C[i*p + j] = 0;
                for (int k = 0; k < n; k++) {
                    C[i*p + j] += A[i*n + k] * B[k*p + j];
                }
            }
        }
        if (t == true) {
            finish = high_resolution_clock::now();
            serial_time = duration_cast<duration<double>>(finish - start).count();
            if (v == true) 
                printf("Serial MM Multiply Time: %f sec(s)\n\n", serial_time);
        }
        return 0;
    }

    //Performs unrolled matrix multiplication in the CPU
  //  template <typename T>
    int matrix_multiply_unrolled_cpu(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t, bool v) {
        T t11 = 0, t12 = 0, t21 = 0, t22 = 0;//, a11, a12, a21, a22, b11, b12, b21, b22;
        if (t == true) {
            start = high_resolution_clock::now();
        }
        for (int i = 0; i < m; i += 2) {
            for (int j = 0; j < p; j += 2) {
                t11 = 0;
                t12 = 0;
                t21 = 0;
                t22 = 0;
                /*
                t11 = C[i*p + j];
                if (j+1 < p) {
                    t12 = C[i*p + j+1];
                }
                if (i+1 < m) {
                    t21 = C[(i+1)*p + j];
                }
                if (j+1 < p && i+1 < m) {
                    t22 = C[(i+1)*p + j+1];
                }
                */
                for (int k = 0; k < n; k++) {
                    t11 += A[i*n + k]*B[k*p + j];
                    if (j+1 < p) {
                        t12 += A[i*n + k]*B[k*p + j+1];
                    }
                    if (i+1 < m) {
                        t21 += A[(i+1)*n + k]*B[k*p + j];
                    }
                    if (i+1 < m && j+1 < p) {
                        t22 += A[(i+1)*n + k]*B[k*p + j+1];
                    }
                }
                C[i*p + j] = t11;
                if (j+1 < p) {
                    C[i*p + j+1] = t12;
                }
                if (i+1 < m) {
                    C[(i+1)*p + j] = t21;
                }
                if (i+1 < m && j+1 < p) {
                    C[(i+1)*p + j+1] = t22;
                }
            }
        }
        if (t == true) {
            finish = high_resolution_clock::now();
            cpu_unrolled_time = duration_cast<duration<double>>(finish - start).count();
            if (v == true)
                printf("CPU MM Multiply Unrolled Time: %f sec(s)\n\n", cpu_unrolled_time);
        }
        return 0;
    }

    //Performs unrolled matrix multiplication on the GPU.
    //Returns 0 if correct, also shows time
   // template <typename T>
    int matrix_multiply_unrolled_gpu(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t, bool v) {
        //Getting device parameters of each GPU
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        int threads_per_block  = devProp.maxThreadsPerBlock;
        int thread_x, thread_y, block_x, block_y;
        thread_x = std::min((int)m,(int)floor(sqrt(threads_per_block)));
        thread_y = std::min((int)p,(int)floor(sqrt(threads_per_block)));
        block_x = (int)ceil((float)m/(float)thread_x);
        block_y = (int)ceil((float)p/(float)thread_y);

        //Initializing arrays for GPU
        T *d_A, *d_B, *d_C;

        //Allocating memory for the arrays on the GPU
        cudaMalloc((void **)&d_A, m*n*sizeof(*d_A));
        cudaMalloc((void **)&d_B, n*p*sizeof(*d_B));
        cudaMalloc((void **)&d_C, m*p*sizeof(*d_C));

        //Copying input matrices in the GPU
        cudaMemcpy(d_A, A, m*n*sizeof(*d_A), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n*p*sizeof(*d_B), cudaMemcpyHostToDevice);

        //Matrix multiplication
        if (t == true) {
            start = high_resolution_clock::now();
        }
        dim3 gridDims(block_x, block_y), blockDims(thread_x, thread_y);
        matrix_multiply_unrolled<< <gridDims, blockDims>> >(d_A, d_B, d_C, m, n, p);
        cudaDeviceSynchronize();
        if (t == true) {
            finish = high_resolution_clock::now();
            gpu_unrolled_time = duration_cast<duration<double>>(finish - start).count();
            if (v == true) {
                printf("GPU MM Multiply Unrolled Time: %f sec(s)\n\n", gpu_unrolled_time);
            }
        }

        //Copying output matrix from GPU
        cudaMemcpy(C, d_C, m*p*sizeof(*d_C), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaDeviceReset();
        return 0;

    }

    double getExecTime(std::string type) {
        if (type == "normal") {
            return serial_time;
        } else if (type == "cpu_unrolled") {
            return cpu_unrolled_time;
        } else if (type == "gpu_unrolled") {
            return gpu_unrolled_time;
        } else {
            printf("Does not exist...\n");
            return -1;
        }
    }

private:
    high_resolution_clock::time_point start, finish;
    double serial_time = 0, cpu_unrolled_time = 0, gpu_unrolled_time = 0;

};

#endif