#ifndef MMM_UNROLLED_CUH
#define MMM_UNROLLED_CUH

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <iomanip>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

using namespace std::chrono;

//Compare matrices
template <typename T>
int compare_matrices(T *A, T *B, size_t m, size_t n) {
    float temp = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float prev_temp = temp;
            temp += (float)(A[j + n*i] - B[j + n*i]) * (A[j + n*i] - B[j + n*i]);
            if (temp - prev_temp > 0.1) {
                std::cout << "Matrix 1: " << A[j + n*i] << ", " << i << ", " << j << ", Matrix 2: " << B[j + n*i] << "\n";
                return -1;
            }
        }
    }
    printf("MSE: %f\n", temp/(float)(m*n));
    return 0;
}

//Generates random matrix
template <typename T>
void gen_rand_mat(T *Mat, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Mat[j + n*i] = (T)((2*((float)rand()/(float)RAND_MAX)-1) * 10);
        }
    }
}

//Prints matrix
template <typename T>
void print_matrix(T *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        printf("|| ");
        for (int j = 0; j < n; j++) {
            std::cout /*<< std::setprecision(3)*/ << A[j + n*i] << "\t";
        }
        printf(" ||\n");
    }
    printf("\n");
}

//Unrolls the outermost loop and performs matrix multiplication in the GPU (performs C := A*B)
template <int unrollFactor, typename T>
__global__ void matrix_multiply_unrolled_one_dim(T *A, T *B, T *C, size_t m, size_t n, size_t p) {
    int tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    T temp;

    //Unrolling outer loops and multiplying the elements present in shared memory
    #pragma unroll (unrollFactor)
    for (int i = 0; i < unrollFactor; i++) {
        temp = 0;
        if (tid1*unrollFactor + i < m && tid2 < p) {
            for (int k = 0; k < n; k++) {
                temp += A[n*(tid1*unrollFactor + i) + k]*B[k*p + tid2];
            }
            C[(tid1*unrollFactor + i)*p + tid2] = temp;
        }
    }
}

//Both outer loops unrolled matrix multiplication in the GPU (performs C := A*B)
template <int unrollFactor, typename T>
__global__ void matrix_multiply_unrolled(T *A, T *B, T *C, size_t m, size_t n, size_t p) {
    int tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    T temp = 0;

    //Unrolling outer loops and multiplying the elements present in shared memory
    #pragma unroll (unrollFactor)
    for (int i = 0; i < unrollFactor; i++) {
        #pragma unroll (unrollFactor)
        for (int j = 0; j < unrollFactor; j++) {
            temp = 0;
            if (tid1*unrollFactor + i < m && tid2*unrollFactor + j < p) {
                for (int k = 0; k < n; k++) {
                    temp += A[n*(tid1*unrollFactor + i) + k]*B[k*p + (tid2*unrollFactor + j)];
                }
                C[(tid1*unrollFactor + i)*p + tid2*unrollFactor + j] = temp;
            }
        }
    }
}

//Function to add matrices in GPU
template <typename T>
__global__ void add_mat_gpu(T *A, T *B, T *C, int n) {
    int tid, tid1, tid2;
    tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    tid = tid1*n + tid2;
    if (tid1 < n && tid2 < n) {
        C[tid] = A[tid] + B[tid];
    }
}

//Function to subtract matrices in GPU
template <typename T>
__global__ void sub_mat_gpu(T *A, T *B, T *C, int n) {
    int tid, tid1, tid2;
    tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    tid = tid1*n + tid2;
    if (tid1 < n && tid2 < n) {
        C[tid] = A[tid] - B[tid];
    }
}

template <typename T>
class mmm_unrolled {

public:
    mmm_unrolled() {}
    ~mmm_unrolled() {}

    //Performs normal matrix multiplication in the CPU
    int matrix_multiply_normal(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t, bool v) {
        if (t == true) {
            start = high_resolution_clock::now();
        }
        mul_mat(A, B, C, m, n, p);
        if (t == true) {
            finish = high_resolution_clock::now();
            serial_time = duration_cast<duration<double>>(finish - start).count();
            if (v == true) 
                printf("Serial MM Multiply Time: %f sec(s)\n\n", serial_time);
        }
        return 0;
    }

    //Add matrices
    inline void add_mat(T *A, T *B, T *C, size_t m, size_t n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i*n + j] = A[i*n + j] + B[i*n + j];
            }
        }
    }

    //Subtract matrices
    inline void sub_mat(T *A, T *B, T *C, size_t m, size_t n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i*n + j] = A[i*n + j] - B[i*n + j];
            }
        }
    }

    //Multiply matrices
    inline void mul_mat(T *A, T *B, T *C, size_t m, size_t n, size_t p) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                C[i*p + j] = 0;
                for (int k = 0; k < n; k++) {
                    C[i*p + j] += A[i*n + k] * B[k*p + j];
                }
            }
        }
    }

    //Performs unrolled matrix multiplication in the CPU (4 output elements are calculated for each iteration)
    int matrix_multiply_unrolled_cpu(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t = false, bool v = false) {
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
    template <int num_unroll>
    int matrix_multiply_unrolled_gpu_one_dim(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t = false, bool v = false) {
        //Getting device parameters of each GPU
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        int threads_per_block  = devProp.maxThreadsPerBlock;
        int thread_x, thread_y, block_x, block_y;
        thread_x = std::min((int)ceil((float)m/(float)num_unroll), (int)floor(sqrt(threads_per_block)));
        thread_y = std::min((int)p, (int)floor(sqrt(threads_per_block)));
        block_x = (int)ceil((float)((int)ceil((float)m/(float)num_unroll))/(float)thread_x);
        block_y = (int)ceil((float)p/(float)thread_y);

        //std::cout << thread_x << "," << thread_y << "," << block_x << "," << block_y << "," << ((int)ceil((float)m/(float)num_unroll)) << "\n";

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
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims/*, num_unroll*thread_x*thread_y*sizeof(T)*/>> >(d_A, d_B, d_C, m, n, p);
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

    //Performs unrolled matrix multiplication on the GPU.
    template <int num_unroll>
    int matrix_multiply_unrolled_gpu(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t = false, bool v = false) {
        //Getting device parameters of each GPU
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        int threads_per_block  = devProp.maxThreadsPerBlock;
        int thread_x, thread_y, block_x, block_y;
        thread_x = std::min((int)ceil(m/(float)num_unroll), std::min((int)floor(sqrt(threads_per_block)), 16));
        thread_y = std::min((int)ceil(p/(float)num_unroll), std::min((int)floor(sqrt(threads_per_block)), 16));
        block_x = (int)ceil((float)((int)ceil(m/(float)num_unroll))/(float)thread_x);
        block_y = (int)ceil((float)((int)ceil(p/(float)num_unroll))/(float)thread_y);

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
        matrix_multiply_unrolled<num_unroll><< <gridDims, blockDims/*, num_unroll*num_unroll*thread_x*thread_y*sizeof(T)*/>> >(d_A, d_B, d_C, m, n, p);
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


    //Implements Wingrad implementation of Strassens's algorithm for square matrices
    int strassens_algo_cpu(T *A, T *B, T *C, size_t n, bool t, bool v) {
        if (n%2 != 0) {
            printf("Please make number of elements per row and column even...\n");
            return -1;
        }

        //Temp matrices to store intermediate results
        T *T1, *T2, *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22;
        T1 = (T *)malloc((n/2)*(n/2)*sizeof(*T1));
        T2 = (T *)malloc((n/2)*(n/2)*sizeof(*T2));
        A11 = (T *)malloc((n/2)*(n/2)*sizeof(*A11));
        A12 = (T *)malloc((n/2)*(n/2)*sizeof(*A12));
        A21 = (T *)malloc((n/2)*(n/2)*sizeof(*A21));
        A22 = (T *)malloc((n/2)*(n/2)*sizeof(*A22));
        B11 = (T *)malloc((n/2)*(n/2)*sizeof(*B11));
        B12 = (T *)malloc((n/2)*(n/2)*sizeof(*B12));
        B21 = (T *)malloc((n/2)*(n/2)*sizeof(*B21));
        B22 = (T *)malloc((n/2)*(n/2)*sizeof(*B22));
        C11 = (T *)malloc((n/2)*(n/2)*sizeof(*C11));
        C12 = (T *)malloc((n/2)*(n/2)*sizeof(*C12));
        C21 = (T *)malloc((n/2)*(n/2)*sizeof(*C21));
        C22 = (T *)malloc((n/2)*(n/2)*sizeof(*C22));

        //Storing parts of input matrices in temporary matrices
        for (int i = 0; i < n/2; i++) {
            for (int j = 0; j < n/2; j++) {
                A11[i*(n/2) + j] = A[i*n + j];
                A12[i*(n/2) + j] = A[i*n + j + n/2];
                A21[i*(n/2) + j] = A[(i + n/2)*n + j];
                A22[i*(n/2) + j] = A[(i + n/2)*n + j + n/2];
                B11[i*(n/2) + j] = B[i*n + j];
                B12[i*(n/2) + j] = B[i*n + j + n/2];
                B21[i*(n/2) + j] = B[(i + n/2)*n + j];
                B22[i*(n/2) + j] = B[(i + n/2)*n + j + n/2];
            }
        }

        if (t == true) {
            start = high_resolution_clock::now();
        }
        //Implementation starts
        sub_mat(A11, A21, T1, n/2, n/2);
        sub_mat(B22, B12, T2, n/2, n/2);
        mul_mat(T1, T2, C21, n/2, n/2, n/2);
        add_mat(A21, A22, T1, n/2, n/2);
        sub_mat(B12, B11, T2, n/2, n/2);
        mul_mat(T1, T2, C22, n/2, n/2, n/2);
        sub_mat(T1, A11, T1, n/2, n/2);
        sub_mat(B22, T2, T2, n/2, n/2);
        mul_mat(T1, T2, C11, n/2, n/2, n/2);
        sub_mat(A12, T1, T1, n/2, n/2);
        mul_mat(T1, B22, C12, n/2, n/2, n/2);
        add_mat(C22, C12, C12, n/2, n/2);
        mul_mat(A11, B11, T1, n/2, n/2, n/2);
        add_mat(C11, T1, C11, n/2, n/2);
        add_mat(C11, C12, C12, n/2, n/2);
        add_mat(C11, C21, C11, n/2, n/2);
        sub_mat(T2, B21, T2, n/2, n/2);
        mul_mat(A22, T2, C21, n/2, n/2, n/2);
        sub_mat(C11, C21, C21, n/2, n/2);
        add_mat(C11, C22, C22, n/2, n/2);
        mul_mat(A12, B21, C11, n/2, n/2, n/2);
        add_mat(T1, C11, C11, n/2, n/2);
        //Implementation finished

        //Storing output parts in the output matrix
        for (int i = 0; i < n/2; i++) {
            for (int j = 0; j < n/2; j++) {
                C[i*n + j] = C11[i*(n/2) + j];
                C[i*n + j + n/2] = C12[i*(n/2) + j];
                C[(i + n/2)*n + j] = C21[i*(n/2) + j];
                C[(i + n/2)*n + j + n/2] = C22[i*(n/2) + j];
            }
        }

        if (t == true) {
            finish = high_resolution_clock::now();
            str_cpu = duration_cast<duration<double>>(finish - start).count();
            if (v == true) {
                printf("CPU Strassen-Winograd Algorithm: %f sec(s)\n\n", str_cpu);
            }
        }

        free(A11);
        free(A12);
        free(A21);
        free(A22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(C11);
        free(C12);
        free(C21);
        free(C22);
        free(T1);
        free(T2);

        return 0;
    }

    //Performs unrolled matrix multiplication on the GPU.
    //Returns 0 if correct, also shows time
    template <int num_unroll>
    int strassens_algo_gpu(T *A, T *B, T *C, size_t n, bool t = false, bool v = false) {
        if (n%2 != 0) {
            printf("Please make number of elements per row and column even...\n");
            return -1;
        }
        //Getting device parameters of first GPU in the system
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        int threads_per_block  = devProp.maxThreadsPerBlock;
        int thread_x, thread_y, block_x, block_y;
        thread_x = std::min((int)ceil((n/2)/(float)num_unroll), std::min((int)floor(sqrt(threads_per_block)), 16));
        thread_y = std::min((int)ceil((n/2)/(float)num_unroll), std::min((int)floor(sqrt(threads_per_block)), 16));
        block_x = (int)ceil((float)((int)ceil((n/2)/(float)num_unroll))/(float)thread_x);
        block_y = (int)(ceil((float)((int)ceil((n/2)/(float)num_unroll))/(float)thread_y));
        dim3 gridDims(block_x, block_y), blockDims(thread_x, thread_y);

        //Initializing sub-arrays for GPU
        T *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22, *T1, *T2;

        //Allocating memory for the arrays on the GPU
        cudaMalloc((void **)&A11, ((n*n)/4)*sizeof(*A11));
        cudaMalloc((void **)&A12, ((n*n)/4)*sizeof(*A12));
        cudaMalloc((void **)&A21, ((n*n)/4)*sizeof(*A21));
        cudaMalloc((void **)&A22, ((n*n)/4)*sizeof(*A22));
        cudaMalloc((void **)&B11, ((n*n)/4)*sizeof(*B11));
        cudaMalloc((void **)&B12, ((n*n)/4)*sizeof(*B12));
        cudaMalloc((void **)&B21, ((n*n)/4)*sizeof(*B21));
        cudaMalloc((void **)&B22, ((n*n)/4)*sizeof(*B22));
        cudaMalloc((void **)&C11, ((n*n)/4)*sizeof(*C11));
        cudaMalloc((void **)&C12, ((n*n)/4)*sizeof(*C12));
        cudaMalloc((void **)&C21, ((n*n)/4)*sizeof(*C21));
        cudaMalloc((void **)&C22, ((n*n)/4)*sizeof(*C22));
        cudaMalloc((void **)&T1, ((n*n)/4)*sizeof(*T1));
        cudaMalloc((void **)&T2, ((n*n)/4)*sizeof(*T2));

        //Copying input matrices in the GPU
        for (int i = 0; i < n/2; i++) {
            cudaMemcpy(&A11[i*n/2], &A[i*n], (n/2)*sizeof(*A11), cudaMemcpyHostToDevice);
            cudaMemcpy(&A12[i*n/2], &A[i*n + n/2], (n/2)*sizeof(*A12), cudaMemcpyHostToDevice);
            cudaMemcpy(&A21[i*n/2], &A[(i + n/2)*n], (n/2)*sizeof(*A21), cudaMemcpyHostToDevice);
            cudaMemcpy(&A22[i*n/2], &A[(i + n/2)*n + n/2], (n/2)*sizeof(*A22), cudaMemcpyHostToDevice);
            cudaMemcpy(&B11[i*n/2], &B[i*n], (n/2)*sizeof(*B11), cudaMemcpyHostToDevice);
            cudaMemcpy(&B12[i*n/2], &B[i*n + n/2], (n/2)*sizeof(*B12), cudaMemcpyHostToDevice);
            cudaMemcpy(&B21[i*n/2], &B[(i + n/2)*n], (n/2)*sizeof(*B21), cudaMemcpyHostToDevice);
            cudaMemcpy(&B22[i*n/2], &B[(i + n/2)*n + n/2], (n/2)*sizeof(*B22), cudaMemcpyHostToDevice);
        }

        //Matrix multiplication
        if (t == true) {
            start = high_resolution_clock::now();
        }
        //Implementation starts
        sub_mat_gpu<< <gridDims, blockDims>> >(A11, A21, T1, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(B22, B12, T2, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(T1, T2, C21, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(A21, A22, T1, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(B12, B11, T2, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(T1, T2, C22, n/2, n/2, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(T1, A11, T1, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(B22, T2, T2, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(T1, T2, C11, n/2, n/2, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(A12, T1, T1, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(T1, B22, C12, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C22, C12, C12, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(A11, B11, T1, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, T1, C11, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, C12, C12, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, C21, C11, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(T2, B21, T2, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(A22, T2, C21, n/2, n/2, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(C11, C21, C21, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, C22, C22, n/2);
        matrix_multiply_unrolled_one_dim<num_unroll><< <gridDims, blockDims>> >(A12, B21, C11, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(T1, C11, C11, n/2);
        //Implementation finished
        cudaDeviceSynchronize();
        if (t == true) {
            finish = high_resolution_clock::now();
            str_gpu = duration_cast<duration<double>>(finish - start).count();
            if (v == true) {
                printf("GPU MM Multiply Strassen Winograd Time: %f sec(s)\n\n", str_gpu);
            }
        }

        //Copying output matrix from GPU
        for (int i = 0; i < n/2; i++) {
            cudaMemcpy(&C[i*n], &C11[i*n/2], (n/2)*sizeof(*C11), cudaMemcpyDeviceToHost);
            cudaMemcpy(&C[i*n + n/2], &C12[i*n/2], (n/2)*sizeof(*C12), cudaMemcpyDeviceToHost);
            cudaMemcpy(&C[(i + n/2)*n], &C21[i*n/2], (n/2)*sizeof(*C21), cudaMemcpyDeviceToHost);
            cudaMemcpy(&C[(i + n/2)*n + n/2], &C22[i*n/2], (n/2)*sizeof(*C22), cudaMemcpyDeviceToHost);
        }

        cudaFree(A11);
        cudaFree(A12);
        cudaFree(A21);
        cudaFree(A22);
        cudaFree(B11);
        cudaFree(B12);
        cudaFree(B21);
        cudaFree(B22);
        cudaFree(C11);
        cudaFree(C12);
        cudaFree(C21);
        cudaFree(C22);
        cudaFree(T1);
        cudaFree(T2);
        cudaDeviceReset();
        return 0;

    }

    //Checks status of cublas and other function returns
    void status_check(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (status == CUBLAS_STATUS_INVALID_VALUE) {
                std::cout << "CUBLAS_STATUS_INVALID_VALUE.\n";
            } else if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
                std::cout << "CUBLAS_STATUS_NOT_INITIALIZED.\n";
            } else if (status == CUBLAS_STATUS_ARCH_MISMATCH) {
                std::cout << "CUBLAS_STATUS_ARCH_MISMATCH.\n";
            } else {
                std::cout << "CUBLAS_STATUS_EXECUTION_FAILED.\n";
            }
            std::cout << "Err: function not performed.\n";
        }
    }

    //Gets the latest execution time of the function mentioned as an argument
    double getExecTime(std::string type) {
        if (type == "normal") {
            return serial_time;
        } else if (type == "cpu_unrolled") {
            return cpu_unrolled_time;
        } else if (type == "gpu_unrolled") {
            return gpu_unrolled_time;
        } else if (type == "gemm") {
            return gemm_time;
        } else if (type == "strassens_cpu") {
            return str_cpu;
        } else if (type == "strassens_gpu") {
            return str_gpu;        
        } else {
            printf("Does not exist...\n");
            return -1;
        }
    }

private:
    high_resolution_clock::time_point start, finish;
    double serial_time = 0, cpu_unrolled_time = 0, gpu_unrolled_time = 0, gemm_time = 0, str_cpu = 0, str_gpu = 0;

};

#endif