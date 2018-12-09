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

#ifndef NUM_UNROLL
    #define NUM_UNROLL 2
#endif

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


//Unrolled matrix multiplication in the GPU (performs C := C + A*B)
template <int unrollFactor, typename T>
__global__ void matrix_multiply_unrolled(T *A, T *B, T *C, size_t m, size_t n, size_t p) {
    //extern __shared__ T temp_arr[];
    //temp_arr[threadIdx.x + blockDim.x*threadIdx.y] = 0;
    int tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    T temp = 0;
//    T *temp_A = temp_arr, *temp_B = &temp_arr[NUM_UNROLL*blockDim.x];

    for (int k = 0; k < n; k += NUM_UNROLL) {
        /*
        #pragma unroll (unrollFactor)
        for (int j = 0; j < NUM_UNROLL; j++) {
            temp_A[j + threadIdx.x*NUM_UNROLL] = A[n*tid1 + k + j];
            temp_B[j + threadIdx.y*NUM_UNROLL] = B[(k+j)*p + tid2];
        }
*/
        #pragma unroll (unrollFactor)
        for (int j = 0; j < NUM_UNROLL; j++) {
            if (tid1 < m && tid2 < p) {
                temp += A[n*tid1 + k + j]*B[(k+j)*p + tid2];
            }
        }
    }
    if (n % NUM_UNROLL != 0) {
        for (int k = n - (n % NUM_UNROLL); k < n; k++) {
            if (tid1 < m && tid2 < p) {
                temp += A[n*tid1 + k]*B[k*p + tid2];
            }
        }
    }
    if (tid1 < m && tid2 < p) {
        C[tid1*p + tid2] = temp;
    }
}

/*
//Strassens algorithm in GPU
template <typename T>
__global__ void strassen_algo(T *A11, T *A12, T *A21, T *A22, T *B11, T *B12, T *B21, T *B22, T *C11, T *C12, T *C21, T *C22, T *T1, T *T2, size_t n_) {
    int n = n_/2, tid, tid1, tid2;
    T temp = 0;
    tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
    tid = tid1*n + tid2;

    if (tid1 < n && tid2 < n) {
        //Steps 1,2,3
        C12[tid] = A21[tid] - A11[tid];
        C21[tid] = B11[tid] + B12[tid];
        __syncthreads();
        temp = 0;
        for (int k = 0; k < n; k++) {
            temp += C12[n*tid1 + k]*C21[k*n + tid2];
        }
        C22[tid] = temp;
        //Steps 4,5,6
        C12[tid] = A12[tid] - A22[tid];
        C21[tid] = B21[tid] + B22[tid];
        __syncthreads();
        temp = 0;
        for (int k = 0; k < n; k++) {
            temp += C12[n*tid1 + k]*C21[k*n + tid2];
        }
        C11[tid] = temp;
        //Steps 7,8,9,10,11
        C12[tid] = A11[tid] + A22[tid];
        C21[tid] = B11[tid] + B22[tid];
        __syncthreads();
        temp = 0;
        for (int k = 0; k < n; k++) {
            temp += C12[n*tid1 + k]*C21[k*n + tid2];
        }
        T1[tid] = temp;
        C11[tid] += T1[tid];
        C22[tid] += T1[tid];
        //Steps 12,13,14
        T2[tid] = A21[tid] + A22[tid];
        __syncthreads();
        temp = 0;
        for (int k = 0; k < n; k++) {
            temp += T2[n*tid1 + k]*B11[k*n + tid2];
        }

    }
}

//Strassen Winograd algorithm in GPU
__device__ bool flag1, flag2;
template <typename T>
__global__ void strassen_winograd_algo(T *A11, T *A12, T *A21, T *A22, T *B11, T *B12, T *B21, T *B22, T *C11, T *C12, T *C21, T *C22, T *T1, T *T2, size_t n_) {
    //n/2 is the size of the sub-matrices
    int n = n_/2, tid, tid1, tid2;
    T temp = 0;// temp_t1, temp_t2;
  //  if (blockIdx.y < gridDim.y) {
        //thread idx for first half of iterations
        tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*blockIdx.y;
        tid = tid1*n + tid2;
        /*
        //Flags to synchronize between two parts
        if (tid == 0)
            flag1 = false;
            
        if (tid1 < n && tid2 < n) {
            //Steps 1, 2, 3
            T1[tid] = A11[tid] - A21[tid];
            T2[tid] = B22[tid] - B12[tid];
            __syncthreads();
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += T1[n*tid1 + k]*T2[k*n + tid2];
            }
            C21[tid] = temp;

            //Steps 4, 5, 6
            T1[tid] = A21[tid] + A22[tid];
            T2[tid] = B12[tid] - B11[tid];
            __syncthreads();
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += T1[n*tid1 + k]*T2[k*n + tid2];
            }
            C22[tid] = temp;

            //Steps 7, 8, 9
            T1[tid] = T1[tid] - A11[tid];
            T2[tid] = B22[tid] - T2[tid];
            __syncthreads();
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += T1[n*tid1 + k]*T2[k*n + tid2];
            }
            C11[tid] = temp;

            //Steps 10, 11, 12
            T1[tid] = A12[tid] - T1[tid];
            __syncthreads();
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += T1[n*tid1 + k]*B22[k*n + tid2];
            }
            C12[tid] = C22[tid] + temp;

            //Steps 13, 14, 15, 16
            __syncthreads();
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += A11[n*tid1 + k]*B11[k*n + tid2];
            }
            __syncthreads();
            T1[tid] = temp;
            C11[tid] += T1[tid];
            C12[tid] += C11[tid];
            C11[tid] += C21[tid];

            __syncthreads();
            //Steps 17, 18, 19, 20
            temp = 0;
            T2[tid] = T2[tid] - B21[tid];
            __syncthreads();
            for (int k = 0; k < n; k++) {
                temp += A22[n*tid1 + k]*T2[k*n + tid2];
            }
            C21[tid] = C11[tid] - temp;
            C22[tid] += C11[tid];

            __syncthreads();
            //Steps 21, 22
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += A12[n*tid1 + k]*B21[k*n + tid2];
            }
            __syncthreads();
            C11[tid] = temp + T1[tid];

        }
    //}
    /* else {
        //thread idx for second half of iterations
        tid1 = threadIdx.x + blockDim.x*blockIdx.x, tid2 = threadIdx.y + blockDim.y*(blockIdx.y - n);
        tid = tid1*n + tid2;
        //Flags to synchronize between two parts
        if (tid == 0)
            flag2 = false;
        if (tid1 < n/2 && tid2 < n/2) {
            //Steps 4, 5, 6
            T1[tid] = A21[tid] + A22[tid];
            T2[tid] = B22[tid] - B11[tid];
            __syncthreads();
            if (tid == 0)
                flag2 = true;
            for (int k = 0; k < n; k++) {
                temp += T1[n*tid1 + k]*T2[k*n + tid2];
            }
            C22[tid] = temp;
        }
    }    
}
*/

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

    //Multiply matrrices
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
    //Returns 0 if correct, also shows time
    int matrix_multiply_unrolled_gpu(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t = false, bool v = false) {
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
        matrix_multiply_unrolled<NUM_UNROLL><< <gridDims, blockDims/*, NUM_UNROLL*thread_x*thread_y*sizeof(T)*/>> >(d_A, d_B, d_C, m, n, p);
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

    //Transposes the matrix so that the indexing can be swapped between row and column major
    void row_col_major_swap(T *A, size_t m, size_t n) {
        T temp = 0;

        for (int i = 0; i < m; i++) {
            for (int j = i; j < n; j++) {
                temp = A[i*n + j];
                A[i*n + j] = A[j*n + i];
                A[j*n + i] = temp;
            }
        }

    }

    /*
    //Implements cublas gemm function and calculates execution time
    int matrix_multiply_cublas_gemm(T *A, T *B, T *C, size_t m, size_t n, size_t p, bool t, bool v) {
        //Initializing arrays for GPU
        T *d_A, *d_B, *d_C;
     //   float *A_, *B_, *C_;
        cublasStatus_t status;

        //Allocating memory for the arrays on the GPU
        cudaMalloc((void **)&d_A, m*n*sizeof(*d_A));
        cudaMalloc((void **)&d_B, n*p*sizeof(*d_B));
        cudaMalloc((void **)&d_C, m*p*sizeof(*d_C));

        //Transposing matrix A and B since they are stored in row major format
        row_col_major_swap(A, m, n);
        print_matrix(A, n, m);
        row_col_major_swap(B, n, p);
        print_matrix(B, p, n);

        //Copying input matrices in the GPU
        cudaMemcpy(d_A, A, m*n*sizeof(*d_A), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n*p*sizeof(*d_B), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alp = 1, bet = 0;

        //cublas gemm for float
        if (t == true) {
            start = high_resolution_clock::now();
        }
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)p, (int)n, alpha, d_A, (int)m, d_B, (int)n, beta, d_C, (int)m);
        status_check(status);
        cudaDeviceSynchronize();
        
        if (t == true) {
            finish = high_resolution_clock::now();
            gemm_time = duration_cast<duration<double>>(finish - start).count();
            if (v == true) {
                printf("GPU gemm Time: %f sec(s)\n\n", gemm_time);
            }
        }
        
        //Copying output matrix from GPU
        cudaMemcpy(C, d_C, m*p*sizeof(*d_C), cudaMemcpyDeviceToHost);
        status_check(status);
        //Changing the matrices indexing back to row major
        row_col_major_swap(A, n, m);
        row_col_major_swap(B, p, n);
        row_col_major_swap(C, p, m);
       // C = reinterpret_cast<T *>(C_);

        status = cublasDestroy(handle);
        status_check(status);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaDeviceReset();
        return 0;

    }
    */

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
        thread_x = std::min((int)(n/2),(int)floor(sqrt(threads_per_block)));
        thread_y = std::min((int)(n/2),(int)floor(sqrt(threads_per_block)));
        block_x = (int)ceil((float)(n/2)/(float)thread_x);
        block_y = (int)(ceil((float)(n/2)/(float)thread_y));
        dim3 gridDims(block_x, block_y), blockDims(thread_x, thread_y);

        //Initializing sub-arrays for GPU
        T *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22, *T1, *T2;

        printf("Allocting memory...\n");
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

        printf("Copying matrices to GPU...\n");
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

        printf("Starting matrix multiplication...\n");
        //Matrix multiplication
        if (t == true) {
            start = high_resolution_clock::now();
        }
        //strassen_winograd_algo<< <gridDims, blockDims>> >(A11, A12, A21, A22, B11, B12, B21, B22, C11, C12, C21, C22, T1, T2, n);
        //Implementation starts
        sub_mat_gpu<< <gridDims, blockDims>> >(A11, A21, T1, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(B22, B12, T2, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(T1, T2, C21, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(A21, A22, T1, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(B12, B11, T2, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(T1, T2, C22, n/2, n/2, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(T1, A11, T1, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(B22, T2, T2, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(T1, T2, C11, n/2, n/2, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(A12, T1, T1, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(T1, B22, C12, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C22, C12, C12, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(A11, B11, T1, n/2, n/2, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, T1, C11, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, C12, C12, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, C21, C11, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(T2, B21, T2, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(A22, T2, C21, n/2, n/2, n/2);
        sub_mat_gpu<< <gridDims, blockDims>> >(C11, C21, C21, n/2);
        add_mat_gpu<< <gridDims, blockDims>> >(C11, C22, C22, n/2);
        matrix_multiply_unrolled<1><< <gridDims, blockDims>> >(A12, B21, C11, n/2, n/2, n/2);
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

        printf("Copying output to GPU...\n");
        //Copying output matrix from GPU
        for (int i = 0; i < n/2; i++) {
            cudaMemcpy(&C[i*n], &C11[i*n/2], (n/2)*sizeof(*C11), cudaMemcpyDeviceToHost);
            cudaMemcpy(&C[i*n + n/2], &C12[i*n/2], (n/2)*sizeof(*C12), cudaMemcpyDeviceToHost);
            cudaMemcpy(&C[(i + n/2)*n], &C21[i*n/2], (n/2)*sizeof(*C21), cudaMemcpyDeviceToHost);
            cudaMemcpy(&C[(i + n/2)*n + n/2], &C22[i*n/2], (n/2)*sizeof(*C22), cudaMemcpyDeviceToHost);
        }

        printf("Freeing memory...\n");
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

private:
    high_resolution_clock::time_point start, finish;
    double serial_time = 0, cpu_unrolled_time = 0, gpu_unrolled_time = 0, gemm_time = 0, str_cpu = 0, str_gpu = 0;

};

#endif