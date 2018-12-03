#include "mmm_unrolled.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef float testType;

//Prints matrix
template <typename T>
void print_matrix(T *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        printf("|| ");
        for (int j = 0; j < n; j++) {
            std::cout << std::setprecision(3) << A[j + n*i] << "\t";
        }
        printf(" ||\n");
    }
    printf("\n");
}

//Compare matrices
template <typename T>
void compare_matrices(T *A, T *B, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs((float)(A[j + n*i] - B[j + n*i])) > 0.1) {
                printf("Matrices not equal!\n");
                return;
            }
        }
    }
    printf("Matrices equal.\n");
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

int main(int argc, char* argv[]) {

    size_t m = 100, n = 100, p = 100;
    int iter = 10;
    double serial = 0, cpu_ur = 0, gpu_ur = 0;

    //For different types of argument inputs
    //Can provide all dimensions for an MxN and an NxP matrix with [./main 100 200 300]
    //For square matrices [./main 100]
    //If number if iterations are to be provided then [./main [dims] [iters] ] Eg: [./main 100 10] or [./main 100 200 300 10]
    if (argc == 4) {
        m = ((size_t)atoi(argv[1]));
        n = ((size_t)atoi(argv[2]));
        p = ((size_t)atoi(argv[3]));
    } else if (argc == 5) {
        m = ((size_t)atoi(argv[1]));
        n = ((size_t)atoi(argv[2]));
        p = ((size_t)atoi(argv[3]));
        iter = atoi(argv[4]);
    } else if (argc == 3) {
        m = ((size_t)atoi(argv[1]));
        n = m;
        p = n;
        iter = atoi(argv[2]);
    } else if (argc == 2) {
        m = ((size_t)atoi(argv[1]));
        n = m;
        p = n;   
    }

    //Setting random input matrices
    testType *A, *B, *C1, *C2, *C3;
    A = (testType *)malloc(m*n*sizeof(*A));
    B = (testType *)malloc(n*p*sizeof(*B));
    C1 = (testType *)malloc(m*p*sizeof(*C1));
    C2 = (testType *)malloc(m*p*sizeof(*C2));
    C3 = (testType *)malloc(m*p*sizeof(*C3));
    memset(A, 0, m*n*sizeof(*A));
    memset(B, 0, n*p*sizeof(*B));
    memset(C1, 0, m*p*sizeof(*C1));
    memset(C2, 0, m*p*sizeof(*C2));
    memset(C3, 0, m*p*sizeof(*C3));

    srand(time(NULL));
    //Generating random input matrices
    gen_rand_mat(A, m, n);
    gen_rand_mat(B, n, p);
   // print_matrix(A, m, n);
   // print_matrix(B, n, p);

    //Defining matrix multiplication object
    mmm_unrolled <testType> gemm;

    printf("[Matrix Multiplication]\nMatrix A: %dx%d, Matrix B: %dx%d\n", (int)m, (int)n, (int)n, (int)p);

    for (int i = 0; i < iter; i++) {
        gemm.matrix_multiply_normal(A, B, C1, m, n, p, true, false);
        serial += gemm.getExecTime("normal");
    }
    printf("[Serial] matrix multiplication time for %d iterations: %f secs\n", iter, serial);
    //Print output matrix and sets it to 0
    //print_matrix(C1, m, p);

    for (int i = 0; i < iter; i++) {
        gemm.matrix_multiply_unrolled_cpu(A, B, C2, m, n, p, true, false);
        cpu_ur += gemm.getExecTime("cpu_unrolled");
    }
    printf("[CPU unrolled] matrix multiplication time for %d iterations: %f secs\n", iter, cpu_ur);
    printf("Checking matrices...");
    compare_matrices(C1, C2, m, p);
    //Print output matrix and sets it to 0
    //print_matrix(C2, m, p);

    for (int i = 0; i < iter; i++) {
        gemm.matrix_multiply_unrolled_gpu(A, B, C3, m, n, p, true, false);
        gpu_ur += gemm.getExecTime("gpu_unrolled");
    }
    printf("[GPU unrolled] matrix multiplication time for %d iterations: %f secs\n", iter, gpu_ur);
    printf("Checking matrices...");
    compare_matrices(C1, C3, m, p);
    //print_matrix(C3, m, p);
    printf("Speed-up due to CPU unrolling: %f\n", serial/cpu_ur);
    printf("Speed-up due to GPU unrolling: %f\n", serial/gpu_ur);

    //Freeing allocated memory
    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);

    //std::getchar();
    return 0;

}