#include "mmm_unrolled.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <chrono>
#include <type_traits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef float testType;

//Compare matrices
template <typename T>
void compare_matrices(T *A, T *B, size_t m, size_t n) {
    float temp = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float prev_temp = temp;
            temp += (float)(A[j + n*i] - B[j + n*i]) * (A[j + n*i] - B[j + n*i]);
            if (temp - prev_temp > 0.1) {
                std::cout << "Matrix 1: " << A[j + n*i] << ", " << i << ", " << j << ", Matrix 2: " << B[j + n*i] << "\n"; 
            }
            /*
            if (fabs((float)(A[j + n*i] - B[j + n*i])) > 0.1) {
                printf("Matrices not equal!\n");
                return;
            }
            */
        }
    }
    printf("MSE: %f\n", temp/(float)(m*n));
    //printf("Matrices equal.\n");
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
    double serial = 0, cpu_ur = 0, gpu_ur = 0, gemmtime = 0, gpu_str = 0;

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
    testType *A, *B, *C0, *C1, *C2, *C3, *C4;
    A = (testType *)malloc(m*n*sizeof(*A));
    B = (testType *)malloc(n*p*sizeof(*B));
    C0 = (testType *)malloc(m*p*sizeof(*C0));
    C1 = (testType *)malloc(m*p*sizeof(*C1));
    C2 = (testType *)malloc(m*p*sizeof(*C2));
    C3 = (testType *)malloc(m*p*sizeof(*C3));
    C4 = (testType *)malloc(m*p*sizeof(*C4));
  //  print_matrix(A, m, n);
  //  print_matrix(B, n, p);

    //Defining matrix multiplication object
    mmm_unrolled <testType> gemm;

    printf("[Matrix Multiplication]\nMatrix A: %dx%d, Matrix B: %dx%d\n", (int)m, (int)n, (int)n, (int)p);
/*
    if (std::is_floating_point<testType>::value) {
        for (int i = 0; i < iter; i++) {
            gemm.matrix_multiply_cublas_gemm(A, B, C0, m, n, p, true, false);
            gemmtime += gemm.getExecTime("gemm");
        }
        printf("[GPU gemm] matrix multiplication time for %d iterations: %f secs\n", iter, gemmtime);
        print_matrix(C0, m, p);
    }
    */
  //  print_matrix(B, n, p);

    for (int i = 0; i < iter; i++) {

        srand(time(NULL));
        //Generating random input matrices
        gen_rand_mat(A, m, n);
        gen_rand_mat(B, n, p);

        gemm.matrix_multiply_normal(A, B, C1, m, n, p, true, false);
        serial += gemm.getExecTime("normal");
   // }
    /*
    if (std::is_floating_point<testType>::value) {
        printf("Checking output with gemm...");
        compare_matrices(C0, C1, m, p);
    }
    */
    //Print output matrix and sets it to 0
   // print_matrix(C1, m, p);

 //   for (int i = 0; i < iter; i++) {
        gemm.matrix_multiply_unrolled_cpu(A, B, C2, m, n, p, true, false);
        cpu_ur += gemm.getExecTime("cpu_unrolled");
  //  }
    /*
    if (std::is_floating_point<testType>::value) {
        printf("Checking output with gemm...");
        compare_matrices(C0, C2, m, p);
    }
    */
  //  printf("Checking output with serial...");
   // compare_matrices(C1, C2, m, p);
    //Print output matrix and sets it to 0
   // print_matrix(C2, m, p);

  //  for (int i = 0; i < iter; i++) {
        gemm.matrix_multiply_unrolled_gpu(A, B, C3, m, n, p, true, false);
        gpu_ur += gemm.getExecTime("gpu_unrolled");
    //  }
    //  printf("Checking output with serial...");
    // compare_matrices(C1, C3, m, p);
        /*
        if (std::is_floating_point<testType>::value) {
            printf("Checking output with gemm...");
            compare_matrices(C0, C3, m, p);
        }
        */


        if (m == n && n == p && m == p) {
    //     for (int i = 0; i < iter; i++) {
                gemm.strassens_algo_gpu(A, B, C4, m, true, false);
                gpu_str+= gemm.getExecTime("strassens_gpu");
        //   }
        // printf("Checking output with serial...");
            //compare_matrices(C1, C4, m, p);
           // print_matrix(C3, m, p);
        }
        
    }
    
    compare_matrices(C1, C2, m, p);
    compare_matrices(C1, C3, m, p);
    compare_matrices(C1, C4, m, p);
   // print_matrix(C3, m, p);
   // printf("Speed-up due to cublas gemm: %f\n", serial/gemmtime);


    printf("[Serial] matrix multiplication time for %d iterations: %f secs\n", iter, serial);
    printf("[CPU unrolled] matrix multiplication time for %d iterations: %f secs\n", iter, cpu_ur);
    printf("[GPU unrolled] matrix multiplication time for %d iterations: %f secs\n", iter, gpu_ur);
    printf("[GPU Strassens Winograd Variant] matrix multiplication time for %d iterations: %f secs\n", iter, gpu_str);
    printf("Speed-up due to CPU unrolling: %f\n", serial/cpu_ur);
    printf("Speed-up due to GPU unrolling: %f\n", serial/gpu_ur);
    printf("Speed-up due to GPU Strassens: %f\n", serial/gpu_str);


    //Freeing allocated memory
    free(A);
    free(B);
    free(C0);
    free(C1);
    free(C2);
    free(C3);
    free(C4);

    //std::getchar();
    return 0;

}