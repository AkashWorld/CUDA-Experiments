#include "mmm_unrolled.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <chrono>
#include <vector>
#include <type_traits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef float testType;

int main(int argc, char* argv[]) {

    size_t m = 100, n = 100, p = 100;
    int iter = 10;
    double serial = 0, cpu_ur = 0, gemmtime = 0, gpu_str = 0;
    std::vector<double> gpu_ur(8, 0);
    std::vector<double> gpu_ur_od(8, 0);

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

        gemm.matrix_multiply_unrolled_cpu(A, B, C2, m, n, p, true, false);
      //  compare_matrices(C1, C2, m, p);
        cpu_ur += gemm.getExecTime("cpu_unrolled");
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

        //Two dinemstion unroll
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<1>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 1 loop unroll\n";
        }
        gpu_ur[0] += gemm.getExecTime("gpu_unrolled");
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<2>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 2 loop unroll\n";
        }
        gpu_ur[1] += gemm.getExecTime("gpu_unrolled");
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<4>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 4 loop unroll\n";
        }
        gpu_ur[2] += gemm.getExecTime("gpu_unrolled");
        
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<8>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 8 loop unroll\n";
        }
        gpu_ur[3] += gemm.getExecTime("gpu_unrolled");
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<16>(A, B, C3, m, n, p, true, false);
      //  print_matrix(C3, m, p);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 16 loop unroll\n";
        }
        gpu_ur[4] += gemm.getExecTime("gpu_unrolled");
        
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<32>(A, B, C3, m, n, p, true, false);
     //   print_matrix(C3, m, p);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 32 loop unroll\n";
        }
        gpu_ur[5] += gemm.getExecTime("gpu_unrolled");
        
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<64>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 64 loop unroll\n";
        }
        gpu_ur[6] += gemm.getExecTime("gpu_unrolled");

        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu<128>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 128 loop unroll\n";
        }
        gpu_ur[7] += gemm.getExecTime("gpu_unrolled");

        //Single dimension unroll
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<1>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 1 loop unroll\n";
        }
        gpu_ur_od[0] += gemm.getExecTime("gpu_unrolled");
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<2>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 2 loop unroll\n";
        }
        gpu_ur_od[1] += gemm.getExecTime("gpu_unrolled");
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<4>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 4 loop unroll\n";
        }
        gpu_ur_od[2] += gemm.getExecTime("gpu_unrolled");
        
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<8>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 8 loop unroll\n";
        }
        gpu_ur_od[3] += gemm.getExecTime("gpu_unrolled");
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<16>(A, B, C3, m, n, p, true, false);
      //  print_matrix(C3, m, p);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 16 loop unroll\n";
        }
        gpu_ur_od[4] += gemm.getExecTime("gpu_unrolled");
        
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<32>(A, B, C3, m, n, p, true, false);
     //   print_matrix(C3, m, p);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 32 loop unroll\n";
        }
        gpu_ur_od[5] += gemm.getExecTime("gpu_unrolled");
        
        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<64>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 64 loop unroll\n";
        }
        gpu_ur_od[6] += gemm.getExecTime("gpu_unrolled");

        memset(C3, 0, m*p*sizeof(*C3));
        gemm.matrix_multiply_unrolled_gpu_one_dim<128>(A, B, C3, m, n, p, true, false);
        if (compare_matrices(C1, C3, m, p) == -1) {
            std::cerr << "Err with 128 loop unroll\n";
        }
        gpu_ur_od[7] += gemm.getExecTime("gpu_unrolled");
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
            gemm.strassens_algo_gpu<1>(A, B, C4, m, true, false);
        //    compare_matrices(C1, C4, m, p);
            gpu_str += gemm.getExecTime("strassens_gpu");
            // print_matrix(C3, m, p);
        }
        
    }
    
   // print_matrix(C3, m, p);
   // printf("Speed-up due to cublas gemm: %f\n", serial/gemmtime);


    printf("[Serial] matrix multiplication time for %d iterations: %f secs\n", iter, serial);
    printf("[CPU unrolled] matrix multiplication time for %d iterations: %f secs\n", iter, cpu_ur);
    printf("Two dimension unroll:\n");
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 1, gpu_ur[0]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 2, gpu_ur[1]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 4, gpu_ur[2]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 8, gpu_ur[3]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 16, gpu_ur[4]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 32, gpu_ur[5]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 64, gpu_ur[6]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 128, gpu_ur[7]);
    printf("One dimension unroll:\n");
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 1, gpu_ur_od[0]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 2, gpu_ur_od[1]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 4, gpu_ur_od[2]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 8, gpu_ur_od[3]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 16, gpu_ur_od[4]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 32, gpu_ur_od[5]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 64, gpu_ur_od[6]);
    printf("[GPU unrolled] matrix multiplication time for %d iterations and %d unrolls: %f secs\n", iter, 128, gpu_ur_od[7]);


    printf("[GPU Strassens Winograd Variant] matrix multiplication time for %d iterations: %f secs\n", iter, gpu_str);
    printf("Speed-up due to CPU unrolling: %f\n", serial/cpu_ur);
    printf("Two dimension unroll:\n");
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 1, serial/gpu_ur[0]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 2, serial/gpu_ur[1]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 4, serial/gpu_ur[2]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 8, serial/gpu_ur[3]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 16, serial/gpu_ur[4]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 32, serial/gpu_ur[5]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 64, serial/gpu_ur[6]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 128, serial/gpu_ur[7]);
    printf("One dimension unroll:\n");
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 1, serial/gpu_ur_od[0]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 2, serial/gpu_ur_od[1]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 4, serial/gpu_ur_od[2]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 8, serial/gpu_ur_od[3]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 16, serial/gpu_ur_od[4]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 32, serial/gpu_ur_od[5]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 64, serial/gpu_ur_od[6]);
    printf("Speed-up due to GPU unrolling with %d unrolls: %f\n", 128, serial/gpu_ur_od[7]);
    printf("Speed-up for GPU Strassen Winograd Algorithm: %f\n", serial/gpu_str);


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