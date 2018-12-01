#include "matrix.h"
#include <logger.h>
#include <chrono>

void naive_matrix_multiplication(const std::size_t size, const unsigned int iteration)
{
    fprintf(stdout, YEL(BOLD("[NAIVE]") " Matrix multiplication of " YEL("%lux%lu") " Matrices for " YEL("%u") " iterations\n"), size, size, iteration);
    fprintf(stdout, BOLD(RED("Calculating...")));
    fflush(stdout);
    auto lh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
    auto rh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
    Matrix<unsigned int> result(size, size);
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < iteration; ++i)
    {
        result = lh_mat * rh_mat;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    long duration = elapsed_seconds.count();
    fprintf(stdout,"\r" BOLD(YEL("[NAIVE]")) " Elapsed time" YEL("%ld seconds\n"), duration);
}

int main(int argc, char **argv)
{
    fprintf(stdout, (BOLD(GRN("Matrix Multiplication\n"))));
    naive_matrix_multiplication(100, 1000);
    naive_matrix_multiplication(100, 2000);
    naive_matrix_multiplication(100, 3000);
    return 0;
}