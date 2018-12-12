#include "matrix.h"
#include <chrono>
#include <logger.h>

void cuda_matrix_multiplication(const std::size_t size,
                                const unsigned int iteration) {
  fprintf(stdout,
          GRN(BOLD("[CUDA]") " Matrix multiplication of " YEL(
              "%lux%lu") " Matrices for " YEL("%u") " iterations...\n"),
          size, size, iteration);
  fprintf(stdout, BOLD(RED("Calculating...")));
  fflush(stdout);
  auto lh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
  auto rh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
  Matrix<unsigned int> result(size, size);
  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iteration; ++i) {
    result = lh_mat.cuda_multiply(rh_mat);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto delta = end - start;
  auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
  fprintf(stdout,
          "\r"
          "elapsed time " YEL("%lld ns\n"),
          dur_ns.count());
}

void cuda_block_matrix_multiplication(const std::size_t size,
	const unsigned int iteration) {
	fprintf(stdout,
		GRN(BOLD("[CUDA_BLOCK]") " Matrix multiplication of " YEL(
			"%lux%lu") " Matrices for " YEL("%u") " iterations...\n"),
		size, size, iteration);
	fprintf(stdout, BOLD(RED("Calculating...")));
	fflush(stdout);
	auto lh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
	auto rh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
	Matrix<unsigned int> result(size, size);
	auto start = std::chrono::high_resolution_clock::now();
	for (unsigned int i = 0; i < iteration; ++i) {
		result = lh_mat.cuda_block_multiply(rh_mat);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto delta = end - start;
	auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
	fprintf(stdout,
		"\r"
		"elapsed time " YEL("%lld ns\n"),
		dur_ns.count());
}

void cublas_matrix_multiplication(const std::size_t size,
                                  const unsigned int iteration) {
  fprintf(stdout,
          CYN(BOLD("[CUBLAS]") " Matrix multiplication of " YEL(
              "%lux%lu") " Matrices for " YEL("%u") " iterations...\n"),
          size, size, iteration);
  fprintf(stdout, BOLD(RED("Calculating...")));
  fflush(stdout);
  auto lh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
  auto rh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
  Matrix<unsigned int> result(size, size);
  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iteration; ++i) {
    result = lh_mat.cublas_multiply(rh_mat);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto delta = end - start;
  auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
  fprintf(stdout,
          "\r"
          "elapsed time " YEL("%lld ns\n"),
          dur_ns.count());
}

void naive_matrix_multiplication(const std::size_t size,
                                 const unsigned int iteration) {
  fprintf(stdout,
          YEL(BOLD("[NAIVE]") " Matrix multiplication of " YEL(
              "%lux%lu") " Matrices for " YEL("%u") " iterations...\n"),
          size, size, iteration);
  fprintf(stdout, BOLD(RED("Calculating...")));
  fflush(stdout);
  auto lh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
  auto rh_mat = Matrix<unsigned int>::generate_random_matrix(size, size);
  Matrix<unsigned int> result(size, size);
  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iteration; ++i) {
    result = lh_mat * rh_mat;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto delta = end - start;
  auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
  fprintf(stdout,
          "\r"
          "elapsed time " YEL("%lld ns\n"),
          dur_ns.count());
}

int main(int argc, char **argv) {
  fprintf(stdout, (BOLD(GRN("Matrix Multiplication\n"))));
  cublas_matrix_multiplication(256, 500);
  cublas_matrix_multiplication(256, 1000);
  cublas_matrix_multiplication(256, 1500);
  cuda_matrix_multiplication(256, 500);
  cuda_matrix_multiplication(256, 1000);
  cuda_matrix_multiplication(256, 1500);
  cuda_block_matrix_multiplication(256, 500);
  cuda_block_matrix_multiplication(256, 1000);
  cuda_block_matrix_multiplication(256, 1500);
  naive_matrix_multiplication(256, 500);
  naive_matrix_multiplication(256, 1000);
  naive_matrix_multiplication(256, 1500);
  return 0;
}