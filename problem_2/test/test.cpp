#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <vector>
#include <unordered_set>
#include "../part 1 and 2/matrix.h"

TEST_CASE("Naive M*N Matrix", "[Matrix]")
{
	SECTION("Operation Test")
	{
		float test_arr[6] = { 2.4, 3.1, 4.6, 6.9, 9.9, 1.0 };
		Matrix<float> mat(test_arr, 3, 2);
		REQUIRE(mat[0][0] == 2.4f);
		REQUIRE(mat[1][1] == 6.9f);
		REQUIRE(mat[2][1] == 1.0f);
	}
	SECTION("Simple CPU Multiplication Test")
	{
		unsigned int arr_0[6] = { 1, 2, 3, 4, 5, 6 };
		unsigned int arr_1[6] = { 7, 8, 9, 10, 11, 12 };
		Matrix<unsigned int> mat_0(arr_0, 2, 3);
		Matrix<unsigned int> mat_1(arr_1, 3, 2);
		Matrix<unsigned int> final_mat = mat_0 * mat_1;
		REQUIRE(!final_mat.is_empty());
		REQUIRE(final_mat[0][0] == mat_0[0][0] * mat_1[0][0] + mat_0[0][1] * mat_1[1][0] + mat_0[0][2] * mat_1[2][0]);
		REQUIRE(final_mat[1][1] == mat_0[1][0] * mat_1[0][1] + mat_0[1][1] * mat_1[1][1] + mat_0[1][2] * mat_1[2][1]);
	}
	SECTION("Visual Random Generator Test")
	{
		Matrix<unsigned int> matrix = Matrix<unsigned int>::generate_random_matrix(10, 10);
		std::unordered_set<unsigned int> number_counter;
		for (unsigned int numb : matrix.get_vector())
		{
			number_counter.insert(numb);
		}
		REQUIRE(number_counter.size() > 1);
	}
}

TEST_CASE("CUDA MxN Multiplication", "[CUDA]")
{
	SECTION("CUDA accelerated Multiplication Test")
	{
		float arr_0[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
		float arr_1[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12 };
		Matrix<float> mat_0(arr_0, 2, 3);
		Matrix<float> mat_1(arr_1, 3, 2);
		auto normal_result = mat_0 * mat_1;
		auto cuda_result = mat_0.cuda_multiply(mat_1);
		auto is_eq = normal_result.is_equal(cuda_result);
		if (is_eq == false)
		{
			printf("Normal Result:\n");
			normal_result.print();
			printf("Cuda Result:\n");
			cuda_result.print();
		}
		REQUIRE(is_eq);
	}
	SECTION("CUDA Accelerated Multiplication Test Large")
	{

		auto rand_0 = Matrix<float>::generate_random_matrix(50, 75);
		auto rand_1 = Matrix<float>::generate_random_matrix(75, 80);
		auto normal_result = rand_0 * rand_1;
		auto cuda_result = rand_0.cuda_multiply(rand_1);
		auto is_eq = normal_result.is_equal(cuda_result);
		if (is_eq == false)
		{
			printf("Normal Result:\n");
			normal_result.print();
			printf("Cuda Result:\n");
			cuda_result.print();
		}
		REQUIRE(is_eq);
	}
}

TEST_CASE("cuBlas MxN multiplication test", "[cuBlas]")
{
	SECTION("cuBLAS accelerated Multiplication Test")
	{
		float arr_0[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
		float arr_1[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12 };
		Matrix<float> mat_0(arr_0, 2, 3);
		Matrix<float> mat_1(arr_1, 3, 2);
		auto normal_result = mat_0 * mat_1;
		auto cublas_result = mat_0.cublas_multiply(mat_1);
		auto is_equal = normal_result.is_equal(cublas_result);
		if (is_equal == false)
		{
			printf("Normal Result\n");
			normal_result.print();
			printf("cuBlas Result\n");
			cublas_result.print();
		}
		REQUIRE(is_equal);
	}
	SECTION("cuBLAS Accelerated Multiplication Test Large")
	{

		auto rand_0 = Matrix<float>::generate_random_matrix(50, 75);
		auto rand_1 = Matrix<float>::generate_random_matrix(75, 50);
		auto normal_result = rand_0 * rand_1;
		auto cuda_result = rand_0.cublas_multiply(rand_1);
		auto is_eq = normal_result.is_equal(cuda_result);
		if (is_eq == false)
		{
			printf("Normal Result:\n");
			normal_result.print();
			printf("cuBlas Result:\n");
			cuda_result.print();
		}
		REQUIRE(is_eq);
	}
}

TEST_CASE("Cuda Block Muliplication Test")
{
	SECTION("128 Block Test %32 = 0") {
		auto rand_0 = Matrix<float>::generate_random_matrix(128, 128);
		auto rand_1 = Matrix<float>::generate_random_matrix(128, 128);
		auto normal_result = rand_0 * rand_1;
		auto cuda_result = rand_0.cuda_block_multiply(rand_1);
		auto is_eq = normal_result.is_equal(cuda_result);
		if (is_eq == false)
		{
			printf("Normal Result:\n");
			normal_result.print();
			printf("Cuda Result:\n");
			cuda_result.print();
		}
		REQUIRE(is_eq);
	}
}