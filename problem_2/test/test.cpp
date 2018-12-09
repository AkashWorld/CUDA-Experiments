#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <vector>
#include <unordered_set>
#include "../src/matrix.h"

TEST_CASE("M*N Matrix", "[Matrix]")
{
    SECTION("Operation Test")
    {
        float test_arr[6] = {2.4, 3.1, 4.6, 6.9, 9.9, 1.0};
        Matrix<float> mat(test_arr, 3, 2);
        REQUIRE(mat[0][0] == 2.4f);
        REQUIRE(mat[1][1] == 6.9f);
        REQUIRE(mat[2][1] == 1.0f);
    }
    SECTION("Simple CPU Multiplication Test")
    {
        unsigned int arr_0[6] = {1, 2, 3, 4, 5, 6};
        unsigned int arr_1[6] = {7, 8, 9, 10, 11, 12};
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
	/*
    SECTION("CPU Block Multiplication Test, NxN")
    {
        auto mat_1 = Matrix<unsigned int>::generate_random_matrix(20, 20);
        auto mat_2 = Matrix<unsigned int>::generate_random_matrix(20, 20);
        auto normal_result = mat_1 * mat_2;
        auto result = mat_1.block_multiply(mat_2);
        REQUIRE(!result.is_empty());
        REQUIRE(result.get_row_size() == 20);
        REQUIRE(result.get_col_size() == 20);
        REQUIRE(normal_result.is_equal(result));
    }*/
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
	SECTION("CUDA Accelerated Multiplication Test Med")
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
	/*
	SECTION("cuBLAS accelerated Multiplication Test")
	{
		float arr_0[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
		float arr_1[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12 };
		Matrix<float> mat_0(arr_0, 2, 3);
		Matrix<float> mat_1(arr_1, 3, 2);
		auto normal_result = mat_0 * mat_1;
		auto cublas_result = mat_0.cublas_multiply(mat_1);
		REQUIRE(normal_result.is_equal(cublas_result));
	}
	*/
}