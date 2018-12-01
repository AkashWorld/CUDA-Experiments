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
        final_mat.print_dec();
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
    SECTION("CPU Block Multiplication Test, NxN")
    {
        auto mat_1 = Matrix<unsigned int>::generate_random_matrix(10, 10);
        auto mat_2 = Matrix<unsigned int>::generate_random_matrix(10, 10);
        auto normal_result = mat_1 * mat_2;
        auto result = mat_1.block_multiply(mat_2);
        REQUIRE(!result.is_empty());
        REQUIRE(result.get_row_size() == 10);
        REQUIRE(result.get_col_size() == 10);
        debug_logln("Printing normal result%s\n", "");
        normal_result.print_dec();
        debug_logln("Printing blocked result%s\n", "");
        result.print_dec();
        REQUIRE(normal_result.is_equal(result));
    }
}