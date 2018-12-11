#pragma once
#include <cstddef>																	                                       \


float *fl_cuda_matrix_multiply(float *rh_mat, float *lh_mat,
                               const std::size_t lh_row,
                               const std::size_t lh_col,
                               const std::size_t rh_row,
                               const std::size_t rh_col);

float *fl_cuda_block_matrix_multiply(float *rh_mat, float *lh_mat,
                                     const std::size_t lh_row,
                                     const std::size_t lh_col,
                                     const std::size_t rh_row,
                                     const std::size_t rh_col);

float *fl_cublas_matrix_multiply(float *rh_mat, float *lh_mat,
                                 std::size_t lh_row, std::size_t lh_col,
                                 std::size_t rh_row, std::size_t rh_col);