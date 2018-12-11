#pragma once
#include "matrix.cuh"
#include <cassert>
#include <limits>
#include <logger.h>
#include <random>
#include <vector>

template <typename T> class Matrix {
private:
	std::vector<T> data_arr;
	std::size_t row;
	std::size_t col;

public:
	Matrix(std::size_t row, std::size_t col) {
		data_arr.resize(row * col);
		this->row = row;
		this->col = col;
	}
	Matrix(std::vector<T> &data_arr, std::size_t row, std::size_t col) {
		this->data_arr.assign(data_arr.begin(), data_arr.end());
		this->row = row;
		this->col = col;
	}
	Matrix(T arr[], std::size_t row, std::size_t col) {
		this->data_arr.assign(arr, arr + (row * col));
		this->row = row;
		this->col = col;
	}
	inline std::vector<T> &get_vector() { return data_arr; }
	inline std::size_t get_row_size() { return row; }
	inline std::size_t get_col_size() { return col; }
	inline bool is_empty() { return row == 0 || col == 0; }
	inline std::size_t element_count() { return this->row * this->col; }
	T *operator[](std::size_t index) { return data_arr.data() + (col * index); }
	/*Standard, least efficient*/
	Matrix operator*(Matrix &rh_matrix) {
		if (this->col != rh_matrix.row || this->is_empty() ||
			rh_matrix.is_empty()) {
			debug_logln("Invalid matrix input.");
			return Matrix(0, 0);
		}
		const std::size_t row = this->row;
		const std::size_t col = rh_matrix.col;
		Matrix ret_mat(row, col);
		for (std::size_t i = 0; i < row; ++i) {
			for (std::size_t j = 0; j < col; ++j) {
				ret_mat[i][j] = 0;
				for (std::size_t k = 0; k < this->col; ++k) {
					ret_mat[i][j] += (*this)[i][k] * rh_matrix[k][j];
				}
			}
		}
		return ret_mat;
	}
	/*Tiling*/
	Matrix block_multiply(Matrix &rh_matrix) {
		assert(this->row % 10 == 0);
		assert(this->col % 10 == 0);
		assert(rh_matrix.row % 10 == 0);
		assert(rh_matrix.col % 10 == 0);
		if (this->row != rh_matrix.col || this->col != rh_matrix.row ||
			this->is_empty() || rh_matrix.is_empty()) {
			debug_logln("Invalid matrix input.");
			return Matrix(0, 0);
		}
		const std::size_t row = this->row;
		const std::size_t col = rh_matrix.col;
		constexpr std::size_t BLOCK_SIZE = 10;
		Matrix ret_matrix(row, col);
		for (std::size_t i = 0; i < row; i += BLOCK_SIZE) {
			for (std::size_t j = 0; j < col; j += BLOCK_SIZE) {
				for (std::size_t ii = i; ii < i + BLOCK_SIZE; ++ii) {
					for (std::size_t jj = j; jj < j + BLOCK_SIZE; ++jj) {
						ret_matrix[ii][jj] = 0;
						for (std::size_t k = 0; k < this->col; k += BLOCK_SIZE) {
							for (std::size_t kk = k; kk < k + BLOCK_SIZE; ++kk) {
								ret_matrix[ii][jj] += (*this)[ii][kk] * rh_matrix[kk][jj];
							}
						}
					}
				}
			}
		}
		return ret_matrix;
	}
	Matrix<T> cuda_multiply(Matrix<T> &rh_matrix) {
		/*Requires type specific implementation*/
		return Matrix<T>(0, 0);
	}
	Matrix<T> cuda_block_multiply(Matrix<T> &rh_matrix)
	{
		return Matrix<T>(0, 0);
	}
	Matrix<T> cublas_multiply(Matrix<T> &rh_matrix) {
		/*Requires type specific implementation*/
		return Matrix<T>(0, 0);
	}
	bool is_equal(Matrix &other) {
		if (this->row != other.row || this->col != other.col) {
			return false;
		}
		for (std::size_t i = 0; i < this->row; ++i) {
			for (std::size_t j = 0; j < this->col; ++j) {
				if ((*this)[i][j] != other[i][j]) {
					return false;
				}
			}
		}
		return true;
	}
	void print() {
		err_logln("Please implement a type specific print function.%s", "");
	}
	static Matrix generate_random_matrix(std::size_t row, std::size_t col) {
		return Matrix(row, col);
	}
};

template <>
Matrix<float> Matrix<float>::cuda_multiply(Matrix<float> &rh_matrix) {
	if (this->col != rh_matrix.row || this->is_empty() || rh_matrix.is_empty()) {
		debug_logln("Invalid matrix input (both matrices are empty!).");
		return Matrix(0, 0);
	}
	const std::size_t row = this->row;
	const std::size_t col = rh_matrix.col;
	float *result = fl_cuda_matrix_multiply(
		this->get_vector().data(), rh_matrix.get_vector().data(), this->row,
		this->col, rh_matrix.row, rh_matrix.col);
	if (result == NULL) {
		err_logln("Error performing CUDA multiplication of two matrices!");
		return Matrix(0, 0);
	}
	return Matrix(result, row, col);
}

template <>
Matrix<float> Matrix<float>::cuda_block_multiply(Matrix<float> &rh_matrix) {
	if (this->col != rh_matrix.row || this->is_empty() || rh_matrix.is_empty()) {
		debug_logln("Invalid matrix input (both matrices are empty!).");
		return Matrix(0, 0);
	}
	const std::size_t row = this->row;
	const std::size_t col = rh_matrix.col;
	float *result = fl_cuda_block_matrix_multiply(
		this->get_vector().data(), rh_matrix.get_vector().data(), this->row,
		this->col, rh_matrix.row, rh_matrix.col);
	if (result == NULL) {
		err_logln("Error performing CUDA multiplication of two matrices!");
		return Matrix(0, 0);
	}
	return Matrix(result, row, col);
}

template <>
Matrix<float> Matrix<float>::cublas_multiply(Matrix<float> &rh_matrix) {
	if (this->col != rh_matrix.row || this->is_empty() || rh_matrix.is_empty()) {
		debug_logln("Invalid matrix input.");
		return Matrix(0, 0);
	}
	const std::size_t row = this->row;
	const std::size_t col = rh_matrix.col;
	std::size_t n = this->col * this->row;
	float *result = fl_cublas_matrix_multiply(
		this->get_vector().data(), rh_matrix.get_vector().data(), this->row,
		this->col, rh_matrix.row, rh_matrix.col);
	if (result == NULL) {
		err_logln("Error performing cuBLAS multiplication of two matrices!");
		return Matrix(0, 0);
	}
	return Matrix(result, row, col);
}

template <>
Matrix<unsigned int>
Matrix<unsigned int>::generate_random_matrix(std::size_t row, std::size_t col) {
	Matrix<unsigned int> matrix(row, col);
	std::random_device rd;
	for (std::size_t i = 0; i < matrix.get_row_size(); ++i) {
		for (std::size_t j = 0; j < matrix.get_col_size(); ++j) {
			matrix[i][j] = rd();
		}
	}
	return matrix;
}

template <>
Matrix<float> Matrix<float>::generate_random_matrix(std::size_t row,
	std::size_t col) {
	Matrix<float> matrix(row, col);
	std::random_device rd;
	for (std::size_t i = 0; i < matrix.get_row_size(); ++i) {
		for (std::size_t j = 0; j < matrix.get_col_size(); ++j) {
			float new_number =
				static_cast<float>(rd()) /
				static_cast<float>(RAND_MAX / (std::numeric_limits<float>::max() -
					std::numeric_limits<float>::min()));
			matrix[i][j] = new_number;
		}
	}
	return matrix;
}

template <> void Matrix<unsigned int>::print() {
	for (std::size_t i = 0; i < row; ++i) {
		printf(BOLD("[%s"), "");
		for (std::size_t j = 0; j < col; ++j) {
			printf(BOLD(" %u"), (*this)[i][j]);
			if (j != col - 1) {
				printf(BOLD(",%s"), "");
			}
		}
		printf(BOLD("%s]\n"), "");
	}
}

template <> void Matrix<float>::print() {
	for (std::size_t i = 0; i < row; ++i) {
		printf(BOLD("[%s"), "");
		for (std::size_t j = 0; j < col; ++j) {
			printf(BOLD(" %f"), (*this)[i][j]);
			if (j != col - 1) {
				printf(BOLD(",%s"), "");
			}
		}
		printf(BOLD("%s]\n"), "");
	}
}