#pragma once
#include <vector>

template <typename T>
class Matrix
{
  public:
    std::vector<double> data_arr;
    const std::size_t row;
    const std::size_t col;
    Matrix(std::size_t row, std::size_t col)
    {
        data_arr.resize(row * col);
        this->row = row;
        this->col = col;
    }
    bool is_empty()
    {
        return row == 0 && col == 0;
    }
    Matrix operator*(Matrix &rh_matrix)
    {
        if (this->row != rh_matrix->col || this->col != rh_matrix->row ||
            this->is_empty() || rh_matrix.is_empty())
        {
            return Matrix(0, 0);
        }
        const std::size_t row = this->row;
        const std::size_t col = rh_matrix->col;
        Matrix ret_mat(row, col);
        /*TODO: Mult*/
        return ret_mat;
    }
};