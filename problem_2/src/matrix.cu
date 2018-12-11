#include "matrix.cuh"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <logger.h>
#include <stdlib.h>
#define CUDA_IDX2C(i, j, col_dim) (((i) * (col_dim)) + (j))
#define CUBLAS_IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static const char *_cudaGetErrorEnum(cublasStatus_t error);

#define CHECK_ERR(x)                                                           \
  if (x != cudaSuccess) {                                                      \
    err_logln("Cuda error caught! Error: ", cudaGetErrorString(x));            \
    goto free;                                                                 \
  }		

#define BLOCK_WIDTH 32

__global__ void blocked_mat_mul(float *lh_mat, float *rh_mat, float *res_mat,
	int lh_row, int lh_col,
	int rh_row, int rh_col)
{
	float res_val = 0;
	int row = (blockIdx.y*BLOCK_WIDTH) + threadIdx.y;
	int col = (blockIdx.x*BLOCK_WIDTH) + threadIdx.x;
	__shared__ float lh_block[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float rh_block[BLOCK_WIDTH][BLOCK_WIDTH];
	for (int i = 0; i < (BLOCK_WIDTH + lh_col - 1) / BLOCK_WIDTH; ++i)
	{
		if (i*BLOCK_WIDTH + threadIdx.x < lh_col && row < lh_row)
		{
			lh_block[threadIdx.y][threadIdx.x] = lh_mat[lh_col*row + i * BLOCK_WIDTH + threadIdx.x];
		}
		else
		{
			lh_block[threadIdx.y][threadIdx.x] = 0.0f;
		}
		if (i*BLOCK_WIDTH + threadIdx.y < rh_row && col < rh_col)
		{
			rh_block[threadIdx.y][threadIdx.x] = rh_mat[rh_col*(i*BLOCK_WIDTH + threadIdx.y) + col];
		}
		else
		{
			rh_block[threadIdx.y][threadIdx.x] = 0.0;
		}
	}

	__syncthreads();

	for (int i = 0; i < BLOCK_WIDTH; ++i)
	{
		res_val += lh_block[threadIdx.y][i] * rh_block[i][threadIdx.x];
	}

	__syncthreads();

	if (row < lh_row && col < rh_col)
	{
		res_mat[(blockIdx.y*blockDim.y + threadIdx.y)*rh_col + (blockIdx.x*blockDim.x + threadIdx.x)] = res_val;
	}
}

float *fl_cuda_block_matrix_multiply(float *lh_mat, float *rh_mat,
	const std::size_t lh_row,
	const std::size_t lh_col,
	const std::size_t rh_row,
	const std::size_t rh_col)
{
	const std::size_t n = rh_row * lh_col;
	const std::size_t size = n * sizeof(float);
	float *result_matrix = (float *)malloc(size);
	if (result_matrix == NULL) {
		err_logln("Error allocating host memory!%s", "");
		return NULL;
	}
	float *dev_rh_mat, *dev_lh_mat, *dev_res_mat;
	cudaError_t error_stat;
	if ((error_stat = cudaMalloc(&dev_rh_mat, rh_row * rh_col * sizeof(float))) !=
		cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s",
			cudaGetErrorString(error_stat));
		free(result_matrix);
		return NULL;
	}
	if ((error_stat = cudaMalloc(&dev_lh_mat, lh_row * lh_col * sizeof(float))) !=
		cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s",
			cudaGetErrorString(error_stat));
		free(result_matrix);
		cudaFree(dev_rh_mat);
		return NULL;
	}
	if ((error_stat = cudaMalloc(&dev_res_mat, size)) != cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s",
			cudaGetErrorString(error_stat));
		free(result_matrix);
		cudaFree(dev_rh_mat);
		cudaFree(dev_lh_mat);
		return NULL;
	}
	error_stat = cudaMemcpy(dev_rh_mat, rh_mat, rh_row * rh_col * sizeof(float),
		cudaMemcpyHostToDevice);
	CHECK_ERR(error_stat);
	error_stat = cudaMemcpy(dev_lh_mat, lh_mat, lh_row * lh_col * sizeof(float),
		cudaMemcpyHostToDevice);
	CHECK_ERR(error_stat);
	dim3 dim_block(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dim_grid(rh_col / dim_block.x, lh_row / dim_block.y);
	blocked_mat_mul <<<dim_grid, dim_block>>> (dev_lh_mat, dev_rh_mat, dev_res_mat, lh_row, lh_col, rh_row, rh_col);
	error_stat = cudaMemcpy(result_matrix, dev_res_mat, size, cudaMemcpyDeviceToHost);
	CHECK_ERR(error_stat);
cuda_free:
	if ((error_stat = cudaFree(dev_rh_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s",
			cudaGetErrorString(error_stat));
	}
	if ((error_stat = cudaFree(dev_lh_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s",
			cudaGetErrorString(error_stat));
	}
	if ((error_stat = cudaFree(dev_res_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s",
			cudaGetErrorString(error_stat));
	}
	return result_matrix;
free:
	free(result_matrix);
	goto cuda_free;
}

__global__ void matrix_multiply(const float *lh_mat, const float *rh_mat,
	float *res_mat, const int lh_row,
	const int lh_col, const int rh_row,
	const int rh_col) {
	int final_row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int final_col = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (final_row >= lh_row || final_col >= rh_col) {
		return;
	}
	for (std::size_t k = 0; k < lh_col; ++k) {
		res_mat[CUDA_IDX2C(final_row, final_col, rh_col)] +=
			lh_mat[CUDA_IDX2C(final_row, k, lh_col)] *
			rh_mat[CUDA_IDX2C(k, final_col, rh_col)];
	}
}

float *fl_cuda_matrix_multiply(float *lh_mat, float *rh_mat,
	const std::size_t lh_row,
	const std::size_t lh_col,
	const std::size_t rh_row,
	const std::size_t rh_col) {
	const std::size_t n = rh_row * lh_col;
	const std::size_t size = n * sizeof(float);
	float *result_matrix = (float *)malloc(size);
	if (result_matrix == NULL) {
		err_logln("Error allocating host memory!%s", "");
		return NULL;
	}
	float *dev_rh_mat, *dev_lh_mat, *dev_res_mat;
	cudaError_t error_stat;
	if ((error_stat = cudaMalloc(&dev_rh_mat, rh_row * rh_col * sizeof(float))) !=
		cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s",
			cudaGetErrorString(error_stat));
		free(result_matrix);
		return NULL;
	}
	if ((error_stat = cudaMalloc(&dev_lh_mat, lh_row * lh_col * sizeof(float))) !=
		cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s",
			cudaGetErrorString(error_stat));
		free(result_matrix);
		cudaFree(dev_rh_mat);
		return NULL;
	}
	if ((error_stat = cudaMalloc(&dev_res_mat, size)) != cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s",
			cudaGetErrorString(error_stat));
		free(result_matrix);
		cudaFree(dev_rh_mat);
		cudaFree(dev_lh_mat);
		return NULL;
	}
	error_stat = cudaMemcpy(dev_rh_mat, rh_mat, rh_row * rh_col * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_ERR(error_stat);
	error_stat = cudaMemcpy(dev_lh_mat, lh_mat, lh_row * lh_col * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_ERR(error_stat);
	dim3 threads_per_block(32, 32);
	dim3 numb_blocks(lh_row / threads_per_block.x + 1,
		rh_col / threads_per_block.y + 1);
	matrix_multiply <<<numb_blocks, threads_per_block >>> (
		dev_lh_mat, dev_rh_mat, dev_res_mat, lh_row, lh_col, rh_row, rh_col);
	error_stat = cudaMemcpy(result_matrix, dev_res_mat, size, cudaMemcpyDeviceToHost);
	CHECK_ERR(error_stat);
free:
	if ((error_stat = cudaFree(dev_rh_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s",
			cudaGetErrorString(error_stat));
	}
	if ((error_stat = cudaFree(dev_lh_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s",
			cudaGetErrorString(error_stat));
	}
	if ((error_stat = cudaFree(dev_res_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s",
			cudaGetErrorString(error_stat));
	}
	return result_matrix;
}

float *fl_cublas_matrix_multiply(float *lh_mat, float *rh_mat,
	std::size_t lh_row, std::size_t lh_col,
	std::size_t rh_row, std::size_t rh_col) {
	cudaError_t cuda_stat;
	cublasStatus_t status;
	cublasHandle_t handle;
	std::size_t n = lh_row * rh_col;
	std::size_t size = n * sizeof(float);
	float *ret_result = (float *)malloc(size);
	if (ret_result == NULL) {
		err_logln("Error allocating host memory.%s", "");
		return NULL;
	}
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		err_logln("Error initializing " GRN("cuBLAS") " context!");
		return NULL;
	}
	float *d_ret_result = NULL;
	float *dev_ptr_lh = NULL;
	float *dev_ptr_rh = NULL;
	if ((cuda_stat = cudaMalloc(&d_ret_result, size)) != cudaSuccess) {
		err_logln("Cuda error caught! Error: ", cudaGetErrorString(cuda_stat));
		goto destroy;
	}
	if ((cuda_stat = cudaMalloc(&dev_ptr_lh, sizeof(float) * lh_row * lh_col)) !=
		cudaSuccess) {
		err_logln("Cuda error caught! Error: ", cudaGetErrorString(cuda_stat));
		cudaFree(d_ret_result);
		goto destroy;
	}
	if ((cuda_stat = cudaMalloc(&dev_ptr_rh, sizeof(float) * rh_row * rh_col)) !=
		cudaSuccess) {
		err_logln("Cuda error caught! Error: ", cudaGetErrorString(cuda_stat));
		cudaFree(d_ret_result);
		cudaFree(dev_ptr_lh);
		goto destroy;
	}
	if ((cuda_stat =
		cudaMemcpy(dev_ptr_lh, lh_mat, sizeof(float) * lh_row * lh_col,
			cudaMemcpyHostToDevice)) != cudaSuccess) {
		err_logln("Cuda error caught! Error: ", cudaGetErrorString(cuda_stat));
		goto free;
	}
	if ((cuda_stat =
		cudaMemcpy(dev_ptr_rh, rh_mat, sizeof(float) * rh_row * rh_col,
			cudaMemcpyHostToDevice)) != cudaSuccess) {
		err_logln("Cuda error caught! Error: ", cudaGetErrorString(cuda_stat));
		goto free;
	}
	float alpha = 1.0f;
	float beta = 0.0f;
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rh_col, lh_row, lh_col,
		&alpha, dev_ptr_rh, rh_col, dev_ptr_lh, lh_col, &beta,
		d_ret_result, MIN(rh_col, lh_col));
	if (status != CUBLAS_STATUS_SUCCESS) {
		err_logln("Error multiplying matrices! Error code: %s",
			_cudaGetErrorEnum(status));
		goto free;
	}
	if ((cuda_stat = cudaMemcpy(ret_result, d_ret_result, size,
		cudaMemcpyDeviceToHost)) != cudaSuccess) {
		err_logln("Cuda error caught! Error: ", cudaGetErrorString(cuda_stat));
		goto free;
	}
cuda_free:
	cudaFree(dev_ptr_lh);
	cudaFree(dev_ptr_rh);
	cudaFree(d_ret_result);
destroy:
	cublasDestroy(handle);
	return ret_result;
free:
	free(ret_result);
	goto cuda_free;
}

/* cuBLAS API errors */
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}