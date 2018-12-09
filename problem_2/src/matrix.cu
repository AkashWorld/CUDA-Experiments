#include "matrix.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <logger.h>
#include <stdlib.h>
#define IDX2C(i,j,col_dim) (((i)*(col_dim))+(j))

static const char *_cudaGetErrorEnum(cublasStatus_t error);

__global__ void matrix_multiply(const float *lh_mat, const float *rh_mat, float *res_mat,
								const int lh_row, const int lh_col, 
								const int rh_row, const int rh_col)
{	
	int final_row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int final_col = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (final_row >= lh_row || final_col >= rh_col)
	{
		return;
	}
	for(std::size_t k = 0; k < lh_col; ++k)
	{
		res_mat[IDX2C(final_row, final_col, rh_col)] += lh_mat[IDX2C(final_row, k, lh_col)] * rh_mat[IDX2C(k, final_col, rh_col)];
	}
}

#define CHECK_ERR(x) if (x != cudaSuccess) {						\
	err_logln("Cuda error caught! Error: ", cudaGetErrorString(x)); \
	goto free;														\
}																	\

/*TODO: Complete*/
float *fl_cuda_matrix_multiply(float *lh_mat, float *rh_mat,
	const std::size_t lh_row, const std::size_t lh_col, const std::size_t rh_row, const std::size_t rh_col)
{
	const std::size_t n = rh_row * lh_col;
	const std::size_t size = n * sizeof(float);
	float *result_matrix = (float *)malloc(size);
	if (result_matrix == NULL)
	{
		err_logln("Error allocating host memory!%s", "");
		return NULL;
	}
	float *dev_rh_mat, *dev_lh_mat, *dev_res_mat;
	cudaError_t error_stat;
	if ((error_stat = cudaMalloc(&dev_rh_mat, rh_row*rh_col*sizeof(float))) != cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s", cudaGetErrorString(error_stat));
		free(result_matrix);
		return NULL;
	}
	if ((error_stat = cudaMalloc(&dev_lh_mat, lh_row*lh_col*sizeof(float))) != cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s", cudaGetErrorString(error_stat));
		free(result_matrix);
		cudaFree(dev_rh_mat);
		return NULL;
	}
	if ((error_stat = cudaMalloc(&dev_res_mat, size)) != cudaSuccess) {
		err_logln("Error allocating device memory! Error: %s", cudaGetErrorString(error_stat));
		free(result_matrix);
		cudaFree(dev_rh_mat);
		cudaFree(dev_lh_mat);
		return NULL;
	}
	error_stat = cudaMemcpy(dev_rh_mat, rh_mat, rh_row*rh_col * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_ERR(error_stat);
	error_stat = cudaMemcpy(dev_lh_mat, lh_mat, lh_row*lh_col * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_ERR(error_stat);
	dim3 threads_per_block(32, 32);
	dim3 numb_blocks(lh_row/threads_per_block.x + 1, rh_col/threads_per_block.y + 1);
	matrix_multiply <<<numb_blocks, threads_per_block>>> (dev_lh_mat, dev_rh_mat, dev_res_mat, lh_row, lh_col, rh_row, rh_col);
	error_stat = cudaMemcpy(result_matrix, dev_res_mat, size, cudaMemcpyDeviceToHost);
	CHECK_ERR(error_stat);
free:
	if ((error_stat = cudaFree(dev_rh_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s", cudaGetErrorString(error_stat));
	}
	if ((error_stat = cudaFree(dev_lh_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s", cudaGetErrorString(error_stat));
	}
	if ((error_stat = cudaFree(dev_res_mat)) != cudaSuccess) {
		err_logln("Error freeing device memory! Error: %s", cudaGetErrorString(error_stat));
	}
	return result_matrix;
}


/*TODO: Complete*/
float *fl_cublas_matrix_multiply(float *rh_mat, float *lh_mat,
	std::size_t lh_row, std::size_t lh_col, std::size_t rh_row, std::size_t rh_col)
{
	std::size_t n = lh_row * rh_col;
	std::size_t size = n * sizeof(float);
	float *ret_result = (float *)malloc(size);
	if (ret_result == NULL)
	{
		err_logln("Error allocating host memory.%s", "");
		return NULL;
	}
	cudaError_t cuda_stat;
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		err_logln("Error initializing " GRN("cuBLAS") " context!");
		return NULL;
	}
	float *dev_ptr_lh, *dev_ptr_rh;
	cuda_stat = cudaMalloc(&dev_ptr_lh, sizeof(float)*n);
	if (cuda_stat != cudaSuccess)
	{
		err_logln("Error allocating " GRN("device") "memory!, Error code: %d", cuda_stat);
		return NULL;
	}
	cuda_stat = cudaMalloc(&dev_ptr_rh, sizeof(float)*n);
	if (cuda_stat != cudaSuccess)
	{
		err_logln("Error allocating " GRN("device") "memory! Error code: %d", cuda_stat);
		return NULL;
	}
	status = cublasSetVector(n, sizeof(float), rh_mat, sizeof(float), dev_ptr_rh, sizeof(float));
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		err_logln("Error copying matrix into device! Error code: %s", _cudaGetErrorEnum(status));
		return NULL;
	}
	status = cublasSetVector(n, sizeof(float), lh_mat, sizeof(float), dev_ptr_lh, sizeof(float));
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		err_logln("Error copying matrix into device! Error code: %s", _cudaGetErrorEnum(status));
		return NULL;
	}
	status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, lh_row, lh_col, rh_col, NULL, dev_ptr_lh, lh_row,
		dev_ptr_rh, rh_row, NULL, ret_result, lh_row);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		err_logln("Error multiplying matrices! Error code: %s", _cudaGetErrorEnum(status));
		return NULL;
	}
	cublasDestroy(handle);
	return ret_result;
}

// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
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