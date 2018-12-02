#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <logger.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static const char *_cudaGetErrorEnum(cublasStatus_t error);

float *fl_cublas_matrix_multiply(float *rh_mat, float *lh_mat, 
		std::size_t lh_row, std::size_t lh_col, std::size_t rh_row, std::size_t rh_col)
{
	float *ret_result;
	std::size_t n = lh_row * rh_col;
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