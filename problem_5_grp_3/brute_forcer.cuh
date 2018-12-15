#include "md5.cu"

__device__ __host__ bool iterate(uint8_t*, char*, uint32_t);
__global__ void compare_hashes(uint8_t, char*, uint32_t, uint32_t, uint32_t, uint32_t);
__device__ inline void md5(unsigned char*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
__host__ void md5_to_str(uint32_t[], char *);