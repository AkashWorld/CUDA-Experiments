#include <stdio.h>
#include <iostream>
#include "brute_forcer.cuh"


#define TOTAL_BLOCKS 8192
#define TOTAL_THREADS 128
#define HASH_ITER 64
#define WORD_LENGTH_MAX 8
#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+~,."
#define CHARSET_LENGTH (sizeof(CHARSET) - 1)

char block_word[WORD_LENGTH_MAX];
char host_charset[CHARSET_LENGTH + 1];
char host_cracked[WORD_LENGTH_MAX];
uint8_t curr_len;

__device__ char device_charset[CHARSET_LENGTH + 1];
__device__ char device_cracked[WORD_LENGTH_MAX];

int main(int argc, char* argv[]){
    if(argc < 2){
        std::cout << "Correct usage: brute_forcer <md5 hash>\n";
        exit(1);
    } else if(strlen(argv[1]) != 32){
        std::cout << "MD5 not correct length\n";
        exit(1);
    }
  
  std::cout << "Attempting to find password for hash " << argv[1] << std::endl;

  uint32_t digest[4];
  md5_to_str(digest, argv[1]);


  

  memset(block_word, 0, WORD_LENGTH_MAX);
  memset(host_cracked, 0, WORD_LENGTH_MAX);
  memcpy(host_charset, CHARSET, CHARSET_LENGTH);
  

  char * words = new char;
  curr_len = 1;
  

  cudaEvent_t start;
  cudaEvent_t stop;
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  cudaMemcpyToSymbol(device_charset, host_charset, sizeof(uint8_t) * CHARSET_LENGTH + 1, 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(device_cracked, host_cracked, sizeof(uint8_t) * WORD_LENGTH_MAX, 0, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&words, sizeof(uint8_t) * WORD_LENGTH_MAX);
  
  while(true){
    bool found = false;
    bool inc_res = false;
    
      cudaMemcpy(words, block_word, sizeof(uint8_t) * WORD_LENGTH_MAX, cudaMemcpyHostToDevice); 
    
      compare_hashes<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(curr_len, words, digest[0], digest[1], digest[2], digest[3]);
      
      inc_res = iterate(&curr_len, block_word, TOTAL_THREADS * HASH_ITER * TOTAL_BLOCKS);
      
    
      cudaDeviceSynchronize();

      cudaMemcpyFromSymbol(host_cracked, device_cracked, sizeof(uint8_t) * WORD_LENGTH_MAX, 0, cudaMemcpyDeviceToHost); 
      
      if(found = *host_cracked != 0){     
        std::cout << "Password found: " << host_cracked << std::endl; 
        break;
      }
    
    if(!inc_res || found){
      break;
    }
  }
  
  cudaFree((void*)words);

  float ms = 0;
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  
  std::cout << "Computation took " << ms << " ms" << std::endl;
}

__device__ __host__ bool iterate(uint8_t* length, char* word, uint32_t increment){

  uint32_t inc = 0;
  uint32_t idx = 0;
  
  while(increment > 0 && idx < WORD_LENGTH_MAX){
    if(idx >= *length && increment > 0){
      increment--;
    }
    inc = increment + word[idx];
    word[idx] = inc % CHARSET_LENGTH;
    increment = inc / CHARSET_LENGTH;
    idx++;
  }
  
  if(idx > *length){
    *length = idx;
  }
  
  if(idx > WORD_LENGTH_MAX){
    return false;
  }

  return true;
}

__global__ void compare_hashes(uint8_t length, char* word_char, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04){

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASH_ITER;
  
  __shared__ char shared_charset[CHARSET_LENGTH + 1];
  
  uint32_t local_hash01, local_hash02, local_hash03, local_hash04;
  uint8_t local_len;
  char word_start[WORD_LENGTH_MAX];
  char curr_word[WORD_LENGTH_MAX];

  memcpy(word_start, word_char, WORD_LENGTH_MAX);
  memcpy(&local_len, &length, sizeof(uint8_t));
  memcpy(shared_charset, device_charset, sizeof(uint8_t) * CHARSET_LENGTH + 1);
  
  iterate(&local_len, word_start, idx);
  
  for(uint32_t hash = 0; hash < HASH_ITER; hash++){
    for(uint32_t i = 0; i < local_len; i++){
      curr_word[i] = shared_charset[word_start[i]];
    }
    
    md5((unsigned char*)curr_word, local_len, &local_hash01, &local_hash02, &local_hash03, &local_hash04);   

    if(local_hash01 == hash01 && local_hash02 == hash02 && local_hash03 == hash03 && local_hash04 == hash04){
      memcpy(device_cracked, curr_word, local_len);
    }
    
    if(!iterate(&local_len, word_start, 1)){
      break;
    }
  }
}

__host__ void md5_to_str(uint32_t digest[], char * str) {
  for(uint8_t i = 0; i < 4; i++){
    char tmp[16];
    strncpy(tmp, str + i * 8, 8);
    sscanf(tmp, "%x", &digest[i]);   
    digest[i] = (digest[i] & 0xFF000000) >> 24 | (digest[i] & 0x00FF0000) >> 8 | (digest[i] & 0x0000FF00) << 8 | (digest[i] & 0x000000FF) << 24;
  }
}