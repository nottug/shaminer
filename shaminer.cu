#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include "sha256.cuh"

#define SHA_PER_ITERATIONS 8'388'608
#define BLOCK_SIZE 512
#define NUM_BLOCKS (SHA_PER_ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE

struct Job {
  char *input;
  size_t input_size;
  char *target;
  size_t target_size;
};

struct Solution {
  int *found;
  char *nonce;
  unsigned char *hash;
};

// get current time
time_t get_time_ms() {
  struct timeval time_now {};
  gettimeofday(&time_now, nullptr);
  time_t time_now_ms = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);

  return time_now_ms;
}

// converts a hex string into a byte
__device__ unsigned char hex2byte(unsigned char c) {
  if ((c >= '0') && (c <= '9')) {
    return c - 0x30;
  } else if ((c >= 'a') && (c <= 'f')) {
    return c - 0x61 + 0x0A;
  } else if ((c >= 'A') && (c <= 'F')) {
    return c - 0x41 + 0x0A;
  }

  return 0;
}

// checks a hash against a given target
__device__ bool check_hash(unsigned char *hash, char *target,
                           size_t target_size) {
  if (target_size > 64) {
    return false;
  }

  for (int i = 0; i < target_size; i++) {
    // @TODO: this is an inefficient way to check for equality
    unsigned char elem = hash[i / 2] % 16;
    if (i % 2 == 0) {
      elem = (hash[i / 2] - elem) / 16;
    }

    if (elem != hex2byte(target[i])) {
      return false;
    }
  }

  return true;
}

// does the same as sprintf(char*, "%d%s", int, const char*) but a bit faster
__device__ uint8_t nonce_to_str(uint64_t nonce, unsigned char *out) {
  uint64_t result = nonce;
  uint8_t remainder;
  uint8_t nonce_size = nonce == 0 ? 1 : floor(log10((double)nonce)) + 1;
  uint8_t i = nonce_size;
  while (result >= 10) {
    remainder = result % 10;
    result /= 10;
    out[--i] = remainder + '0';
  }

  out[0] = result + '0';
  i = nonce_size;
  out[i] = 0;
  return i;
}

// copy the k values from host to device
void sha256_preflight() {
  checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0,
                                     cudaMemcpyHostToDevice));
}

extern __shared__ char input_array[];
__global__ void sha256_kernel(Job job, uint64_t nonce_offset, Solution sol) {
  // If this is the first thread of the block, init the seed string in shared
  // memory
  char *seed = (char *)&input_array[0];
  if (threadIdx.x == 0) {
    memcpy(seed, job.input, job.input_size + 1);
  }

  __syncthreads(); // wait for seed string to be written in shared memory

  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t raw_nonce = idx + nonce_offset;

  // the first byte we can write because there is the input string at the
  // begining respects the memory padding of 8 bit (char)
  size_t const min_array =
      static_cast<size_t>(ceil((job.input_size + 1) / 8.f) * 8);

  uintptr_t hash_addr = threadIdx.x * (64) + min_array;
  uintptr_t nonce_addr = hash_addr + 32;

  unsigned char *hash = (unsigned char *)&input_array[hash_addr];
  unsigned char *nonce = (unsigned char *)&input_array[nonce_addr];
  memset(nonce, 0, 32);
  uint8_t nonce_size = nonce_to_str(raw_nonce, nonce);

  {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (unsigned char *)seed, job.input_size);
    sha256_update(&ctx, nonce, nonce_size);
    sha256_final(&ctx, hash);
  }

  if (check_hash(hash, job.target, job.target_size) &&
      atomicExch(sol.found, 1) == 0) {
    memcpy(sol.hash, hash, 32);
    memcpy(sol.nonce, nonce, nonce_size);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("usage: ./shaminer {INPUT} {TARGET}\n");
    exit(1);
  }

  char *input = argv[1];
  char *target = argv[2];

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  sha256_preflight();

  // initialize job
  Job job;
  job.input_size = strlen(input);
  job.target_size = strlen(target);
  cudaMalloc(&(job.input), job.input_size + 1);
  cudaMalloc(&(job.target), job.target_size + 1);
  cudaMemcpy(job.input, input, job.input_size + 1, cudaMemcpyHostToDevice);
  cudaMemcpy(job.target, target, job.target_size + 1, cudaMemcpyHostToDevice);

  // initialize solution
  Solution sol;
  cudaMallocManaged(&(sol.found), sizeof(int));
  cudaMallocManaged(&(sol.nonce), job.input_size + 32 + 1);
  cudaMallocManaged(&(sol.hash), 32);

  // initialize loop variables
  static uint64_t nonce = 0;
  size_t shared_size =
      (ceil((job.input_size + 1) / 8.f) * 8) + (64 * BLOCK_SIZE);

  // initialize hashrate values
  float hashrate = 0.0;
  uint64_t last_nonce = 0;
  time_t time_start_ms = get_time_ms();
  time_t time_last_ms = time_start_ms;

  // loop until solution has been found
  while (!*sol.found) {
    sha256_kernel<<<NUM_BLOCKS, BLOCK_SIZE, shared_size>>>(job, nonce, sol);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("cuda device error\n");
      exit(1);
    }

    // @TODO: manage hashrate reporting in a more reasonable way

    // every 5 seconds print hashrate
    time_t time_now_ms = get_time_ms();
    time_t time_diff = time_now_ms - time_last_ms;
    if (time_diff > 5000) {
      hashrate = (float)(nonce - last_nonce) / ((float)time_diff / 1000);
      printf("hashrate: %0.4f\n", hashrate);

      last_nonce = nonce;
      time_last_ms = time_now_ms;
    }

    nonce += NUM_BLOCKS * BLOCK_SIZE;
  }

  // @TODO: fix luck calc by using the actual nonce

  // calculate final statistics
  uint64_t expected_nonces = pow(16, strlen(target));
  float luck = ((float)expected_nonces / (float)nonce) * 100;
  float duration = (float)(get_time_ms() - time_start_ms) / 1000;

  // print results
  printf("\nfound solution in %.2fs with %.2f%% luck", duration, luck);
  printf("\ninput: %s%s", input, sol.nonce);
  printf("\nnonce: %s", sol.nonce);
  printf("\nhash: ");
  for (uint8_t i = 0; i < 32; ++i) {
    printf("%02x", sol.hash[i]);
  }
  printf("\n");

  // cleanup device
  cudaFree(sol.hash);
  cudaFree(sol.nonce);
  cudaFree(sol.found);
  cudaFree(job.target);
  cudaFree(job.input);
  cudaDeviceReset();

  return 0;
}
