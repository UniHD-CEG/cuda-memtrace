#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <unistd.h>

#include "../support/device-support.cu"
#include "../lib/Common.h"

#define cudaChecked(code) do {\
  cudaError_t err = code;\
  if (err != cudaSuccess) {\
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
        cudaGetErrorString(err));\
    abort();\
  }\
} while(0)\

extern "C"
__constant__ traceinfo_t globalVar;

extern "C" {
void __trace_touch(cudaStream_t stream);
void __trace_start(cudaStream_t stream, const char *kernel_name);
void __trace_fill_info(const void *info, cudaStream_t stream);
void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info);
void __trace_stop(cudaStream_t stream);
}

__global__ void test_kernel(uint8_t* records, uint8_t* allocs, uint8_t* commits, int32_t rounds, int32_t modulo) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid % modulo != 0) return;
  for (int i = 0; i < rounds; ++i) {
    __mem_trace(records, allocs, commits, gid, gid/32, gid/32 % SLOTS_NUM);
  }
}

int main(int argc, char** argv) {
  // check if unified addressing is used, so that cudaHostGetDevicePtr is the
  // identity function.
  cudaDeviceProp prop;
  cudaChecked(cudaGetDeviceProperties(&prop, 0));

  const char* rounds_str = getenv("ROUNDS");
  int32_t rounds = rounds_str ? strtol(rounds_str, NULL, 10) : 1;

  const char* threads_total_str = getenv("THREADS");
  int32_t threads_total = threads_total_str ? strtol(threads_total_str, NULL, 10) : 32;

  const char* modulo_str = getenv("MODULO");
  int32_t modulo = modulo_str ? strtol(modulo_str, NULL, 10) : 2;

  if (!prop.unifiedAddressing) {
    printf("unified addressing not supported, unable to test device access from host\n");
    exit(0);
  }

  int32_t threads = threads_total > 1024 ? 1024 : threads_total;
  int32_t blocks = (threads + 1023) / 1024;

  printf("blockSize.x = %d, blockDim.x = %d (total: %d), rounds: %d\n", threads, blocks, threads*blocks, rounds);
  printf("guard modulo: if (gid %% %d != 0) return;\n", modulo);
  printf("expected trace records: %" PRIu64 "\n", (uint64_t)threads*blocks*rounds / modulo);
  printf("---\n");

  setenv("MEMTRACE_PATTERN", "./test-trace-device", 1);
  __trace_touch(NULL);
  printf("starting trace\n");
  __trace_start(NULL, "test");

  traceinfo_t info;
  __trace_fill_info(&info, NULL);
  __trace_copy_to_symbol(NULL, "globalVar", &info);

  uint8_t *allocs = info.allocs;
  uint8_t *commits = info.commits;
  uint8_t *records = info.records;
  test_kernel<<<blocks, threads>>>(records, allocs, commits, rounds, modulo);
  cudaMemcpyToSymbol(globalVar, &info, sizeof(traceinfo_t), cudaMemcpyHostToDevice);
  cudaChecked(cudaDeviceSynchronize());

  printf("stopping trace\n");
  __trace_stop(NULL);
  cudaChecked(cudaStreamSynchronize(NULL));
  cudaChecked(cudaDeviceSynchronize());

  return 0;
}
