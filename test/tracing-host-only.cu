#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../lib/Common.h"

extern "C" {
void __trace_touch(cudaStream_t stream);
void __trace_start(cudaStream_t stream, const char *kernel_name);
void __trace_fill_info(const void *info, cudaStream_t stream);
void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info);
void __trace_stop(cudaStream_t stream);
}

#define cudaChecked(code) do {\
  cudaError_t err = code;\
  if (err != cudaSuccess) {\
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
        cudaGetErrorString(err));\
    abort();\
  }\
} while(0)\

void add_trace(uint32_t *fronts, uint32_t *backs, uint64_t *traces,
    int slot, uint64_t desc, uint64_t addr, uint64_t size) {

  while (fronts[slot] >= SLOTS_SIZE) {}

  size_t offset = slot * SLOTS_SIZE + fronts[slot]*3;
  fronts[slot]++;
  traces[offset + 0] = desc;
  traces[offset + 1] = addr;
  traces[offset + 2] = size;
  backs[slot]++;
}

int main(int argc, char** argv) {
  // check if unified addressing is used, so that cudaHostGetDevicePtr is the
  // identity function.
  cudaDeviceProp prop;
  cudaChecked(cudaGetDeviceProperties(&prop, 0));

  if (!prop.unifiedAddressing) {
    printf("unified addressing not supported, unable to test device access from host\n");
    exit(0);
  }

  setenv("MEMTRACE_PATTERN", "./test-trace-host", 1);
  __trace_touch(NULL);
  printf("starting trace\n");
  __trace_start(NULL, "test");
  traceinfo_t info;
  __trace_fill_info(&info, NULL);

  uint32_t *fronts = info.front;
  uint32_t *backs = info.back;
  uint64_t *traces = info.slot;

  for (int i = 0; i < 256; ++i) {
    add_trace(fronts, backs, traces, 0, 3*i, 3*i + 1, 3*i + 2);
  }

  printf("stopping trace\n");
  __trace_stop(NULL);
  cudaChecked(cudaStreamSynchronize(NULL));

  return 0;
}
