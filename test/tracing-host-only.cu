#include <stdio.h>
#include <stdlib.h>
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
  printf("touching trace\n");
  __trace_touch(NULL);
  printf("starting trace\n");
  __trace_start(NULL, "test");
  printf("getting trace info\n");
  traceinfo_t info;
  __trace_fill_info(&info, NULL);
  printf("stopping trace\n");
  __trace_stop(NULL);
  printf("stopped trace\n");

  cudaChecked(cudaStreamSynchronize(NULL));

  return 0;
}
