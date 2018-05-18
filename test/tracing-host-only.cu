#include <stdio.h>
#include <stdlib.h>
#include "../lib/Common.h"

#define cudaChecked(code) do {\
  if (code != cudaSuccess) {\
    printf("CUDA error at %s:%s: %s", __FILE__, __LINE__,\
        cudaGetErrorString(code));\
    abort();\
  }\
}\

int main(int argc, char** argv) {
  // check if unified addressing is used, so that cudaHostGetDevicePtr is the
  // identity function.
  cudaDeviceProp prop;
  cudaChecked(cudaGetDeviceProperties(&prop, 0));

  if (!prop.unifiedAddressing) {
    printf("unified addressing not supported, unable to test device access from host\n");
    exit(0);
  }

  setenv("MEMTRACE_PATTERN", "./test-trace-host");
  __trace_start(NULL, "test");
  traceinfo_t info;
  __trace_fill_info(&info, NULL);
  __trace_stop(NULL);
}
