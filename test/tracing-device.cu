#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../lib/Common.h"

#define cudaChecked(code) do {\
  cudaError_t err = code;\
  if (err != cudaSuccess) {\
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
        cudaGetErrorString(err));\
    abort();\
  }\
} while(0)\

extern "C" {
void __trace_touch(cudaStream_t stream);
void __trace_start(cudaStream_t stream, const char *kernel_name);
void __trace_fill_info(const void *info, cudaStream_t stream);
void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info);
void __trace_stop(cudaStream_t stream);
}

extern "C"
__device__ void __mem_trace (uint8_t* records, uint8_t* allocs, uint8_t* commits,
        uint64_t desc, uint64_t addr, uint32_t slot) {
    uint64_t cta = blockIdx.x;
    cta <<= 16;
    cta |= blockIdx.y;
    cta <<= 16;
    cta |= blockIdx.z;

    uint32_t lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));

    uint32_t active   = __ballot(1); // get number of active threads 
    uint32_t rlane_id = __popc(active << (32 - lane_id));
    uint32_t n_active = __popc(active);
    uint32_t lowest   = __ffs(active)-1;

    uint32_t *alloc = (uint32_t*)(&allocs[slot * CACHELINE]);
    uint32_t *commit = (uint32_t*)(&commits[slot * CACHELINE]);

    volatile uint32_t *valloc = alloc;
    volatile uint32_t *vcommit = commit;
    unsigned int id = 0;

    if (lane_id == lowest) {
      while(*valloc > (SLOTS_SIZE - 32) || (id = atomicAdd(alloc, n_active)) > (SLOTS_SIZE - 32)) {
        (void)0;
      }
    }

    uint32_t slot_offset = slot * SLOTS_SIZE;
    uint32_t record_offset = __shfl(id, lowest) + rlane_id;
    record_t *record = (record_t*) &(records[(slot_offset + record_offset) * RECORD_SIZE]);
    record->desc = desc;
    record->addr = addr;
    record->cta  = cta;
    __threadfence_system(); 

    if (lane_id == lowest ) atomicAdd(commit, n_active);
}

__global__ void test_kernel(uint8_t* records, uint8_t* allocs, uint8_t* commits, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid % 2 == 0)
    return;
  for (int i = 0; i < n; ++i) {
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

  const char* threads_str = getenv("THREADS");
  int32_t threads = threads_str ? strtol(threads_str, NULL, 10) : 32;

  if (!prop.unifiedAddressing) {
    printf("unified addressing not supported, unable to test device access from host\n");
    exit(0);
  }

  printf("trace for threads with gid.x %% 2 != 0\n");
  printf("%d threads, %d rounds, expect %d records\n", threads, rounds, threads * rounds / 2);

  setenv("MEMTRACE_PATTERN", "./test-trace-device", 1);
  __trace_touch(NULL);
  printf("starting trace\n");
  __trace_start(NULL, "test");

  traceinfo_t info;
  __trace_fill_info(&info, NULL);

  uint8_t *allocs = info.allocs;
  uint8_t *commits = info.commits;
  uint8_t *records = info.records;
  test_kernel<<<1, threads>>>(records, allocs, commits, rounds);
  cudaChecked(cudaDeviceSynchronize());

  printf("stopping trace\n");
  __trace_stop(NULL);
  cudaChecked(cudaStreamSynchronize(NULL));
  cudaChecked(cudaDeviceSynchronize());

  return 0;
}
