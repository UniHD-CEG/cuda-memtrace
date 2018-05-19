#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
//__device__ void __mem_trace (uint64_t* __dbuff, uint32_t* __inx1,
//    uint32_t* __inx2, uint32_t __max_n, uint64_t desc,
//    uint64_t addr_val, uint32_t lane_id, uint32_t slot);
}

struct record_t {
  uint64_t desc;
  uint64_t addr;
  uint64_t cta;
};

__device__ void __mem_trace (
        uint64_t* __dbuff,
        uint32_t* fronts,
        uint32_t* backs,
        uint64_t desc,
        uint64_t addr_val,
        uint32_t slot) {

    uint64_t cta = blockIdx.x;
    cta <<= 16;
    cta |= blockIdx.y;
    cta <<= 16;
    cta |= blockIdx.z;

    uint32_t *front = fronts + slot;
    uint32_t *back = backs + slot;
    volatile uint32_t *vfront = front;

    record_t *slot_buf = ((record_t*)__dbuff) + slot * SLOTS_SIZE;

    uint32_t lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));

    //volatile uint32_t *vi2 = i2;
    uint32_t ballot   = __ballot(1); // get number of active threads 
    //int rlane_id = __rlaneid(active, lane_id);
    int n_active = __popc(ballot);
    // relative lane id: number of active threads with lower ids.
    uint32_t rel_lane_id = __popc(ballot >> (32 - lane_id));
    int lowest = __ffs(ballot)-1;
    unsigned int old_front = 0;

    // if buffer is full (less than one record per active thread in this warp),
    // busy wait on host consumer to clear up buffer. Due to atomic add only
    // performed by first active thread in this warp
    if (lane_id == lowest) {
      while (*vfront >= (SLOTS_SIZE - 32) ||
          (old_front = atomicAdd(front, n_active)) >= (SLOTS_SIZE - 32)) {
        (void)0;
      }
    }

    // Calculate index in warp in warp:
    size_t idx = __shfl(old_front, lowest) + rel_lane_id;
    slot_buf[idx].desc = desc;
    slot_buf[idx].addr = addr_val;
    slot_buf[idx].cta = cta;
    if (lane_id == lowest) atomicAdd(back, n_active);
    __threadfence_system(); 
    return;
}

__global__ void test_kernel(uint64_t* traces, uint32_t* front, uint32_t* back, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid % 2 == 0)
    return;
  for (int i = 0; i < n; ++i) {
    __mem_trace(traces, front, back, gid, 2, 0);
  }
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

  setenv("MEMTRACE_PATTERN", "./test-trace-device", 1);
  __trace_touch(NULL);
  printf("starting trace\n");
  __trace_start(NULL, "test");

  traceinfo_t info;
  __trace_fill_info(&info, NULL);

  uint32_t *fronts = info.front;
  uint32_t *backs = info.back;
  uint64_t *traces = info.slot;
  sleep(1);
  test_kernel<<<1, 64>>>(traces, fronts, backs, 4);
  cudaChecked(cudaDeviceSynchronize());

  printf("stopping trace\n");
  __trace_stop(NULL);
  cudaChecked(cudaStreamSynchronize(NULL));
  cudaChecked(cudaDeviceSynchronize());

  return 0;
}
