#include "../lib/Common.h"

extern "C"
__device__ void __mem_trace (
        uint8_t* records,
        uint32_t* allocs,
        uint32_t* commits,
        uint64_t desc,
        uint64_t addr_val,
        uint32_t slot) {

    uint64_t cta = blockIdx.x;
    cta <<= 16;
    cta |= blockIdx.y;
    cta <<= 16;
    cta |= blockIdx.z;

    uint32_t *alloc = &allocs[slot];
    uint32_t *commit = &commits[slot];

    uint32_t lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));

    int active   = __ballot(1); // get number of active threads 
    int rlane_id = __popc(active >> (32 - lane_id));
    int n_active = __popc(active);
    int lowest   = __ffs(active)-1;

    volatile uint32_t *valloc = alloc;
    unsigned int id = 0;
    if (lane_id == lowest) {
        while( *valloc > (SLOTS_SIZE - 32) || (id = atomicAdd(alloc, n_active)) > (SLOTS_SIZE - 32));
    }

    int offset = slot * SLOTS_SIZE * RECORD_SIZE;
    unsigned int idx = (__shfl(id, lowest) + rlane_id) * RECORD_SIZE;
    *(uint64_t*)(records + offset + idx + 0) = desc;
    *(uint64_t*)(records + offset + idx + 1) = addr_val;
    *(uint64_t*)(records + offset + idx + 2) = cta;
    if (lane_id == lowest ) atomicAdd(commit, n_active);
    __threadfence_system(); 
    return;
}
