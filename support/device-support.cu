#include "../lib/Common.h"

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
