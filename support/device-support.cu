__device__ __inline__ uint32_t __rlaneid( unsigned int bitmap,int lane_id ) {
    int r_count = 0;
    for(int i = 0; i < lane_id; i++) {
        if (bitmap & 0x1) ++r_count; 
        bitmap >>=1;
    } 
    return r_count;
}

__device__ void __mem_trace (
        uint64_t* __dbuff,
        uint32_t* __inx1,
        uint32_t* __inx2,
        uint32_t __max_n,
        uint64_t desc,
        uint64_t addr_val,
        uint32_t lane_id,
        uint32_t slot) {
    uint64_t cta = blockIdx.x;
    uint16_t ctay = blockIdx.y;
    uint16_t ctaz = blockIdx.z;
    cta  <<= 16;
    cta  =  cta | ctay;
    cta  <<= 16;
    cta  =  cta | ctaz;

    unsigned int idx;
    uint32_t *i1 = &(__inx1[slot]);
    uint32_t *i2 = &(__inx2[slot]);

    volatile uint32_t *vi2 = i2;
    int active   = __ballot(1); // get number of active threads 
    int rlane_id = __rlaneid(active, lane_id);
    int n_active = __popc(active);
    int lowest   = __ffs(active)-1;
    unsigned int id = 0;

    if (lane_id == lowest) {
        while( *vi2 >= __max_n-95 || (id = atomicAdd(i1, n_active*3)) >= __max_n-95);
    }

    idx = __shfl(id, lowest) + 3 * rlane_id;
    int offset = slot * __max_n;
    __dbuff[offset + idx]     = desc;
    __dbuff[offset + idx + 1] = addr_val;
    __dbuff[offset + idx + 2] = cta;
    if (lane_id == lowest ) atomicAdd(i2, n_active*3);
    __threadfence_system(); 
    return;
}
