#pragma once

#include <stdint.h>

// Reference type definition
typedef struct {
  uint8_t *allocs;
  uint8_t *commits;
  uint8_t *records;
  uint32_t slot_size;
} traceinfo_t;

typedef struct record_t {
  uint64_t desc;
  uint64_t addr;
  uint64_t cta;
} record_t;

#define ACCESS_TYPE_SHIFT (28)
enum ACCESS_TYPE {
  ACCESS_LOAD = 0,
  ACCESS_STORE = 1,
  ACCESS_ATOMIC = 2,
  ACCESS_UNKNOWN = 3,
};

#define GET_ACCESS_SM(R) (int)((R.desc >> 28) & 0xf)
#define GET_ACCESS_TYPE(R) (int)((R.desc >> 28) & 0xf)
#define GET_ACCESS_SIZE(R) (int)(record.desc & (0x0fffffff))
#define GET_ACCESS_ADDR(R) (uint64_t)(record.addr)
#define GET_ACCESS_CTAIDX(R) (int)((record.cta >> 32) & 0xffffffff)
#define GET_ACCESS_CTAIDY(R) (int)((record.cta >> 16) & 0x0000ffff)
#define GET_ACCESS_CTAIDZ(R) (int)((record.cta) & 0x0000ffff)

// Field "desc"
#define MAKE_ACCESS_SM(id)     ((((uint64_t)id) &   0xffffffff) << 32)
#define MAKE_ACCESS_TYPE(type) ((((uint64_t)type) & 0x0000000f) << 28)
#define MAKE_ACCESS_SIZE(size)  (((uint64_t)size) & 0x0fffffff)

// Field "desc"



// Size of a record in bytes, contents of a record:
// 32 bit meta info, 32bit size, 64 bit address, 64 bit cta id
#define RECORD_SIZE (24)
// 6M buffer, devided into 4 parallel slots.
// Buffers: SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE
// Absolute minimum is the warp size, all threads in a warp must collectively
// wait or be able to write a record
#define SLOTS_SIZE (256 * 1024)
// Number of slots must be power of two!
#define SLOTS_NUM (8)

#define CACHELINE (64)

#ifdef INCLUDE_LLVM_MEMTRACE_STUFF

// functions need to be static because we link it into both host and device
// instrumentation

#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/Twine.h"

static
llvm::StructType* getTraceInfoType(llvm::LLVMContext &ctx) {
  using llvm::Type;
  using llvm::StructType;

  Type *fields[] = {
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt32Ty(ctx),
  };

  return StructType::create(fields, "traceinfo_t");
}

static
std::string getSymbolNameForKernel(const llvm::Twine &kernelName) {
  return ("__" + kernelName + "_trace").str();
}

#endif
