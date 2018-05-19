#pragma once

#include <stdint.h>

// Reference type definition
typedef struct {
  uint32_t *front;  
  uint32_t *back;
  uint64_t *slot;
  uint32_t slot_size;
} traceinfo_t;

// Size of a record in bytes, contents of a record:
// 32 bit meta info, 32bit size, 64 bit address, 64 bit cta id
#define RECORD_SIZE (24)
// 6M buffer, devided into 4 parallel slots.
// Buffers: SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE
// Absolute minimum is the warp size, all threads in a warp must collectively
// wait or be able to write a record
//#define SLOTS_SIZE (256 * 1024)
#define testing
#define SLOTS_SIZE (64)
#define SLOTS_NUM (4)

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
    Type::getInt32PtrTy(ctx),
    Type::getInt32PtrTy(ctx),
    Type::getInt64PtrTy(ctx),
    Type::getInt32Ty(ctx),
  };

  return StructType::create(fields, "traceinfo_t");
}

static
std::string getSymbolNameForKernel(const llvm::Twine &kernelName) {
  return ("__" + kernelName + "_trace").str();
}

#endif
