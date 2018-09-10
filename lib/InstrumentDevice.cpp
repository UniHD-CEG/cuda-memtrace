#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#define INCLUDE_LLVM_MEMTRACE_STUFF
#include "Common.h"

#define DEBUG_TYPE "memtrace-device"

#define ADDRESS_SPACE_GENERIC 0
#define ADDRESS_SPACE_GLOBAL 1
#define ADDRESS_SPACE_INTERNAL 2
#define ADDRESS_SPACE_SHARED 3
#define ADDRESS_SPACE_CONSTANT 4
#define ADDRESS_SPACE_LOCAL 5

using namespace llvm;

// enum TraceType {
//     Load    = 0 << 28,
//     Store   = 1 << 28,
//     Atomic  = 3 << 28,
//     Unknown = 4 << 28,
// };

/******************************************************************************
 * Various helper functions
 */

// Prototype
// __device__ void __mem_trace (uint8_t* records, uint8_t* allocs,
//  uint8_t* commits, uint64_t desc, uint64_t addr, uint32_t slot) {
Constant *getOrInsertTraceDecl(Module &M) {
  LLVMContext &ctx = M.getContext();

  Type *voidTy = Type::getVoidTy(ctx);
  Type *i8PtrTy = Type::getInt8PtrTy(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *i32Ty = Type::getInt32Ty(ctx);

  return M.getOrInsertFunction("__mem_trace", voidTy,
      i8PtrTy, i8PtrTy, i8PtrTy, i64Ty, i64Ty, i32Ty);
}

std::vector<Function*> getKernelFunctions(Module &M) {
    std::vector<Function*> Kernels;
    NamedMDNode * kernel_md = M.getNamedMetadata("nvvm.annotations");
    if (kernel_md) {
        // MDNodes in NamedMDNode
        for (const MDNode *node : kernel_md->operands()) {
            //node->dump();
            // MDOperands in MDNode
            for (const MDOperand &op : node->operands()) {
                Metadata * md = op.get();
                ValueAsMetadata *v = dyn_cast_or_null<ValueAsMetadata>(md);
                if (!v) continue;
                Function *f = dyn_cast<Function>(v->getValue());
                if (!f) continue;
                Kernels.push_back(f);
            }
        }
    }
    return Kernels;
}

GlobalVariable* defineDeviceGlobal(Module &M, Type* T, const Twine &name) {
  Constant *zero = Constant::getNullValue(T);
  auto *globalVar = new GlobalVariable(M, T, false,
      GlobalValue::ExternalLinkage, zero, name, nullptr,
      GlobalVariable::NotThreadLocal, 1, true);
  globalVar->setAlignment(1);
  globalVar->setDSOLocal(true);
  return globalVar;
}

/******************************************************************************
 * A poor man's infer address spaces, but instead of address spaces, we try
 * to infer visibility and it is implemented as a value analysis.
 */

enum PointerKind {
  PK_OTHER = 0,
  PK_GLOBAL,
  PK_UNINITIALIZED,
};

PointerKind mergePointerKinds(PointerKind pk1, PointerKind pk2) {
  return pk1 < pk2 ? pk1 : pk2;
}

PointerKind getPointerKind(Value* val, bool isKernel) {
  SmallPtrSet<Value*, 16> seen;
  SmallVector<Value*, 8> stack;
  PointerKind kind = PK_UNINITIALIZED;

  stack.push_back(val);
  while (!stack.empty()) {
    Value* node = stack.pop_back_val();
    if (seen.count(node) > 0)
      continue;
    seen.insert(node);

    //skip casts
    while (auto *cast = dyn_cast<BitCastOperator>(node)) {
      node = cast->getOperand(0);
    }
    if (isa<AllocaInst>(node)) {
      kind = mergePointerKinds(kind, PK_OTHER);
    } else if (isa<GlobalValue>(node)) {
      kind = mergePointerKinds(kind, PK_GLOBAL);
    } else if (isa<Argument>(node)) {
      kind = mergePointerKinds(kind, isKernel ? PK_GLOBAL : PK_OTHER);
    } else if (auto *gep = dyn_cast<GEPOperator>(node)) {
      stack.push_back(gep->getPointerOperand());
    } else if (auto *gep = dyn_cast<GetElementPtrInst>(node)) {
      stack.push_back(gep->getPointerOperand());
    } else if (auto *atomic = dyn_cast<AtomicRMWInst>(node)) {
      stack.push_back(atomic->getPointerOperand());
    } else if (auto *call = dyn_cast<CallInst>(node)) {
      report_fatal_error("Base Pointer is result of function. No.");
    } else if (auto *phi = dyn_cast<PHINode>(node)) {
      int numIncoming = phi->getNumIncomingValues();
      for (int i = 0; i < numIncoming; ++i) {
        stack.push_back(phi->getIncomingValue(i));
      }
    }
  }

  return kind;
}


/******************************************************************************
 * Device instrumentation pass.
 * It performs 3 fundamental steps for each kernel:
 *
 * 1. collect globally visible memory accesses in this kernel
 * 2. setup data structures used by tracing infrastructure
 * 3. instrument globally visible memory accesses with traces
 *
 * This pass does not analyze across function boundaries and therefore requires
 * any device functions to be inlined.
 */


// Needs to be a ModulePass because we modify the global variables.
struct InstrumentDevicePass : public ModulePass {
    static char ID;
    InstrumentDevicePass() : ModulePass(ID) {}

    struct TraceInfoValues {
      Value *Allocs;
      Value *Commits;
      Value *Records;

      Value *Desc;
      Value *Slot;
    };

    std::vector<Instruction*> collectGlobalMemAccesses(Function* kernel) {
      std::vector<Instruction*> result;
      for (auto &BB : *kernel) {
        for (auto &inst : BB) {
          PointerKind kind = PK_OTHER;
          if (auto *load = dyn_cast<LoadInst>(&inst)) {
            kind = getPointerKind(load->getPointerOperand(), true);
          } else if (auto *store = dyn_cast<StoreInst>(&inst)) {
            kind = getPointerKind(store->getPointerOperand(), true);
          } else if (auto *call = dyn_cast<CallInst>(&inst)) {
            Function* callee = call->getCalledFunction();
            if (callee == nullptr) continue;
            StringRef calleeName = callee->getName();
            if (calleeName.startswith("llvm.nvvm.atomic")) {
              kind = getPointerKind(call->getArgOperand(0), true);
            } else if ( calleeName == "__mem_trace") {
              report_fatal_error("already instrumented!");
            } else if ( !calleeName.startswith("llvm.") ) {
              std::string error = "call to non-intrinsic: ";
              error.append(calleeName);
              report_fatal_error(error.c_str());
            }
          } else {
            continue;
          }

          if (kind != PK_GLOBAL)
            continue;
          result.push_back(&inst);
        }
      }
      return result;
    }

    void setupTraceInfo(Function* kernel, TraceInfoValues *info) {
      LLVMContext &ctx = kernel->getParent()->getContext();
      Type *traceInfoTy = getTraceInfoType(ctx);

      IRBuilder<> IRB(kernel->getEntryBlock().getFirstNonPHI());

      Module &M = *kernel->getParent();
      std::string symbolName = getSymbolNameForKernel(kernel->getName());
      //errs() << "creating device symbol " << symbolName << "\n";
      auto* globalVar = defineDeviceGlobal(M, traceInfoTy, symbolName);
      assert(globalVar != nullptr);

      Value *AllocsPtr = IRB.CreateStructGEP(nullptr, globalVar, 0);
      Value *Allocs = IRB.CreateLoad(AllocsPtr, "allocs");

      Value *CommitsPtr = IRB.CreateStructGEP(nullptr, globalVar, 1);
      Value *Commits = IRB.CreateLoad(CommitsPtr, "commits");

      Value *RecordsPtr = IRB.CreateStructGEP(nullptr, globalVar, 2);
      Value *Records = IRB.CreateLoad(RecordsPtr, "records");

      IntegerType* I32Type = IntegerType::get(M.getContext(), 32);

      FunctionType *ASMFTy = FunctionType::get(I32Type, false);

      InlineAsm *SMIdASM = InlineAsm::get(ASMFTy,
          "mov.u32 $0, %smid;", "=r", false,
          InlineAsm::AsmDialect::AD_ATT );
      Value *SMId = IRB.CreateCall(SMIdASM);

      auto SMId64 = IRB.CreateZExtOrBitCast(SMId, IRB.getInt64Ty(), "desc");
      auto Desc = IRB.CreateShl(SMId64, 32);

      auto Slot = IRB.CreateAnd(SMId, IRB.getInt32(SLOTS_NUM - 1));

      info->Allocs = Allocs;
      info->Commits = Commits;
      info->Records = Records;
      info->Desc = Desc;
      info->Slot = Slot;
    }

    void instrumentKernel(Function *F, ArrayRef<Instruction*> MemAccesses,
        TraceInfoValues *info) {
      Module &M = *F->getParent();

      Constant* TraceCall = getOrInsertTraceDecl(M);
      if (!TraceCall) {
        report_fatal_error("No __mem_trace declaration found");
      }

      const DataLayout &DL = F->getParent()->getDataLayout();

      int LoadCounter = 0;
      int AtomicCounter = 0;
      int StoreCounter = 0;

      auto Allocs = info->Allocs;
      auto Commits = info->Commits;
      auto Records = info->Records;
      auto Slot = info->Slot;
      auto Desc = info->Desc;

      IRBuilder<> IRB(F->front().getFirstNonPHI());
      // Get Buffer Segment based on SMID and Load the Pointer

      for (auto *inst : MemAccesses) {
        Value *PtrOperand = nullptr;
        Value *LDesc = nullptr;
        IRB.SetInsertPoint(inst);

        if (auto li = dyn_cast<LoadInst>(inst)) {
          PtrOperand = li->getPointerOperand();
          LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_LOAD << ACCESS_TYPE_SHIFT));
          LoadCounter++;
        } else if (auto si = dyn_cast<StoreInst>(inst)) {
          PtrOperand = si->getPointerOperand();
          LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_STORE << ACCESS_TYPE_SHIFT));
          StoreCounter++;
        } else if (auto *FuncCall = dyn_cast<CallInst>(inst)) {
          assert(FuncCall->getCalledFunction()->getName()
              .startswith("llvm.nvvm.atomic"));
          PtrOperand = FuncCall->getArgOperand(0);
          LDesc      = IRB.CreateOr(Desc, ((uint64_t)ACCESS_ATOMIC << ACCESS_TYPE_SHIFT));
          AtomicCounter++;
        } else {
          report_fatal_error("invalid access type encountered, this should not have happened");
        }

        PointerType *PtrTy = dyn_cast<PointerType>(PtrOperand->getType());
        Type *ElemTy = PtrTy->getElementType();
        uint64_t ValueSize = DL.getTypeStoreSize(ElemTy);

        // Add tracing
        LDesc = IRB.CreateOr(LDesc, (uint64_t) ValueSize);
        auto PtrToStore = IRB.CreatePtrToInt(PtrOperand,  IRB.getInt64Ty());
        IRB.CreateCall(TraceCall,  {Records, Allocs, Commits, LDesc, PtrToStore, Slot});
      }
    }

    bool runOnModule(Module &M) override {
      bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
      if (!isCUDA) return false;

        for (auto *kernel : getKernelFunctions(M)) {
          auto accesses = collectGlobalMemAccesses(kernel);

          TraceInfoValues info;
          setupTraceInfo(kernel, &info);

          instrumentKernel(kernel, accesses, &info);
        }

        //M.dump();

        return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
    }

};
char InstrumentDevicePass::ID = 0;

namespace llvm {
  Pass *createInstrumentDevicePass() {
    return new InstrumentDevicePass();
  }
}

static RegisterPass<InstrumentDevicePass> X("memtrace-device", "includes static and dynamic load/store counting", false, false);
