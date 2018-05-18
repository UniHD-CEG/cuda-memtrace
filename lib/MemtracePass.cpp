
#define INCLUDE_LLVM_MEMTRACE_STUFF
#include "Common.h"
#undef INCLUDE_LLVM_MEMTRACE_STUFF

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/InlineAsm.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "memtrace-device"

#define ADDRESS_SPACE_GENERIC 0
#define ADDRESS_SPACE_GLOBAL 1
#define ADDRESS_SPACE_INTERNAL 2
#define ADDRESS_SPACE_SHARED 3
#define ADDRESS_SPACE_CONSTANT 4
#define ADDRESS_SPACE_LOCAL 5

using namespace llvm;

enum TraceType {
    Load    = 0 << 28,
    Store   = 1 << 28,
    Atomic  = 3 << 28,
    Unknown = 4 << 28,
};

/******************************************************************************
 * Various helper functions
 */

Constant *getOrInsertTraceDecl(Module &M) {
  LLVMContext &ctx = M.getContext();

  Type *voidTy = Type::getVoidTy(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *i64PtrTy = Type::getInt64PtrTy(ctx);

  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i32PtrTy = Type::getInt32PtrTy(ctx);

  return M.getOrInsertFunction("__mem_trace", voidTy,
      i64PtrTy, i32PtrTy, i32PtrTy,
      i32Ty, i64Ty, i64Ty,
      i32Ty, i32Ty);
  // Prototype:
  // __device__ void __mem_trace (
  //   uint64_t* __dbuff, uint32_t* __inx1, uint32_t* __inx2,
  //   uint32_t __max_n, uint64_t desc, uint64_t addr_val,
  //   uint32_t lane_id, uint32_t slot);
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
struct TracePass : public ModulePass {
    static char ID;
    TracePass() : ModulePass(ID) {}

    struct TraceInfoValues {
      Value* Front;
      Value* Back;
      Value* Slots;
      Value* SlotSize;
      Value* BaseDesc;
      Value* IndexSlot;
      Value* LaneId;
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
            StringRef calleeName = call->getCalledFunction()->getName();
            if (calleeName.startswith("llvm.nvvm.atomic")) {
              kind = getPointerKind(call->getArgOperand(0), true);
            } else if ( !calleeName.startswith("llvm.") ) {
              report_fatal_error("non-intrinsic call in kernel!");
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

      //errs() << "patching kernel '" << kernel->getName() << "'\n";

      Module &M = *kernel->getParent();
      auto *globalVar = new GlobalVariable(M, traceInfoTy, false,
          GlobalValue::ExternalLinkage, nullptr,
          getSymbolNameForKernel(kernel->getName()), nullptr);
      assert(globalVar != nullptr);
      globalVar->setAlignment(4);

      IRBuilder<> IRB(kernel->getEntryBlock().getFirstNonPHI());

      Value *FrontPtr = IRB.CreateStructGEP(traceInfoTy, globalVar, 0);
      Value *Front = IRB.CreateLoad(FrontPtr, "front");

      Value *BackPtr = IRB.CreateStructGEP(traceInfoTy, globalVar, 1);
      Value *Back = IRB.CreateLoad(BackPtr, "back");

      Value *SlotsPtr = IRB.CreateStructGEP(traceInfoTy, globalVar, 2);
      Value *Slots = IRB.CreateLoad(SlotsPtr, "slot");

      Value *SlotSizePtr = IRB.CreateStructGEP(traceInfoTy, globalVar, 3);
      Value *SlotSize = IRB.CreateLoad(SlotSizePtr, "slot_size");

      IntegerType* I32Type = IntegerType::get(M.getContext(), 32);

      FunctionType *ASMFTy = FunctionType::get(I32Type, false);

      InlineAsm *SMIdASM = InlineAsm::get(ASMFTy,
          "mov.u32 $0, %smid;", "=r", false,
          InlineAsm::AsmDialect::AD_ATT );
      Value *SMId = IRB.CreateCall(SMIdASM);

      InlineAsm *LaneIdASM = InlineAsm::get(ASMFTy,
          "mov.u32 $0, %laneid;", "=r", false,
          InlineAsm::AsmDialect::AD_ATT );
      Value *LaneId = IRB.CreateCall(LaneIdASM);

      auto SMId64 = IRB.CreateZExtOrBitCast(SMId, IRB.getInt64Ty(), "desc");
      auto BaseDesc = IRB.CreateShl(SMId64, 32);
      auto SlotMask  = IRB.getInt32(SLOTS_NUM - 1);
      auto IndexSlot = IRB.CreateAnd(SMId, SlotMask);

      info->Front = Front;
      info->Back = Back;
      info->Slots = Slots;
      info->SlotSize = SlotSize;
      info->BaseDesc = BaseDesc;
      info->IndexSlot = IndexSlot;
      info->LaneId = LaneId;
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

      auto IndexArray1 = info->Front;
      auto IndexArray2 = info->Back;
      auto DataBuff = info->Slots;
      auto MaxInx = info->SlotSize;
      auto LaneId = info->LaneId;
      auto IndexSlot = info->IndexSlot;
      auto Desc = info->BaseDesc;

      IRBuilder<> IRB(F->front().getFirstNonPHI());
      // Get Buffer Segment based on SMID and Load the Pointer

      for (auto *inst : MemAccesses) {
        Value *PtrOperand = nullptr;
        Value *LDesc = nullptr;
        IRB.SetInsertPoint(inst);

        if (auto li = dyn_cast<LoadInst>(inst)) {
          PtrOperand = li->getPointerOperand();
          LDesc = IRB.CreateOr(Desc, (uint64_t)TraceType::Load);
          LoadCounter++;
        } else if (auto si = dyn_cast<StoreInst>(inst)) {
          PtrOperand = si->getPointerOperand();
          LDesc = IRB.CreateOr(Desc, (uint64_t)TraceType::Store );
          StoreCounter++;
        } else if (auto *FuncCall = dyn_cast<CallInst>(inst)) {
          assert(FuncCall->getCalledFunction()->getName()
              .startswith("llvm.nvvm.atomic"));
          PtrOperand = FuncCall->getArgOperand(0);
          LDesc      = IRB.CreateOr(Desc, (uint64_t)TraceType::Atomic);
          AtomicCounter++;
        } else {
          report_fatal_error("this should be unreachable");
        }

        PointerType *PtrTy = dyn_cast<PointerType>(PtrOperand->getType());
        Type *ElemTy = PtrTy->getElementType();
        uint64_t ValueSize = DL.getTypeStoreSize(ElemTy);

        // Add tracing
        LDesc = IRB.CreateOr(LDesc, (uint64_t) ValueSize);
        auto PtrToStore = IRB.CreatePtrToInt(PtrOperand,  IRB.getInt64Ty());
        IRB.CreateCall(TraceCall, 
            {DataBuff, IndexArray1, IndexArray2,
            MaxInx, LDesc, PtrToStore, LaneId, IndexSlot});
      }
    }

    bool runOnModule(Module &M) override {
        // no CUDA module 
        if(M.getTargetTriple().find("nvptx") == std::string::npos) {
            return false;
        }

        for (auto *kernel : getKernelFunctions(M)) {
          auto accesses = collectGlobalMemAccesses(kernel);

          TraceInfoValues info;
          setupTraceInfo(kernel, &info);

          instrumentKernel(kernel, accesses, &info);
        }

        return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
    }

};
char TracePass::ID = 0;

static RegisterPass<TracePass> X("memtrace-device", "includes static and dynamic load/store counting", false, false);

// This enables Autoregistration of the Pass
static void registerTracePass(const PassManagerBuilder &,legacy::PassManagerBase &PM) {
    PM.add(new TracePass);
}
static RegisterStandardPasses RegisterTracePass(
    PassManagerBuilder::EP_OptimizerLast, registerTracePass);
static RegisterStandardPasses RegisterTracePass0(
    PassManagerBuilder::EP_EnabledOnOptLevel0, registerTracePass);
