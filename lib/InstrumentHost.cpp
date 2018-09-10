/*******************************************************************************
 * This host instrumentation pass inserts calls into the host support library.
 * The host support library is used to set up queues for traces that are sinked
 * into a thread that writes them into a file.
 * A kernel launch is split into two major parts:
 * 1. cudaConfigureCall()
 * 2. <wrapper>() -> cudaLaunch()
 * The function cudaConfigurCall sets up the execution grid and stream to
 * execute in and the wrapper function sets up kernel arguments and launches
 * the kernel.
 * Instrumentation requires the stream, set in cudaConfigureCall, as well as the
 * kernel name, implicitly "set" by the wrapper function.
 * This pass defines the location of a kernel launch as the call to
 * cudaConfigureCall, which the module is searched for.
 *
 * Finding the kernel name boils down to following the execution path assuming
 * no errors occur during config and argument setup until we find:
 * 1. a call cudaLaunch and return the name of the first operand, OR
 * 2. a call to something other than cudaSetupArguent and return its name
 *
 */

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#define INCLUDE_LLVM_MEMTRACE_STUFF
#include "Common.h"

#define DEBUG_TYPE "memtrace-host"

using namespace llvm;

struct KernelCallInfo {
  std::string kernelName;
    
};

void createPrintf(IRBuilder<> &IRB, const Twine &fmt, ArrayRef<Value*> values) {
  Module &M = *IRB.GetInsertBlock()->getModule();
  Function* Printf = M.getFunction("printf");
  auto *FormatGlobal = IRB.CreateGlobalString(fmt.getSingleStringRef());
  Type* charPtrTy = IRB.getInt8Ty()->getPointerTo();
  Value* Format = IRB.CreateBitCast(FormatGlobal, charPtrTy);
  SmallVector<Value*, 4> args;
  args.append({Format});
  args.append(values.begin(), values.end());
  IRB.CreateCall(Printf, args);
}

struct InstrumentHost : public ModulePass {
    static char ID;
    InstrumentHost() : ModulePass(ID) {}

    Type* traceInfoTy = nullptr;

    Constant *TraceFillInfo = nullptr;
    Constant *TraceCopyToSymbol = nullptr;
    Constant *TraceTouch = nullptr;
    Constant *TraceStart = nullptr;
    Constant *TraceStop = nullptr;

    /** Create a global Variable and tell the CUDA runtime to link it with a global
     * variable in device memory with the same name.
     */
    GlobalVariable* getOrCreateCudaGlobalVar(Module &M, Type* T, const Twine &name) {
      // first see if the variable already exists
      GlobalVariable *Global = M.getGlobalVariable(name.getSingleStringRef());
      if (Global) {
        return Global;
      }

      // Variable does not exist, so we create one and register it
      Constant *zero = Constant::getNullValue(T);
      Global = new GlobalVariable(M, T, false, GlobalValue::InternalLinkage, zero, name);
      Global->setAlignment(8);
      assert(Global != nullptr);

      /* Get declaration of cuda initialization function created bei clang
       */
      Function* CudaSetup = M.getFunction("__cuda_register_globals");
      if (CudaSetup == nullptr) {
        errs() << "WARNING: NO '__cuda_register_globals' FOUND! (normal if compiling to bitcode or host-only)\n";
        return Global;
      }
      assert(CudaSetup != nullptr);
      IRBuilder<> IRB(M.getContext());

      //IRB.SetInsertPoint(CudaSetup->getEntryBlock().getFirstNonPHIOrDbg());
      IRB.SetInsertPoint(&CudaSetup->back().back());

      /** Get declaration of __cudaRegisterVar.
       * Protype:
       *  extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle,
       *   char  *hostVar, char  *deviceAddress, const char  *deviceName,
       *   int ext, int size, int constant, int global);
       */
      // no void*/* in llvm, we use i8*/* instead
      Type* voidPtrPtrTy = IRB.getInt8Ty()->getPointerTo()->getPointerTo();
      Type* charPtrTy = IRB.getInt8Ty()->getPointerTo();
      Type* intTy = IRB.getInt32Ty();
      auto *FnTy = FunctionType::get(intTy, {voidPtrPtrTy,
          charPtrTy, charPtrTy, charPtrTy,
          intTy, intTy, intTy, intTy}, false);
      auto *Fn = M.getOrInsertFunction("__cudaRegisterVar", FnTy);
      assert(Fn != nullptr);

      auto *GlobalNameLiteral = IRB.CreateGlobalString(Global->getName());
      auto *GlobalName = IRB.CreateBitCast(GlobalNameLiteral, charPtrTy);

      auto *GlobalAddress = IRB.CreateBitCast(Global, charPtrTy);

      uint64_t GlobalSize = M.getDataLayout().getTypeStoreSize(T);

      Value *CubinHandle = &*CudaSetup->arg_begin();

      //createPrintf(IRB, "registering... symbol name: %s, symbol address: %p, name address: %p\n",
      //    {GlobalName, GlobalAddress, GlobalName});

      IRB.CreateCall(Fn, {CubinHandle, GlobalAddress, GlobalName, GlobalName,
          IRB.getInt32(0), IRB.getInt32(GlobalSize), IRB.getInt32(0), IRB.getInt32(0)});

      //createPrintf(IRB, "return code of cudaRegisterVar: %d\n", {RegisterErr});

      return Global;
    }

    /** Sets up pointers to (and inserts prototypes of) the utility functions
     * from the host-support library.
     * We're pretending all pointer types are identical, linker does not
     * complain in tests.
     *
     * Reference:
     * void __trace_fill_info(const void *info, cudaStream_t stream);
     * void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info);
     * void __trace_touch(cudaStream_t stream);
     * void __trace_start(cudaStream_t stream, const char *kernel_name);
     * void __trace_stop(cudaStream_t stream);
     */
    void findOrInsertRuntimeFunctions(Module &M) {
      LLVMContext &ctx = M.getContext();
      Type* cuStreamPtrTy = Type::getInt8PtrTy(ctx);
      Type* voidPtrTy = Type::getInt8PtrTy(ctx);
      Type* stringTy = Type::getInt8PtrTy(ctx);
      Type* voidTy = Type::getVoidTy(ctx);

      TraceFillInfo = M.getOrInsertFunction("__trace_fill_info",
          voidTy, voidPtrTy, cuStreamPtrTy);
      TraceCopyToSymbol = M.getOrInsertFunction("__trace_copy_to_symbol",
          voidTy, cuStreamPtrTy, voidPtrTy, voidPtrTy);
      TraceTouch = M.getOrInsertFunction("__trace_touch",
          voidTy, cuStreamPtrTy);
      TraceStart = M.getOrInsertFunction("__trace_start",
          voidTy, cuStreamPtrTy, stringTy);
      TraceStop = M.getOrInsertFunction("__trace_stop",
          voidTy, cuStreamPtrTy);
    }

    /** Find the kernel launch or wrapper function belonging to a
     * cudaConfigureCall. Must handle inlined and non-inlined cases.
     */
    CallInst* searchKernelLaunchFor(Instruction *inst) {
      while (inst != nullptr) {
        // on conditional branches, assume everything went okay
        if (auto *br = dyn_cast<BranchInst>(inst)) {
          if (br->isUnconditional()) {
            inst = inst->getNextNode();
            continue;
          } else {
            auto *BB = br->getSuccessor(0);
            StringRef name = BB->getName();
            if (name.startswith("kcall.configok") || name.startswith("setup.next")) {
              inst = BB->getFirstNonPHI();
              continue;
            }
            BB = br->getSuccessor(1);
            name = BB->getName();
            if (name.startswith("kcall.configok") || name.startswith("setup.next")) {
              inst = BB->getFirstNonPHI();
              continue;
            }
            // unrecognized branch
            return nullptr;
          }
        }

        StringRef callee = "";
        if (auto *call = dyn_cast<CallInst>(inst)) {
          callee = call->getCalledFunction()->getName();
          // blacklist helper functions
          if (callee == "cudaSetupArgument" || callee == "cudaConfigureCall"
              || callee.startswith("llvm.lifetime")) {
            inst = inst->getNextNode();
            continue;
          }
          // this either cudaLaunch or a wrapper, break and return
          break;
        }

        // uninteresting, get next
        inst = inst->getNextNode();
        continue;
      }
      return dyn_cast_or_null<CallInst>(inst);
    }

    /** Given a "kernel launch" differentiate whether it is a cudaLaunch or
     * wrapper function call and return the appropriate name.
     */
    StringRef getKernelNameOfLaunch(CallInst *launch) {
      if (launch == nullptr) {
        return "anonymous";
      }

      StringRef calledFunctionName = launch->getCalledFunction()->getName();

      // for kernel launch, return name of first operand
      if (calledFunctionName == "cudaLaunch") {
        auto *op = launch->getArgOperand(0);
        while (auto *cast = dyn_cast<BitCastOperator>(op)) {
          op = cast->getOperand(0);
        }
        return op->getName();
      }

      // otherwise return name of called function itself
      return calledFunctionName;
    }

    /** Updates kernel calls to set up tracing infrastructure on host and device
     * before starting the kernel and tearing everything down afterwards.
     */
    void patchKernelCall(CallInst *configureCall) {
      auto *launch = searchKernelLaunchFor(configureCall);
      assert(launch != nullptr && "did not find kernel launch");

      StringRef kernelName = getKernelNameOfLaunch(launch);
      assert(configureCall->getNumArgOperands() == 6);
      auto *stream = configureCall->getArgOperand(5);

      // insert preparational steps directly after cudaConfigureCall
      // 0. touch consumer to create new one if necessary
      // 1. start/prepare trace consumer for stream
      // 2. get trace consumer info
      // 3. copy trace consumer info to device

      IRBuilder<> IRB(configureCall->getNextNode());

      Type* i8Ty = IRB.getInt8Ty();

      Value* kernelNameVal = IRB.CreateGlobalStringPtr(kernelName);

      // try adding in global symbol + cuda registration
      Module &M = *configureCall->getParent()->getParent()->getParent();

      Type* GlobalVarType = traceInfoTy;
      std::string kernelSymbolName = getSymbolNameForKernel(kernelName);
      GlobalVariable *globalVar = getOrCreateCudaGlobalVar(M, GlobalVarType, kernelSymbolName);

      auto *globalVarPtr = IRB.CreateBitCast(globalVar, IRB.getInt8PtrTy());
      auto* streamPtr = IRB.CreateBitCast(stream, IRB.getInt8PtrTy());

      IRB.CreateCall(TraceTouch, {streamPtr});
      IRB.CreateCall(TraceStart, {streamPtr, kernelNameVal});

      const DataLayout &DL = configureCall->getParent()->getParent()->getParent()->getDataLayout();

      size_t bufSize = DL.getTypeStoreSize(GlobalVarType);

      Value* infoBuf = IRB.CreateAlloca(i8Ty, IRB.getInt32(bufSize));
      IRB.CreateCall(TraceFillInfo, {infoBuf, streamPtr});
      IRB.CreateCall(TraceCopyToSymbol, {streamPtr, globalVarPtr, infoBuf});

      // insert finishing steps after kernel launch was issued
      // 1. stop trace consumer
      IRB.SetInsertPoint(launch->getNextNode());
      IRB.CreateCall(TraceStop, {streamPtr});
    }

    bool runOnModule(Module &M) override {
      bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
      if (isCUDA) return false;

      Function* cudaConfigureCall = M.getFunction("cudaConfigureCall");
      if (cudaConfigureCall == nullptr) {
        return false;
      }

      traceInfoTy = getTraceInfoType(M.getContext());
      findOrInsertRuntimeFunctions(M);

      for (auto* user : cudaConfigureCall->users()) {
        if (auto *call = dyn_cast<CallInst>(user)) {
          patchKernelCall(call);
        }
      }

      return true;
    }
};

char InstrumentHost::ID = 0;

namespace llvm {
  Pass *createInstrumentHostPass() {
    return new InstrumentHost();
  }
}

static RegisterPass<InstrumentHost> X("memtrace-host", "inserts host-side instrumentation for mem-traces", false, false);
