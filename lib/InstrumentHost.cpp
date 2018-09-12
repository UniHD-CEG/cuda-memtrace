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
#include "llvm/IR/InstIterator.h"

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

GlobalVariable* getOrCreateGlobalVar(Module &M, Type* T, const Twine &name) {
  // first see if the variable already exists
  GlobalVariable *Global = M.getGlobalVariable(name.getSingleStringRef());
  if (Global) {
    return Global;
  }

  // Variable does not exist, so we create one and register it.
  // This happens if a kernel is called in a module it is not registered in.
  Constant *zero = Constant::getNullValue(T);
  Global = new GlobalVariable(M, T, false, GlobalValue::LinkOnceAnyLinkage, zero, name);
  Global->setAlignment(8);
  assert(Global != nullptr);
  return Global;
}

void RegisterVars(Function *CudaSetup, ArrayRef<GlobalVariable*> Variables) {
  Module &M = *CudaSetup->getParent();
  IRBuilder<> IRB(M.getContext());

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

  for (auto *Global : Variables) {

    auto *GlobalNameLiteral = IRB.CreateGlobalString(Global->getName());
    auto *GlobalName = IRB.CreateBitCast(GlobalNameLiteral, charPtrTy);
    auto *GlobalAddress = IRB.CreateBitCast(Global, charPtrTy);
    uint64_t GlobalSize = M.getDataLayout().getTypeStoreSize(Global->getType());
    Value *CubinHandle = &*CudaSetup->arg_begin();

    //createPrintf(IRB, "registering... symbol name: %s, symbol address: %p, name address: %p\n",
    //    {GlobalName, GlobalAddress, GlobalName});
    //errs() << "registering device symbol " << name << "\n";

    IRB.CreateCall(Fn, {CubinHandle, GlobalAddress, GlobalName, GlobalName,
        IRB.getInt32(0), IRB.getInt32(GlobalSize), IRB.getInt32(0), IRB.getInt32(0)});
  }
}


void createAndRegisterTraceVars(Function* CudaSetup, Type* VarType) {
  Module &M = *CudaSetup->getParent();
  //SmallVector<Function*, 8> registeredKernels;
  SmallVector<GlobalVariable*, 8> globalVars;
  for (Instruction &inst : instructions(CudaSetup)) {
    auto *call = dyn_cast<CallInst>(&inst);
    if (call == nullptr) {
      continue;
    }
    auto *callee = call->getCalledFunction();
    if (!callee || callee->getName() != "__cudaRegisterFunction") {
      continue;
    }

    // 0: ptx image, 1: wrapper, 2: name, 3: name again, 4+: ?
    auto *wrapperVal = call->getOperand(1)->stripPointerCasts();
    assert(wrapperVal != nullptr && "__cudaRegisterFunction called without wrapper");
    auto *wrapper = dyn_cast<Function>(wrapperVal);
    assert(wrapper != nullptr && "__cudaRegisterFunction called with something other than a wrapper");

    StringRef kernelName = wrapper->getName();
    std::string varName = getSymbolNameForKernel(kernelName);
    GlobalVariable *globalVar = getOrCreateGlobalVar(M, VarType, varName);
    globalVars.push_back(globalVar);
  }

  RegisterVars(CudaSetup, globalVars);
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
      SmallVector<Instruction*, 8> stack;
      stack.push_back(inst);
      while (stack.size() > 0) {
        Instruction *curr = stack.back();
        stack.pop_back();
        if (curr == nullptr) {
          continue;
        }

        if (auto *br = dyn_cast<BranchInst>(curr)) {
          if (br->isUnconditional()) {
            stack.push_back(br->getSuccessor(0)->getFirstNonPHI());
          } else {
            // if this is a branch, try to find a 'config went ok' branch and
            // follow that one, otherwise just skip the branch.
            int numSuccessors = br->getNumSuccessors();
            for (int i = 0; i < numSuccessors; ++i) {
              auto *BB = br->getSuccessor(i);
              StringRef name = BB->getName();
              if (name.startswith("kcall.configok") || name.startswith("setup.next")) {
                stack.push_back(BB->getFirstNonPHI());
                break;
              }
            }
          }
        } else if (auto *call = dyn_cast<CallInst>(curr)) {
          // if this is a call, it gets interesting, use a heuristic to figure out
          // whether this is a cudaConfigure Call
          auto callee = call->getCalledValue();
          if (callee == nullptr) {
            report_fatal_error("non-function callee (e.g. function pointer)");
          }
          auto calleeName = callee->getName();
          // blacklist helper functions
          if (calleeName == "cudaSetupArgument" || calleeName == "cudaConfigureCall"
              || calleeName.startswith("llvm.lifetime")) {
            stack.push_back(curr->getNextNode());
          } else {
            return dyn_cast<CallInst>(curr);
          }
        } else {
          // uninteresting, get next
          stack.push_back(curr->getNextNode());
        }
      }
      return nullptr;
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

      GlobalVariable *globalVar = getOrCreateGlobalVar(M, GlobalVarType, kernelSymbolName);

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

      traceInfoTy = getTraceInfoType(M.getContext());

      // register global variables for trace info for all kernels registered
      // in this module
      Function* CudaSetup = M.getFunction("__cuda_register_globals");
      if (CudaSetup != nullptr) {
        createAndRegisterTraceVars(CudaSetup, traceInfoTy);
      }

      // add instrumentation for all kernels called in this module
      Function* cudaConfigureCall = M.getFunction("cudaConfigureCall");
      if (cudaConfigureCall == nullptr) {
        return false;
      }

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
