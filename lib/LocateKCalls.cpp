#include "LocateKCalls.h"

#include "llvm/IR/Instructions.h"


#define DEBUG_TYPE "memtrace-locate-kernel-launches"

using namespace llvm;

SmallVector<CallInst*, 4> findConfigureCalls(Module &M) {
  Function* F = M.getFunction("cudaConfigureCall");
  if (F == nullptr) {
    return {};
  }

  SmallVector<CallInst*, 4> R;
  for (auto *user : F->users()) {
    auto *CI = dyn_cast<CallInst>(user);
    if (CI != nullptr) {
      R.push_back(CI);
    }
  }
  return R;
}

Instruction* findLaunchFor(CallInst* configureCall) {
  auto* Terminator = configureCall->getParent()->getTerminator();
  auto* Br = dyn_cast<BranchInst>(Terminator);
  if (Br == nullptr) {
    errs() << "configureCall not followed by kcall.configok\n";
    return nullptr;
  }
  // follow to "kcall.configok" block
  BasicBlock *candidate = nullptr;
  for (auto *successor : Br->successors()) {
    if (successor->getName().startswith("kcall.configok")) {
      candidate = successor;
      break;
    }
  }
  if (candidate == nullptr) {
    errs() << "configureCall not followed by kcall.configok\n";
    return nullptr;
  }
  // find first block NOT followed by a "setup.next*" block
  while (true) {
    Terminator = candidate->getTerminator();
    Br = dyn_cast<BranchInst>(Terminator);
    if (Br == nullptr) break;
    BasicBlock *next = nullptr;
    for (auto *successor : Br->successors()) {
      if (successor->getName().startswith("setup.next")) {
        next = successor;
        break;
      }
    }
    if (next == nullptr) break;
    candidate = next;
  }

  Instruction* launch = nullptr;
  for (auto it = candidate->rbegin(); it != candidate->rend(); ++it) {
    if (isa<CallInst>(*it) || isa<InvokeInst>(*it)) {
      launch = &(*it);
      break;
    }
  }
  if (launch == nullptr) {
    errs() << "no launch found for configure call\n";
  }

  return launch;
}

std::string getKernelNameOf(Instruction* launch) {
  Function* callee = nullptr;
  Value *op1 = nullptr;
  CallInst *CI = dyn_cast<CallInst>(launch);
  if (CI != nullptr) {
    callee = CI->getCalledFunction();
    if (CI->getNumArgOperands() > 0) {
      op1 = CI->getArgOperand(0);
    }
  } else {
    InvokeInst *II = dyn_cast<InvokeInst>(launch);
    if (II != nullptr) {
      callee = II->getCalledFunction();
      if (II->getNumArgOperands() > 0) {
        op1 = II->getArgOperand(0);
      }
    } else {
      return "";
    }
  }
  if (callee->hasName() && callee->getName() != "cudaLaunch") {
    return callee->getName();
  } else {
    op1 = op1->stripPointerCasts();
    callee = dyn_cast<Function>(op1);
    if (callee != nullptr && callee->hasName()) {
      return callee->getName();
    }
  }
  return "";
}


namespace llvm {

  LocateKCallsPass::LocateKCallsPass() : ModulePass(ID) {}

  bool LocateKCallsPass::runOnModule(Module &M) {
    launches.clear();
    for (auto *configure : findConfigureCalls(M)) {
      Instruction* launch = findLaunchFor(configure);
      std::string name = getKernelNameOf(launch);
      launches.push_back(KCall(configure, launch, name));
    }
    return false;
  }

  void LocateKCallsPass::releaseMemory() {
    launches.clear();
  }

  SmallVector<KCall, 4> LocateKCallsPass::getLaunches() const {
    return launches;
  }

  void LocateKCallsPass::print(raw_ostream &O, const Module *M) const {
    for (const auto &launch : launches) {
      O << "\n";
      O << "name:   " << launch.kernelName << "\n";
      O << "config: " << *launch.configureCall << "\n";
      if (launch.kernelLaunch != nullptr) {
        O << "launch: " << *launch.kernelLaunch << "\n";
      } else {
        O << "launch: (nullptr)\n";
      }
    }
  }

  char LocateKCallsPass::ID = 0;

  Pass *createLocateKCallsPass() {
    return new LocateKCallsPass();
  }

}

static RegisterPass<LocateKCallsPass>
  X("memtrace-locate-kcalls", "locate kernel launches", false, false);
