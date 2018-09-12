#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "memtrace-mark-device-for-inline"

using namespace llvm;

struct MarkAllDeviceForInlinePass : public ModulePass {
  static char ID;
  MarkAllDeviceForInlinePass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (!isCUDA) return false;

    for (Function &F : M) {
      if (F.isIntrinsic()) continue;
      F.addFnAttr(Attribute::AttrKind::AlwaysInline);
    }

    return true;
  }
};
char MarkAllDeviceForInlinePass::ID = 0;

namespace llvm {
  Pass *createMarkAllDeviceForInlinePass() {
    return new MarkAllDeviceForInlinePass();
  }
}

static RegisterPass<MarkAllDeviceForInlinePass> X("memtrace-mark-inline", "marks all functions for inlining", false, false);
