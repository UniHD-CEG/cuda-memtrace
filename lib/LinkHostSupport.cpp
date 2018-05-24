#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Linker/Linker.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "memtrace-host-utils.h"

#define DEBUG_TYPE "memtrace-link-host-support"

using namespace llvm;

struct LinkHostSupportPass : public ModulePass {
  static char ID;
  LinkHostSupportPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (isCUDA) return false;

    SMDiagnostic Err;
    LLVMContext &ctx = M.getContext();

    StringRef source = StringRef((const char*)host_utils, sizeof(host_utils));
    auto buf = MemoryBuffer::getMemBuffer(source, "source", false);

    auto utilModule = llvm::parseIR(buf->getMemBufferRef(), Err, ctx);
    if (utilModule.get() == nullptr) {
      errs() << "error: " << Err.getMessage() << "\n";
      report_fatal_error("unable to parse");
    }

    for (auto &global : utilModule->globals()) {
      global.setLinkage(GlobalValue::LinkOnceAnyLinkage);
    }
    Linker::linkModules(M, std::move(utilModule));

    return true;
  }
};
char LinkHostSupportPass::ID = 0;

namespace llvm {
  Pass *createLinkHostSupportPass() {
    return new LinkHostSupportPass();
  }
}

static RegisterPass<LinkHostSupportPass> X("memtrace-link-host-support", "links host support functions into module", false, false);
