#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Linker/Linker.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "memtrace-device-utils.h"

#define DEBUG_TYPE "memtrace-link-device-support"

using namespace llvm;

struct LinkDeviceSupportPass : public ModulePass {
  static char ID;
  LinkDeviceSupportPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (!isCUDA) return false;

    SMDiagnostic Err;
    LLVMContext &ctx = M.getContext();

    StringRef source = StringRef((const char*)device_utils, sizeof(device_utils));
    auto buf = MemoryBuffer::getMemBuffer(source, "source", false);

    auto utilModule = llvm::parseIR(buf->getMemBufferRef(), Err, ctx);
    if (utilModule.get() == nullptr) {
      errs() << "error: " << Err.getMessage() << "\n";
      report_fatal_error("unable to parse");
    }
    Linker::linkModules(M, std::move(utilModule));

    return true;
  }
};
char LinkDeviceSupportPass::ID = 0;

namespace llvm {
  Pass *createLinkDeviceSupportPass() {
    return new LinkDeviceSupportPass();
  }
}

static RegisterPass<LinkDeviceSupportPass> X("memtrace-link-device-support", "links device support functions into module", false, false);
