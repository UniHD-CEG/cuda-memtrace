#ifndef __LOCATE_KCALLS_H
#define __LOCATE_KCALLS_H

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

struct KCall {
  KCall(CallInst *cc, Instruction *kl, StringRef kn)
    : configureCall(cc), kernelLaunch(kl), kernelName(kn)
  {}
  CallInst* configureCall;
  Instruction* kernelLaunch;
  std::string kernelName;
};

class LocateKCallsPass : public ModulePass {
public:
  static char ID;
  LocateKCallsPass();
  bool runOnModule(Module &M) override;
  void releaseMemory() override;
  SmallVector<KCall, 4> getLaunches() const;
  void print(raw_ostream &O, const Module *M) const override;
private:
  SmallVector<KCall, 4> launches;
};

}

#endif
