#include "Passes.h"

#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

using namespace llvm;

// This enables Autoregistration of the Pass
static void registerStandardPasses(const PassManagerBuilder &,legacy::PassManagerBase &PM) {
    PM.add(createMarkAllDeviceForInlinePass());
    PM.add(createAlwaysInlinerLegacyPass());
    PM.add(createLinkDeviceSupportPass());
    PM.add(createInstrumentDevicePass());

    //PM.add(createLinkHostSupportPass());
    PM.add(createInstrumentHostPass());
}

static RegisterStandardPasses RegisterTracePass(
    PassManagerBuilder::EP_ModuleOptimizerEarly, registerStandardPasses);
static RegisterStandardPasses RegisterTracePass0(
    PassManagerBuilder::EP_EnabledOnOptLevel0, registerStandardPasses);
