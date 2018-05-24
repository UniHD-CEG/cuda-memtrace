#pragma once

#include "llvm/Pass.h"

namespace llvm {
Pass *createLinkDeviceSupportPass();
Pass *createInstrumentDevicePass();

Pass *createLinkHostSupportPass();
Pass *createInstrumentHostPass();
};
