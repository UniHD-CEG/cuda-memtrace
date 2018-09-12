#pragma once

#include "llvm/Pass.h"

namespace llvm {
Pass *createMarkAllDeviceForInlinePass();
Pass *createLinkDeviceSupportPass();
Pass *createInstrumentDevicePass();

Pass *createLinkHostSupportPass();
Pass *createInstrumentHostPass();
};
