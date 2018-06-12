set(LLVM_NO_RTTI 1)

add_llvm_loadable_module(LLVMMemtrace
  LinkDeviceSupport.cpp
  InstrumentDevice.cpp

  LinkHostSupport.cpp
  InstrumentHost.cpp

  RegisterStandardPasses.cpp
  )
add_dependencies(LLVMMemtrace
  memtrace-device-utils
  memtrace-host-utils
  )

if (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(LLVMMemtrace PUBLIC LLVM)
else ()
  target_link_libraries(LLVMMemtrace PUBLIC
    LLVMSupport
    LLVMCore
    LLVMScalarOpts
    LLVMInstCombine
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMisl
    LLVMMC
    LLVMPasses
    LLVMLinker
    LLVMIRReader
    # The libraries below are required for darwin: http://PR26392
    LLVMBitReader
    LLVMMCParser
    LLVMObject
    LLVMProfileData
    LLVMTarget
    LLVMVectorize
    clangTooling
    )
endif()
