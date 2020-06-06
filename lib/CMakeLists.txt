set(LLVM_NO_RTTI 1)

add_llvm_library(LLVMMemtrace MODULE
  MarkAllDeviceForInline.cpp
  LinkDeviceSupport.cpp
  InstrumentDevice.cpp
  LocateKCalls.cpp

  InstrumentHost.cpp

  RegisterStandardPasses.cpp

  PLUGIN_TOOL
  opt
  )
add_dependencies(LLVMMemtrace
  memtrace-device-utils
  libmemtrace-host-o
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

install(FILES ${MEMTRACE_SOURCE_DIR}/lib/cutrace_io.h
  DESTINATION include)
