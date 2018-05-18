set(LLVM_NO_RTTI 1)

#target_include_directories(LLVMMemtrace PRIVATE
#  "${LLVM_SOURCE_ROOT}/tools/clang/include"
#  "${LLVM_BINARY_DIR}/tools/clang/include")

add_llvm_loadable_module(LLVMMemtrace
  MemtracePass.cpp
  InstrumentHost.cpp
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
