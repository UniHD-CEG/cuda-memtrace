#find_package(CUDAh)

add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc"
  COMMAND bin/clang++
    -std=c++11 -O2
    --cuda-host-only
    -c -emit-llvm
    -o lib/libmemtrace-host.bc
    "${CMAKE_CURRENT_SOURCE_DIR}/host-support.cu"
  
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS host-support.cu ../lib/Common.h clang
  VERBATIM
  )
add_custom_target(libmemtrace-host ALL DEPENDS
  "${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc"
  )
install(FILES ${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc
  DESTINATION lib)

add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc"
  COMMAND bin/clang++
    -std=c++11 -O2
    --cuda-device-only
    --cuda-gpu-arch=sm_50
    -c -emit-llvm
    -o lib/libmemtrace-device.bc
    "${CMAKE_CURRENT_SOURCE_DIR}/device-support.cu"
  
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS device-support.cu ../lib/Common.h clang
  VERBATIM
  )
add_custom_target(libmemtrace-device ALL DEPENDS
  "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc"
  )
install(FILES ${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc
  DESTINATION lib)
