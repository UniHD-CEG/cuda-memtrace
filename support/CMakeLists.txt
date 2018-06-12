#find_package(CUDAh)

###############################################################################
## HOST SUPPORT

set(MEMTRACE_CUDA_FLAGS "-O2" CACHE STRING "clang-cuda flags used to compile memtrace runtime")
message(STATUS "Using MEMTRACE_CUDA_FLAGS:" ${MEMTRACE_CUDA_FLAGS})

# compile to bitcode
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc"
  COMMAND bin/clang++
    -std=c++11
    --cuda-host-only
    -c -emit-llvm -O2
    ${MEMTRACE_CUDA_FLAGS}
    -o lib/libmemtrace-host.bc
    "${CMAKE_CURRENT_SOURCE_DIR}/host-support.cu"
  
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS host-support.cu ../lib/Common.h clang
  VERBATIM
  )
add_custom_target(libmemtrace-host DEPENDS
  "${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc"
  )
#install(FILES ${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc
#  DESTINATION lib)

# pack into header
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/include/memtrace-host-utils.h"
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../tools/headerize.py
        host_utils "lib/libmemtrace-host.bc" > "include/memtrace-host-utils.h"
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS "${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc" libmemtrace-host
  VERBATIM
  )
add_custom_target(memtrace-host-utils DEPENDS
  "${LLVM_BINARY_DIR}/include/memtrace-host-utils.h"
  )

###############################################################################
## DEVICE SUPPORT

# compile to bitcode
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc"
  COMMAND bin/clang++
    --cuda-device-only
    -c -emit-llvm
    --cuda-gpu-arch=sm_30
    ${MEMTRACE_CUDA_FLAGS}
    -o lib/libmemtrace-device.bc
    "${CMAKE_CURRENT_SOURCE_DIR}/device-support.cu"
  
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS device-support.cu ../lib/Common.h clang
  VERBATIM
  )
add_custom_target(libmemtrace-device DEPENDS
  "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc"
  )
#install(FILES ${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc
#  DESTINATION lib)

# pack into header
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/include/memtrace-device-utils.h"
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../tools/headerize.py
        device_utils "lib/libmemtrace-device.bc" > "include/memtrace-device-utils.h"
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc" libmemtrace-device
  VERBATIM
  )
add_custom_target(memtrace-device-utils DEPENDS
  "${LLVM_BINARY_DIR}/include/memtrace-device-utils.h"
  )
