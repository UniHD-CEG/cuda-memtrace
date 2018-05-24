#find_package(CUDAh)

###############################################################################
## HOST SUPPORT

# compile to bitcode
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
#install(FILES ${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc
#  DESTINATION lib)

# pack into header
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/include/memtrace-host-utils.h"
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../tools/headerize.py
        host_utils "lib/libmemtrace-host.bc" > "include/memtrace-host-utils.h"
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS "${LLVM_BINARY_DIR}/lib/libmemtrace-host.bc"
  VERBATIM
  )
add_custom_target(memtrace-host-utils ALL DEPENDS
  "${LLVM_BINARY_DIR}/include/memtrace-host-utils.h"
  )

###############################################################################
## DEVICE SUPPORT

# compile to bitcode
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc"
  COMMAND bin/clang++
    -std=c++11 -O2
    --cuda-device-only
    --cuda-gpu-arch=sm_30
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
#install(FILES ${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc
#  DESTINATION lib)

# pack into header
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/include/memtrace-device-utils.h"
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../tools/headerize.py
        device_utils "lib/libmemtrace-device.bc" > "include/memtrace-device-utils.h"
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS "${LLVM_BINARY_DIR}/lib/libmemtrace-device.bc"
  VERBATIM
  )
add_custom_target(memtrace-device-utils ALL DEPENDS
  "${LLVM_BINARY_DIR}/include/memtrace-device-utils.h"
  )
