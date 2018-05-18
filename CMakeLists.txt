if (NOT DEFINED LLVM_MAIN_SRC_DIR)
  project(Memtrace)
  cmake_minimum_required(VERSION 3.4.3)
  find_package(LLVM CONFIG REQUIRED)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})
  include(HandleLLVMOptions)
  include(AddLLVM)
  include_directories(${LLVM_INCLUDE_DIRS})

  # Sources available, too?
  if (LLVM_BUILD_MAIN_SRC_DIR)
    set(LLVM_SOURCE_ROOT ${LLVM_BUILD_MAIN_SRC_DIR} CACHE PATH
      "Path to LLVM source tree")
  else()
    execute_process(COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-config" --src-root
      OUTPUT_VARIABLE MAIN_SRC_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(LLVM_SOURCE_ROOT ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
  endif()

   # Make sure the isl c files are built as fPIC
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

  # Set directory for polly-isl-test.
else ()
  set(LLVM_SOURCE_ROOT "${LLVM_MAIN_SRC_DIR}")
endif ()

set(MEKONG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MEKONG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


# Add appropriate flags for GCC
if (CMAKE_COMPILER_IS_GNUCXX)
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-exceptions -fno-rtti")
elseif (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
  add_definitions("-D_HAS_EXCEPTIONS=0")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
endif ()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

add_definitions( -D_GNU_SOURCE )
add_subdirectory(lib)
add_subdirectory(support)
add_subdirectory(tools)
