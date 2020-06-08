# CUDA Memtrace

This llvm plugin instruments CUDA code such that all memory accesses to global memory by a kernel are traced on a per-stream basis.
Traces are stored in a simple run-length encoded binary format, for which we
provide an io utils in a header only library.

# Usage

In order to add tracing to your CUDA application, you can compile CUDA applications
using clang as described in the
[official manual](https://prereleases.llvm.org/7.0.0/rc2/docs/CompileCudaWithLLVM.html)
and two additional flags (using at least `-O1` might be necessary for some applications).
Let `$BASE` be a variable containing the base path of your llvm installation (so clang
can be found as `$BASE/bin/clang`), then the required flags are:

1. `-fplugin=$BASE/lib/LLVMMemtrace.so` for the compilation of every `.cu` file. This
	instruments host and device code to emit traces. E.g.
    `$ clang++ -fplugin=$BASE/lib/LLVMMemtrace.so --cuda-path=... --cuda-gpu-arch=sm_30 -O1 -c -o code.o code.cu`
2. `$BASE/lib/libmemtrace-host.o` when linking your application. This links the host
	side runtime (receiving traces from devices and writing them to disk) into
	your application. E.g.
    `$ clang++ -o application code.o $BASE/lib/libmemtrace-host.o`

Afterwards, just run your application.
Traces are written to files named `<your application>-<CUDA stream number>.trc`.
One file is created per stream.

You can take a quick look at your traces by using the tool `$BASE/bin/cutracedump`.
Its source code can also be used as a reference for your own analysis tools.

# Compatibility

The memtrace was developed and tested against LLVM commit `a5b9a59`, Clang
commit `f3e3e06` and CUDA SDK 8.
It has been superficially tested against the LLVM+Clang release 7.0, we expect
incompatibilietes when building against different version due to frequent API
changes.
The monorepo mirror on github likely results in dysfunctional builds, so we
recommend using the multirepo mirror on `https://github.com/llvm-mirror/{repo}`.
The cause of this issue has not been determined yet.

# Building

The Memtracer is an external project to LLVM (like Clang) that lives in the 
`tools` directory of the llvm tree (also like Clang).
The build process from the official 
[LLVM 7 Getting Started Guide](https://prereleases.llvm.org/7.0.0/rc2/docs/GettingStarted.html)
is staying the same with some requirements to the build.

First, download and checkout llvm, clang, and the memtracer e.g.:

```bash
$ cd where-you-want-llvm-to-live
$ git clone http://github.com/llvm-mirror/llvm
#... downloading out llvm repository
$ cd llvm
$ git checkout release_70
#... switching to release 7.0
$ cd tools # llvm/tools
$ git clone http://github.com/llvm-mirror/clang
#... downloading clang
$ cd clang # llvm/tools/clang
$ git checkout release_70
#... switching to release 7.0

# THE FOLLOWING PART IS NEW
$ cd .. # llvm/tools
$ git clone github.com/unihd-ceg/cuda-memtrace
#... downloading cuda-memtrace
```

Next, add an entry to `llvm/tools/CMakeLists.txt` to make CMake aware of the
new external project and include it in the build.
We typically add it after the block containing the other tools, as in the
following diff:

```diff
diff --git a/tools/CMakeLists.txt b/tools/CMakeLists.txt                                 
index b654b8c..7eef359 100644
--- a/tools/CMakeLists.txt
+++ b/tools/CMakeLists.txt
@@ -46,6 +46,7 @@ add_llvm_external_project(clang)                                       
 add_llvm_external_project(llgo)
 add_llvm_external_project(lld)
 add_llvm_external_project(lldb)
+add_llvm_external_project(cuda-memtrace)

 # Automatically add remaining sub-directories containing a 'CMakeLists.txt'             
 # file as external projects.
```

Lastly, configure and build LLVM.
The configuration requires the following flags:

- `-DBUILD_SHARED_LIBS=ON` - the memtracer is implemented as a plugin, which
  currently does not support static builds of LLVM (linker error message:
  duplicate symbols).
- `-DLLVM_ENABLE_ASSERTIONS=ON` - The current analysis pass to locate kernel
  launch sites relies on the basic block labels set by gpucc. Value/BB labels
  are only set in +Assert builds (or with the `-fno-discard-value-names` clang
  flag), so instrumentation fails with disabled assertions (which is the
  default).
- `-DMEMTRACE_CUDA_FLAGS=${PATH TO YOUR CUDA INSTALLATION}` - required if your
  CUDA 8.0 installation is located somewhere other than `/usr/local/cuda` (e.g.
  `/opt/cuda-8.0`).

The resulting LLVM build includes the memtracer and can be used as described
above.

# Software Authors

- Alexander Matz
- Dennis Rieber
- Georg Weisert
