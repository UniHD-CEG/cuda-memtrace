# CUDA Memtrace

This llvm plugin instruments CUDA code such that all memory accesses to global memory by a kernel are traced on a per-stream basis.
Traces are stored in a simple run-length encoded binary format, for which we
provide an io utils in a header only library.

# Usage

In order to add tracing to your CUDA application, you can compile CUDA applications
using clang as described in the
(official manual)[https://prereleases.llvm.org/7.0.0/rc2/docs/CompileCudaWithLLVM.html]
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
It should work with a release LLVM+Clang of version 7.x.x, but problems when
building against other major versions of LLVM/Clang are expected due to the
quick development cycles and API changes of LLVM.

# Building

The Memtracer is an external project to LLVM, so for the most part follow the
official
(LLVM 7 Getting Started Guide)[https://prereleases.llvm.org/7.0.0/rc2/docs/GettingStarted.html].
except for these changes:

before running cmake, clone mekong into your llvm/tools directory, just
like clang, e.g.:

```
$ cd where-you-want-llvm-to-live
$ cd llvm/tools
$ git clone github.com/unihd-ceg/cuda-memtrace
```

And then include the project into the building process by adding an
`add_llvm_external_project` directo to `tools/CMakeLists.txt`.

```
$ cd where-you-want-llvm-to-live
$ cd llvm
$ echo 'add_llvm_external_project(cuda-memtrace)' >> tools/CMakeLists.txt
```

Now you should be able to compile and install llvm as usual.

# Software Authors

- Alexander Matz
- Dennis Rieber
