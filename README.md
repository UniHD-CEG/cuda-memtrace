# CUDA Memtrace

This llvm plugin instruments CUDA code such that all memory accesses to global memory by a kernel are traced on a per-stream basis.
Traces are stored in a simple run-length encoded binary format, for which we
provide an io utils in a header only library.

# Usage

In order to add tracing to your CUDA application, you can compile CUDA applications
using clang as described in the
(official manual)[https://prereleases.llvm.org/7.0.0/rc2/docs/CompileCudaWithLLVM.html]
and two additional flags.
Let `$BASE` be a variable containing the base path of your llvm installation (so clang
can be found as `$BASE/bin/clang`), then the required flags are:

1. `-fplugin=$BASE/lib/LLVMMemtrace` for the compilation of every `.cu` file. This
	instruments host and device code to emit traces.
2. `$BASE/lib/libmemtrace-host.o` when linking your application. This links the host
	side runtime (receiving traces from devices and writing them to disk) into
	your application.

Afterwards, just run your application.
Traces are written to files named `<your aplication>-<CUDA stream number>.trc`.
One file is created per stream.

You can take a quick look at your traces by using the tool `$BASE/bin/cutracedump`.
Its source code can also be used as a reference for your own analysis tools.

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
