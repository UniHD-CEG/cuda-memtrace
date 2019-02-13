CUDAFLAGS := --cuda-path=/opt/cuda-8.0 --cuda-gpu-arch=sm_30
OPTFLAGS += -load /local/amatz/memtrace/build/lib/LLVMMemtrace.so

all: launches.O0.ll launches.O3.ll \
	launches.host.O0.ll launches.host.O3.ll

launches.%.ll: launches.cu
	clang++ $(CUDAFLAGS) --save-temps -$* -c $<
	llvm-dis launches-host-x86_64-unknown-linux-gnu.bc
	mv launches-host-x86_64-unknown-linux-gnu.ll $@
	rm launches-* launches.cu-* launches.o

launches.host.%.ll: launches.cu
	clang++ $(CUDAFLAGS) --cuda-host-only -$* -S -emit-llvm $< -o $@
