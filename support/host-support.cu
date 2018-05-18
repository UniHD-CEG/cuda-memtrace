#include "../lib/Common.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/** Allows to specify base name for traces. The first occurence of
 * "?" is replaced with an ID unique to each stream.
 * Default pattern: "./trace-?.bin"
 */
static std::string traceName(std::string id) {
  const char* pattern_env = getenv("MEMTRACE_PATTERN");
  std::string pattern = pattern_env ? pattern_env : "./trace-?.bin";

  size_t pos = pattern.find("?");
  if (pos != std::string::npos) {
    pattern.replace(pos, 1, id);
  }
  return pattern;
}

/*******************************************************************************
 * TraceConsumer sets up and consumes a queue that can be used by kernels to
 * to write their traces into.
 * Only correct when accessed by a single cuda stream.
 */
class TraceConsumer {
public:
  TraceConsumer(std::string suffix) {
    this->suffix = suffix;

    cudaHostAlloc(&TracesHost, SLOTS_NUM * SLOTS_SIZE * sizeof(uint64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&TracesDevice, TracesHost, 0);

    cudaHostAlloc(&FrontHost, SLOTS_NUM * sizeof(uint32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&FrontDevice, FrontHost, 0);
    memset(FrontHost, 0, SLOTS_NUM * sizeof(uint32_t));

    cudaHostAlloc(&BackHost, SLOTS_NUM * sizeof(uint32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&BackDevice, BackHost, 0);
    memset(BackHost, 0, SLOTS_NUM * sizeof(uint32_t));

    shouldRun = false;
    doesRun = false;

    pipeName = traceName(suffix);

    // char *tmp;
    // asprintf(&tmp, "./trace-%s-%d-%d.gz", kernelName, stream, this->id);
    // gzipName = tmp;
    // free(tmp);

    if (output == nullptr) {
      output = fopen(this->pipeName.c_str(), "wb");
      if (output == nullptr) {
        printf("unable to open trace file '%s' for writing\n", pipeName.c_str());
        abort();
      }

      char recordSize = RECORD_SIZE;
      fwrite(&recordSize, sizeof(char), 1, output);
    }

  }

  virtual ~TraceConsumer() {
    stop();

    fclose(output);

    cudaFree(TracesHost);
    cudaFree(FrontHost);
    cudaFree(BackHost);
  }

  void start(std::string name) {
    //kernelName = name;
    shouldRun = true;

    //    mkfifo(PipeName[S].c_str(), 0666);
    //GzipThread[S] = std::thread([this, S] {
    //    this->GzipPid[S] = getpid();
    //    std::string sys = "gzip -1 < "; 
    //    sys.append(PipeName[S]);
    //    sys.append(" > ");
    //    sys.append(GzipName[S]);
    //    sys.append(" &");
    //    std::system(sys.c_str());
    //    });

    fprintf(output, "\n%s\n", name.c_str()); 
    workerThread = std::thread(consume, this);

    while (!doesRun) {}
  }

  //void start() {
  //  start(kernelName);
  //}

  void stop() {
    shouldRun = false;
    while (doesRun) {}

    char recordSeparator[RECORD_SIZE];
    memset(recordSeparator, 0, RECORD_SIZE);
    fwrite(recordSeparator, 1, RECORD_SIZE, output);

    // reset all buffers and pointers
    memset(TracesHost, 0, SLOTS_NUM * SLOTS_SIZE * sizeof(uint64_t));
    memset(FrontHost, 0, SLOTS_NUM * sizeof(uint32_t));
    memset(BackHost, 0, SLOTS_NUM * sizeof(uint32_t));
  }

  void fillTraceinfo(traceinfo_t *info) {
    info->front = FrontDevice;
    info->back = BackDevice;
    info->slot = TracesDevice;
    info->slot_size = SLOTS_SIZE;
  }

protected:

  static void consume(TraceConsumer *obj) {
    const uint32_t BufferThreshhold  = warpSize * 2;

    obj->doesRun = true;

    while(obj->shouldRun) {
      for(int slot = 0; slot < SLOTS_NUM; slot++) {
        volatile unsigned int *i1 = &(obj->FrontHost[slot]);
        volatile unsigned int *i2 = &(obj->BackHost[slot]);
        if (*i2 >= SLOTS_SIZE - BufferThreshhold) {
          int offset = slot * SLOTS_SIZE;
          unsigned int idx = *i2;
          fwrite(&(obj->TracesHost[offset]), RECORD_SIZE, idx, obj->output);

          *i2=0;
          *i1= 0;
          // just to be safe...
          atomic_thread_fence(std::memory_order::memory_order_seq_cst);
        }
      }
    }
    //clear remainnig Buffers
    for(int slot = 0; slot < SLOTS_NUM; slot++) {
      volatile unsigned int *i2 = &(obj->BackHost[slot]);
      unsigned int idx = *i2;
      int offset = slot * SLOTS_SIZE;
      fwrite(&(obj->TracesHost[offset]), RECORD_SIZE, idx, obj->output);
      *i2=0;
    }

    obj->doesRun = false;
    return;
  }

  std::string suffix;

  std::atomic<bool> shouldRun;
  std::atomic<bool> doesRun;

  //std::thread       gzipThread; // stream and PID
  //pid_t             gzipPid;
  FILE *output;
  std::thread       workerThread;
  std::string       pipeName;
  //std::string       gzipName;

  uint32_t *FrontHost, *FrontDevice;
  uint32_t *BackHost, *BackDevice;
  uint64_t *TracesHost, *TracesDevice;
};

/*******************************************************************************
 * TraceManager acts as a cache for TraceConsumers and ensures only one consumer
 * per stream is exists. RAII on global variable closes files etc.
 */
class TraceManager {
public:
  TraceConsumer *getOrCreateConsumerForStream(cudaStream_t stream) {
    for (auto &consumerPair : consumers) {
      if (consumerPair.first == stream) {
        return consumerPair.second;
      }
    }

    char *suffix;
    asprintf(&suffix, "%d", (int)consumers.size());
    auto newPair = std::make_pair(stream, new TraceConsumer(suffix));
    free(suffix);
    consumers.push_back(newPair);
    return newPair.second;
  }

  virtual ~TraceManager() {
    for (auto &consumerPair : consumers) {
      delete consumerPair.second;
    }
  }
private:
  std::vector<std::pair<cudaStream_t, TraceConsumer*>> consumers;
};

TraceManager __trace_manager;

/*******************************************************************************
 * C Interface
 */

extern "C" {

void __trace_fill_info(const void *info, cudaStream_t stream) {
  auto *consumer = __trace_manager.getOrCreateConsumerForStream(stream);
  consumer->fillTraceinfo((traceinfo_t*) info);
}

void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info) {
  cudaError_t code = cudaMemcpyToSymbolAsync(symbol, info, sizeof(traceinfo_t),
      0, cudaMemcpyHostToDevice, stream);
  if (code != cudaSuccess) {
    printf("Error copying trace info to symbol '%s': %s\n", symbol, cudaGetErrorString(code));
    abort();
  }
}

static void __trace_start_callback(cudaStream_t stream, cudaError_t status, void *vargs);
static void __trace_stop_callback(cudaStream_t stream, cudaError_t status, void *vargs);

void __trace_start(cudaStream_t stream, const char *kernel_name) {
  cudaError_t code = cudaStreamAddCallback(stream, __trace_start_callback,
      (void*)kernel_name, 0);
  if (code != cudaSuccess) {
    printf("Error adding start callback for kernel '%s': %s\n",
        kernel_name, cudaGetErrorString(code));
    abort();
  }
}

void __trace_stop(cudaStream_t stream) {
  cudaError_t code = cudaStreamAddCallback(stream, __trace_stop_callback,
      (void*)nullptr, 0);
  if (code != cudaSuccess) {
    printf("Error adding stop callback: %s\n", cudaGetErrorString(code));
    abort();
  }
}

/***********************************************************
 * private parts of implementation
 */

static void __trace_start_callback(cudaStream_t stream, cudaError_t status, void *vargs) {
  auto *consumer = __trace_manager.getOrCreateConsumerForStream(stream);
  consumer->start((const char*)vargs);
}

static void __trace_stop_callback(cudaStream_t stream, cudaError_t status, void *vargs) {
  auto *consumer = __trace_manager.getOrCreateConsumerForStream(stream);
  consumer->stop();
}


}
