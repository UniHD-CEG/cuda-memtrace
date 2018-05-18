#include "../lib/Common.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define always_assert(cond) do {\
  if (!(cond)) {\
    printf("assertion failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);\
    abort();\
  }\
} while(0)

#define cudaChecked(code) do {\
  cudaError_t err = code;\
  if (err != cudaSuccess) {\
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
        cudaGetErrorString(err));\
    abort();\
  }\
} while(0)

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
 * Usage must follow a strict protocol:
 * - one call to TraceConsumer()
 * - zero or more calls to start() ALWAYS followed by stop()
 * - one call to ~TraceConsumer()
 * Trying to repeatedly start or stop a consumer results in process termination.
 */
class TraceConsumer {
public:
  TraceConsumer(std::string suffix) {
    this->suffix = suffix;

    cudaChecked(cudaHostAlloc(&TracesHost, SLOTS_NUM * SLOTS_SIZE * sizeof(uint64_t), cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&TracesDevice, TracesHost, 0));

    cudaChecked(cudaHostAlloc(&FrontHost, SLOTS_NUM * sizeof(uint32_t), cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&FrontDevice, FrontHost, 0));
    memset(FrontHost, 0, SLOTS_NUM * sizeof(uint32_t));

    cudaChecked(cudaHostAlloc(&BackHost, SLOTS_NUM * sizeof(uint32_t), cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&BackDevice, BackHost, 0));
    memset(BackHost, 0, SLOTS_NUM * sizeof(uint32_t));

    shouldRun = false;
    doesRun = false;

    pipeName = traceName(suffix);

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
    always_assert(!shouldRun);
    fclose(output);

    cudaFreeHost(TracesHost);
    cudaFreeHost(FrontHost);
    cudaFreeHost(BackHost);
  }

  void start(std::string name) {
    always_assert(!shouldRun);
    shouldRun = true;

    fprintf(output, "\n%s\n", name.c_str()); 
    workerThread = std::thread(consume, this);

    while (!doesRun) {}
  }

  void stop() {
    always_assert(shouldRun);
    shouldRun = false;
    while (doesRun) {}
    workerThread.join();

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

  FILE *output;
  std::thread       workerThread;
  std::string       pipeName;

  uint32_t *FrontHost, *FrontDevice;
  uint32_t *BackHost, *BackDevice;
  uint64_t *TracesHost, *TracesDevice;
};

/*******************************************************************************
 * TraceManager acts as a cache for TraceConsumers and ensures only one consumer
 * per stream is exists. RAII on global variable closes files etc.
 * CUDA API calls not allowed inside of stream callback, so TraceConsumer
 * initialization must be performed explicitly;
 */
class TraceManager {
public:
  /** Creates a new consumer for a stream if necessary. Returns true if a new
   * consumer had to be created, false otherwise.
   */
  bool touchConsumer(cudaStream_t stream) {
    for (auto &consumerPair : consumers) {
      if (consumerPair.first == stream) {
        return false;
      }
    }

    char *suffix;
    asprintf(&suffix, "%d", (int)consumers.size());
    auto newPair = std::make_pair(stream, new TraceConsumer(suffix));
    free(suffix);
    consumers.push_back(newPair);
    return true;
  }

  /** Return *already initialized* TraceConsumer for a stream. Aborts application
    * if stream is not initialized.
   */
  TraceConsumer *getConsumer(cudaStream_t stream) {
    for (auto &consumerPair : consumers) {
      if (consumerPair.first == stream) {
        return consumerPair.second;
      }
    }
    always_assert(0 && "trying to get non-existent consumer");
    return nullptr;
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
  auto *consumer = __trace_manager.getConsumer(stream);
  consumer->fillTraceinfo((traceinfo_t*) info);
}

void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info) {
  cudaChecked(cudaMemcpyToSymbolAsync(symbol, info, sizeof(traceinfo_t),
      0, cudaMemcpyHostToDevice, stream));
}

static void __trace_start_callback(cudaStream_t stream, cudaError_t status, void *vargs);
static void __trace_stop_callback(cudaStream_t stream, cudaError_t status, void *vargs);

void __trace_touch(cudaStream_t stream) {
  __trace_manager.touchConsumer(stream);
}

void __trace_start(cudaStream_t stream, const char *kernel_name) {
  cudaChecked(cudaStreamAddCallback(stream,
        __trace_start_callback, (void*)kernel_name, 0));
}

void __trace_stop(cudaStream_t stream) {
  cudaChecked(cudaStreamAddCallback(stream,
        __trace_stop_callback, (void*)nullptr, 0));
}

/***********************************************************
 * private parts of implementation
 */

static void __trace_start_callback(cudaStream_t stream, cudaError_t status, void *vargs) {
  auto *consumer = __trace_manager.getConsumer(stream);
  consumer->start((const char*)vargs);
}

static void __trace_stop_callback(cudaStream_t stream, cudaError_t status, void *vargs) {
  auto *consumer = __trace_manager.getConsumer(stream);
  consumer->stop();
}

}
