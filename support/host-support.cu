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
 *
 * The queue is a multiple producer, single consumer key. Circular queues do not
 * work as expected because we cannot reliably update the in-pointer with a single
 * atomic operation. The result would be corrupted data as the host begins reading
 * data that is falsely assumed to have been committed.
 *
 * Instead we use buffers that are alternatingly filled up by the GPU and cleared
 * out by the CPU.
 * Two pointers are associated with each buffer, an allocation and a commit pointer.
 * A GPU warp first allocates spaces in the buffer using an atomic add on the
 * allocation pointer, then writes its data and increases the commit buffer by the
 * same amount, again using atomic add.
 * The buffered is considered full 
 * a) by the GPU if the allocation pointer is within 32 elements of capacity, and
 * b) by the host if the commit pointer is within 32 elements of capacity.
 * When the buffer is full, all elements are read by the host and the commit and
 * allocation buffer are reset to 0 in this order.
 * 
 * Since a maximum of 1 warp is writing some of the last 32 elements, the commit
 * pointer pointing in this area signals that all warps have written their data.
 * 
 * Several buffers, called "slots", exist in order to reduce contention.
 *
 * All pointers have a dedicated cache lane to avoid cache thrashing.
 */
class TraceConsumer {
public:
  TraceConsumer(std::string suffix) {
    this->suffix = suffix;

    cudaChecked(cudaHostAlloc(&RecordsHost, SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE, cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&RecordsDevice, RecordsHost, 0));

    cudaChecked(cudaHostAlloc(&AllocsHost, SLOTS_NUM * CACHELINE, cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&AllocsDevice, AllocsHost, 0));
    memset(AllocsHost, 0, SLOTS_NUM * CACHELINE);

    cudaChecked(cudaHostAlloc(&CommitsHost, SLOTS_NUM * CACHELINE, cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&CommitsDevice, CommitsHost, 0));
    memset(CommitsHost, 0, SLOTS_NUM * CACHELINE);

    shouldRun = false;
    doesRun = false;

    pipeName = traceName(suffix);

    if (output == nullptr) {
      output = fopen(this->pipeName.c_str(), "wb");
      if (output == nullptr) {
        printf("unable to open trace file '%s' for writing\n", pipeName.c_str());
        abort();
      }

      fputc(0x19, output);
      fputs("CUDATRACE", output);
    }

  }

  virtual ~TraceConsumer() {
    always_assert(!shouldRun);
    fclose(output);

    cudaFreeHost(RecordsHost);
    cudaFreeHost(AllocsHost);
    cudaFreeHost(CommitsHost);
  }

  void start(std::string name) {
    always_assert(!shouldRun);
    shouldRun = true;

    // reset all buffers and pointers
    memset(AllocsHost, 0, SLOTS_NUM * sizeof(uint32_t));
    memset(CommitsHost, 0, SLOTS_NUM * sizeof(uint32_t));
    // just for testing purposes
    memset(RecordsHost, 0, SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE);

    uint16_t nameSize = name.size();
    fputc(0x00, output);
    fwrite(&nameSize, 2, 1, output);
    fwrite(name.c_str(), nameSize, 1, output);

    workerThread = std::thread(consume, this);

    while (!doesRun) {}
  }

  void stop() {
    always_assert(shouldRun);
    shouldRun = false;
    while (doesRun) {}
    workerThread.join();
  }

  void fillTraceinfo(traceinfo_t *info) {
    info->allocs = AllocsDevice;
    info->commits = CommitsDevice;
    info->records = RecordsDevice;
    info->slot_size = SLOTS_SIZE;
  }

protected:

  // clear up a slot if it is full
  static void consumeSlot(uint32_t *allocPtr, uint32_t *commitPtr, uint8_t *recordsPtr,
      FILE* out, bool onlyFull) {
    // commits is written by threads on the GPU, so we need it volatile
    volatile uint32_t *vcommit = commitPtr;

    // in kernel is still active we only want to read full slots
    if (onlyFull && *vcommit <= SLOTS_SIZE - 32) {
      return;
    }

    // we know everything stopped, so we avoid using the volatile reference
    // in the end condition
    uint32_t numRecords = *vcommit;
    for (int32_t i = 0; i < numRecords; ++i) {
      fputc(0xff, out);
      fwrite(&recordsPtr[i * RECORD_SIZE], RECORD_SIZE, 1, out);
    }

    // write commits 
    *commitPtr = 0;
    // ensure commits are reset first
    std::atomic_thread_fence(std::memory_order_release);
    *allocPtr = 0;
  }

  // payload function of queue consumer
  static void consume(TraceConsumer *obj) {
    obj->doesRun = true;

    uint8_t *records = obj->RecordsHost;
    uint32_t *allocs = obj->AllocsHost;
    uint32_t *commits = obj->CommitsHost;

    FILE* sink = obj->output;

    while(obj->shouldRun) {
      for(int slot = 0; slot < SLOTS_NUM; slot++) {
        uint32_t offset = slot * SLOTS_SIZE * RECORD_SIZE;
        consumeSlot(&allocs[slot], &commits[slot], &records[offset], sink, true);
      }
    }

    // after shouldRun flag has been reset to false, no warps are writing, but
    // there might still be data in the buffers
    for(int slot = 0; slot < SLOTS_NUM; slot++) {
      uint32_t offset = slot * SLOTS_SIZE * RECORD_SIZE;
      consumeSlot(&allocs[slot], &commits[slot], &records[offset], sink, false);
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

  uint32_t *AllocsHost, *AllocsDevice;
  uint32_t *CommitsHost, *CommitsDevice;
  uint8_t *RecordsHost, *RecordsDevice;
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
