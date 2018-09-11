#include "../lib/Common.h"
#include "../tools/cutrace_io.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>

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


static const char* getexename() {
  static char* cmdline = NULL;

  if (cmdline != NULL) {
    return cmdline;
  }

  FILE *f = fopen("/proc/self/cmdline", "r");
  if (!f) {
    return NULL;
  }
  size_t n;
  getdelim(&cmdline, &n, 0, f);
  fclose(f);
  cmdline = basename(cmdline);
  return cmdline;
}

/** Allows to specify base name for traces. The first occurence of
 * "?" is replaced with an ID unique to each stream.
 * Default pattern: "./trace-?.bin"
 */
static std::string traceName(std::string id) {
  static const char* exename = getexename(); // initialize once
  const char* pattern_env = getenv("MEMTRACE_PATTERN");
  std::string pattern;
  if (pattern_env) {
    pattern = pattern_env;
  } else if (exename) {
    pattern = "./" + std::string(exename) + "-?.trc";
  } else {
    pattern = "./trace-?.trc";
  }

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
 * Allocation and commit pointers are uint32_t with 64 Byte padding to avoid cache thrashing.
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

    output = fopen(this->pipeName.c_str(), "wb");
    if (output == nullptr) {
      printf("unable to open trace file '%s' for writing\n", pipeName.c_str());
      abort();
    }

    trace_write_header(output, 3);
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
    memset(AllocsHost, 0, SLOTS_NUM * CACHELINE);
    memset(CommitsHost, 0, SLOTS_NUM * CACHELINE);
    // just for testing purposes
    memset(RecordsHost, 0, SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE);

    recordAcc.count = 0; // count == 0 -> uninitialized
    trace_write_kernel(output, name.c_str());

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

  static uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
  }

  // clear up a slot if it is full
  static void consumeSlot(uint8_t *allocPtr, uint8_t *commitPtr, uint8_t *recordsPtr,
      FILE* out, bool kernelActive, trace_record_t *acc) {
    // allocs/commits is written by threads on the GPU, so we need it volatile
    volatile uint32_t *vcommit = (uint32_t*)commitPtr;
    volatile uint32_t *valloc = (uint32_t*)allocPtr;

    
    // if kernel is still active we only want to read full slots
    uint32_t numRecords = *vcommit;
    if (kernelActive && !(numRecords > SLOTS_SIZE - 32)) {
      return;
    }

    trace_record_t newrec;
    // we know writing from the gpu stopped, so we avoid using the volatile
    // reference in the end condition
    for (int32_t i = 0; i < numRecords; ++i) {

      __trace_unpack((uint64_t*)&recordsPtr[i * RECORD_SIZE], &newrec);

      // if this is the first record, intialize it
      if (acc->count == 0) {
        *acc = newrec;
        acc->count = 1;
      } else { // otherwise see if we can increment or have to flush
        uint64_t currAddress = acc->addr + (acc->size * acc->count);
        if (newrec.addr == currAddress && newrec.type == acc->type &&
            newrec.size == acc->size && newrec.ctaid.x == acc->ctaid.x &&
            newrec.ctaid.y == acc->ctaid.y && newrec.ctaid.z == acc->ctaid.z &&
            newrec.smid == acc->smid) {
          acc->count += 1;
        } else {
          trace_write_record(out, acc);
          *acc = newrec;
          acc->count = 1;
        }
      }
    }

    *vcommit = 0;
    // ensure commits are reset first
    std::atomic_thread_fence(std::memory_order_release);
    *valloc = 0;
  }

  // payload function of queue consumer
  static void consume(TraceConsumer *obj) {
    obj->doesRun = true;

    uint8_t *allocs = obj->AllocsHost;
    uint8_t *commits = obj->CommitsHost;
    uint8_t *records = obj->RecordsHost;

    FILE* sink = obj->output;

    while(obj->shouldRun) {
      for(int slot = 0; slot < SLOTS_NUM; slot++) {
        uint32_t allocs_offset = slot * CACHELINE;
        uint32_t commits_offset = slot * CACHELINE;
        uint32_t records_offset = slot * SLOTS_SIZE * RECORD_SIZE;
        consumeSlot(&allocs[allocs_offset], &commits[commits_offset],
            &records[records_offset], sink, true, &obj->recordAcc);
      }
    }

    // after shouldRun flag has been reset to false, no warps are writing, but
    // there might still be data in the buffers
    for(int slot = 0; slot < SLOTS_NUM; slot++) {
      uint32_t allocs_offset = slot * CACHELINE;
      uint32_t commits_offset = slot * CACHELINE;
      uint32_t records_offset = slot * SLOTS_SIZE * RECORD_SIZE;
        consumeSlot(&allocs[allocs_offset], &commits[commits_offset],
            &records[records_offset], sink, false, &obj->recordAcc);
    }

    // flush accumulator and reset to uninitialized
    trace_write_record(sink, &obj->recordAcc);
    obj->recordAcc.count = 0;

    obj->doesRun = false;
    return;
  }

  std::string suffix;

  std::atomic<bool> shouldRun;
  std::atomic<bool> doesRun;
  trace_record_t recordAcc; // recordAccumulator for compression

  FILE *output;
  std::thread       workerThread;
  std::string       pipeName;

  uint8_t *AllocsHost, *AllocsDevice;
  uint8_t *CommitsHost, *CommitsDevice;
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

void __trace_copy_to_symbol(cudaStream_t stream, const void* symbol, const void *info) {
  //printf("cudaMemcpyToSymbol(%p, %p, %zu, 0, cudaMemcpyHostToDevice)\n", symbol, info, sizeof(traceinfo_t));
  cudaChecked(cudaMemcpyToSymbolAsync(symbol, info, sizeof(traceinfo_t), 0, cudaMemcpyHostToDevice, stream));
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
