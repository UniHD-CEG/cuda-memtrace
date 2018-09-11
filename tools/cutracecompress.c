#include "cutrace_io.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#include <stdio.h>

#define die(...) do {\
  fprintf(stderr, __VA_ARGS__);\
  exit(1);\
} while(0)

void usage(const char* program_name) {
  fprintf(stderr, "Usage: %s < input > output\n", program_name);
  fprintf(stderr, "\n");
  //printf("If a file is provided, reads a binary memory trace from it and\n");
  //printf("dumps it to stdout. If no file is provided, uses stdin.\n");
}

int main(int argc, char** argv) {
  FILE *input;
  FILE *output;

  input = stdin;
  output = stdout;

  trace_t *trace = trace_open(input);

  if (trace == NULL) {
    die("%s", trace_last_error);
  }

  trace_write_header(output, 3);

  trace_record_t record;
  int record_is_valid = 0;

  while (trace_next(trace) == 0) {
    if (trace->new_kernel) {
      if (record_is_valid) {
        trace_write_record(output, &record);
        record_is_valid = 0;
      }
      fprintf(stderr, "Kernel name: %s\n", trace->kernel_name);
      trace_write_kernel(output, trace->kernel_name);

    } else {

      trace_record_t *newrec = &trace->record;

      // we dont "recompress" compressed records
      if (newrec->count > 1) {
        if (record_is_valid) {
          trace_write_record(output, &record);
          record_is_valid = 0;
        }
        trace_write_record(output, newrec);
        continue;
      }

      if (!record_is_valid) {
        record = *newrec;
        record_is_valid = 1;
      } else {
        // for compression we required that EVERYTHING is identical except the address, which
        // has to be consecutive
        uint64_t currAddress = record.addr + (record.size * record.count);
        if (newrec->addr == currAddress && newrec->type == record.type &&
            newrec->size == record.size && newrec->ctaid.x == record.ctaid.x &&
            newrec->ctaid.y == record.ctaid.y && newrec->ctaid.z == record.ctaid.z &&
            newrec->smid == record.smid) {
          record.count += 1;
        } else {
          trace_write_record(output, &record);
          record = *newrec;
        }
      }
    }
  }
  if (record_is_valid) {
    trace_write_record(output, &record);
  }
  if (!trace_eof(trace)) {
    die("%s", trace_last_error);
  }
  trace_close(trace);

}
