#include "../lib/cutrace_io.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#include <stdio.h>

#define die(...) do {\
  printf(__VA_ARGS__);\
  exit(1);\
} while(0)

const char* ACC_TYPE_NAMES[] = {
  "LD", "ST", "AT", "??"
};

void usage(const char* program_name) {
  printf("Usage: %s [file]\n", program_name);
  printf("\n");
  printf("If a file is provided, reads a binary memory trace from it and\n");
  printf("dumps it to stdout. If no file is provided, uses stdin.\n");
}

int main(int argc, char** argv) {
  FILE *input;
  if (argc == 1) {
    input = stdin;
  } else if (argc == 2) {
    input = fopen(argv[1], "r");
    if (input == NULL) {
      die("Unable to open file '%s', exiting\n", argv[1]);
    }
  } else {
    usage("cutracedump");
    exit(1);
  }

  int quiet = getenv("QUIET") != NULL;

  trace_t *trace = trace_open(input);

  if (trace == NULL) {
    die("%s", trace_last_error);
  }

  int64_t accesses = -1;
  while (trace_next(trace) == 0) {
    if (trace->new_kernel) {
      if (accesses > -1) {
        printf("  Total number of accesses: %" PRId64 "\n", accesses);
      }
      printf("Kernel name: %s\n", trace->kernel_name);
      accesses = 0;
    } else {
      trace_record_t *r = &trace->record;
      accesses += r->count;
      if (quiet) {
        continue;
      }
      if (r->count == 1) {
        printf("  type: %s, addr: 0x%" PRIx64 ", size: %" PRIu32 ", cta: (%d, %d, %d), sm: %d\n",
          ACC_TYPE_NAMES[r->type], r->addr, r->size,
          r->ctaid.x, r->ctaid.y, r->ctaid.z, r->smid);
      } else {
        printf("  type: %s, start: 0x%" PRIx64 ", count: %" PRIu16 ", size: %" PRIu32 ", cta: (%d, %d, %d), sm: %d\n",
          ACC_TYPE_NAMES[r->type], r->addr, r->count, r->size,
          r->ctaid.x, r->ctaid.y, r->ctaid.z, r->smid);
      }
    }
  }
  if (!trace_eof(trace)) {
    printf("position: %zu\n", ftell(trace->file));
    die("%s", trace_last_error);
  }
  if (accesses > -1) {
    printf("  Total number of accesses: %" PRId64 "\n", accesses);
  }

  trace_close(trace);
}
