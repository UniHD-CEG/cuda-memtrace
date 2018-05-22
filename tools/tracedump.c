// vim: ts=2 sts=2 sw=2 et ai
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include "../lib/Common.h"

/** This tracedump uses the updated trace file format:
 * The first ten bytes are:
 * 0x19 0x43 0x55 0x44 0x41 0x54 0x52 0x41 0x43 0x45
 *
 * The file then consists of arbitrarily many kernel traces.
 * A kernel trace consists of a kernel introduction followed by
 * arbitrarily many trace records.
 *
 * Kernel introduction:
 * first byte: 0x00
 * next two bytes: length of kernel name (host byte order)
 * next n bytes: kernel name
 *
 * Record:
 * first byte: 0xFF
 * next 24 bytes: memory dump of 3 x uint64_t
 */

#define die(...) do {\
  printf(__VA_ARGS__);\
  exit(1);\
} while(0)

#define KERNEL_MAX_NAME 180

typedef struct header_t {
  int version;
} header_t;

typedef struct kernel_t {
  char name[KERNEL_MAX_NAME];
} kernel_t;

int read_header(FILE *f, header_t *h) {
  uint8_t ref[10] = {0x19, 0x43, 0x55, 0x44, 0x41, 0x54, 0x52, 0x41, 0x43, 0x45};
  uint8_t buf[10];
  if (fread(buf, 10, 1, f) != 1) {
    return 1;
  }
  for (int i = 0; i < 10; ++i) {
    if (ref[i] != buf[i]) {
      return 1;
    }
  }
  h->version = 2;
  return 0;
}

int read_kernel(FILE *f, kernel_t *k) {
  int ch = fgetc(f);
  if (ch == EOF) {
    return 1;
  }
  if (ch != 0x00) {
    fseek(f, -1, SEEK_CUR);
    return 1;
  }

  uint16_t nameLen;
  if (fread(&nameLen, 2, 1, f) != 1) {
    return 1;
  }
  char name[nameLen];
  if (fread(name, nameLen, 1, f) != 1) {
    return 1;
  }
  nameLen = nameLen < KERNEL_MAX_NAME - 1 ? nameLen : KERNEL_MAX_NAME-1;
  strncpy(k->name, name, nameLen);
  k->name[nameLen] = '\0';
  return 0;
}

int read_record(FILE *f, record_t *r) {
  int ch = fgetc(f);
  if (ch == EOF) {
    return 1;
  }
  if (ch != 0xFF) {
    fseek(f, -1, SEEK_CUR);
    return 1;
  }

  uint64_t buf[3];
  if (fread(buf, RECORD_SIZE, 1, f) != 1) {
    return 1;
  }
  r->desc = buf[0];
  r->addr = buf[1];
  r->cta = buf[2];
  return 0;
}

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
    if (input == NULL)
      die("Unable to open file '%s', exiting\n", argv[1]);
  } else {
    usage("tracedump");
    exit(1);
  }

  int quiet = getenv("QUIET") != NULL;

  uint64_t num_records = 0;

  header_t header;
  if (read_header(input, &header) != 0)
    die("unable to read header\n");
  while (1) {
    kernel_t kernel;
    record_t record;
    record.addr = 0xfffff;
    if (read_kernel(input, &kernel) != 0)
      break;
    printf("Kernel name: %s\n", kernel.name);
    while (1) {
      if (read_record(input, &record) != 0)
        break;
      num_records += 1;
      if (!quiet) {
        printf("  Record: 0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64 "\n",
            record.desc, record.addr, record.cta);
      }
    }
  }

  printf("Read %" PRIu64 " records\n", num_records);

  if (!feof(input)) {
    printf("warning: unable to parse whole file\n");
  }

  fclose(input);
}
