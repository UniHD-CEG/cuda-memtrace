// vim: ts=2 sts=2 sw=2 et ai
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <inttypes.h>

#define die(...) do {\
  printf(__VA_ARGS__);\
  exit(1);\
} while(0)

typedef struct header_t {
  int record_size;
} header_t;

typedef struct record_t {
  int64_t desc;
  int64_t addr;
  int64_t size;
} record_t;

#define KERNEL_MAX_NAME 180
typedef struct kernel_t {
  char name[KERNEL_MAX_NAME];
} kernel_t;

int read_header(FILE *f, header_t *h) {
  int ch = fgetc(f);
  if (ch == EOF)
    return 1;
  h->record_size = ch;
  return 0;
}

int read_kernel(FILE *f, kernel_t *k) {
  int ch = fgetc(f);
  while (ch != '\n' && ch != EOF) {
    ch = fgetc(f);
  }
  if (ch == EOF)
    return 1;

  ch = fgetc(f);
  int pos = 0;
  while (ch != '\n' && ch != EOF) {
    if (pos < KERNEL_MAX_NAME-1) {
      k->name[pos] = ch;
      pos += 1;
    }
    ch = fgetc(f);
  }
  k->name[pos] = '\0';
  if (ch == EOF)
    return 1;
  return 0;
}

int read_record(FILE *f, record_t *r) {
  uint64_t buf[3];
  int obj_read = fread(buf, sizeof(uint64_t), 3, f);
  if (obj_read != 3)
    return 1;
  r->desc = buf[0];
  r->addr = buf[1];
  r->size = buf[2];
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
  while (!feof(input)) {
    kernel_t kernel;
    record_t record;
    record.addr = 0xfffff;
    if (read_kernel(input, &kernel) != 0)
      break;
    printf("kernel name: '%s'\n", kernel.name);
    while (!feof(input)) {
      if (read_record(input, &record) != 0)
        die("unable to read record\n");
      if (record.addr == 0) {
        break;
      }
      num_records += 1;
      if (!quiet) {
        printf("  Record: 0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64 "\n",
            record.desc, record.addr, record.size);
      }
    }
  }

  printf("Read %" PRIu64 " records\n", num_records);

  fclose(input);
}
