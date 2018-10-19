#ifndef __CUDA_TRACE_READER_H__
#define __CUDA_TRACE_READER_H__

#ifdef __cplusplus__
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Currently two variants of the trace format exist:
 * Version 2: Uncompressed. Each access made by a GPU thread corresponds to an
 *   individual record.
 * Version 3: Compressed. Consecutive accesses of the same size made by the same
 *   CTA are compressed into a single record with the field "count" set to the
 *   number of consecutive accesses.
 *
 * Both versions share identical headers:
 * 10 Byte: magic numbers as identifier
 *
 * 0x00 signals "new kernel"
 *   2 Byte: length of kernel name
 *   n Byte: kernel name
 *
 * 0xFF signals "uncompressed record" (Version 2, Version 3)
 *   4 Byte: SM Id
 *   4 Byte: <4 bit: type of access> <28 bit: size of access>
 *   8 Byte: Address of access
 *   4 Byte: CTA Id X
 *   2 Byte: CTA Id Y
 *   2 Byte: CTA Id Z
 *
 * 0xFE signals "compressed record" (Version 3 only)
 *   4 Byte: SM Id
 *   4 Byte: <4 bit: type of access> <28 bit: size of access>
 *   8 Byte: Address of access
 *   4 Byte: CTA Id X
 *   2 Byte: CTA Id Y
 *   2 Byte: CTA Id Z
 *   2 Byte: Count
 */

static const uint8_t v2[] = {0x19, 0x43, 0x55, 0x44, 0x41, 0x54, 0x52, 0x41, 0x43, 0x45};
static const uint8_t v3[] = {0x1a, 0x43, 0x55, 0x44, 0x41, 0x54, 0x52, 0x41, 0x43, 0x45};

typedef struct trace_record_t {
  uint8_t type;
  uint64_t addr;
  uint32_t size;
  struct {
    uint32_t x;
    uint16_t y;
    uint16_t z;
  } ctaid;
  uint32_t smid;
  uint16_t count;
} trace_record_t;

typedef struct trace_t {
  FILE *file;
  uint8_t version;
  char new_kernel;
  char *kernel_name;
  trace_record_t record;
} trace_t;

const char *trace_last_error = NULL;

/******************************************************************************
 * reader
 *****************************************************************************/

trace_t *trace_open(FILE *f) {
  uint8_t versionbuf[10];
  uint8_t version;

  if (fread(versionbuf, 10, 1, f) != 1) {
    trace_last_error = "unable to read version";
    return NULL;
  }

  if (memcmp(versionbuf, v2, 10) == 0) {
    version = 2;
  } else if (memcmp(versionbuf, v3, 10) == 0) {
    version = 3;
  } else {
    trace_last_error = "invalid version";
    return NULL;
  }

  trace_t *res = (trace_t*)malloc(sizeof(trace_t));
  res->file = f;
  res->version = version;
  res->kernel_name = NULL;
  res->new_kernel = 0;
  return res;
}

void __trace_unpack(const uint64_t buf[3], trace_record_t *record) {
  record->smid = (buf[0] >> 32);
  record->type = (buf[0] >> 28) & 0xf;
  record->size = buf[0] & 0x0fffffff;
  record->addr = buf[1];
  record->ctaid.x = (buf[2] >> 32) & 0xffffffff;
  record->ctaid.y = (buf[2] >> 16) & 0x0000ffff;
  record->ctaid.z = (buf[2] >>  0) & 0x0000ffff;
}

void __trace_pack(const trace_record_t *record, uint64_t buf[3]) {
  buf[0] = 0;
  buf[0] |= (uint64_t)record->smid << 32;
  buf[0] |= (uint64_t)(record->type & 0xf) << 28;
  buf[0] |= record->size & 0x0fffffff;
  buf[1] = record->addr;
  buf[2] = 0;
  buf[2] |= (uint64_t)record->ctaid.x << 32;
  buf[2] |= (uint64_t)(record->ctaid.y & 0x0000ffff) << 16;
  buf[2] |= (uint64_t)record->ctaid.z & 0x0000ffff;
}

// returns 0 on success
int trace_next(trace_t *t) {
  int ch = fgetc(t->file);
  // end of file, this is not an error
  if (ch == EOF) {
    trace_last_error = NULL;
    return 1;
  }

  // Entry is a kernel
  if (ch == 0x00) {
    uint16_t nameLen;
    if (fread(&nameLen, 2, 1, t->file) != 1) {
      trace_last_error = "unable to read kernel name length";
      return 1;
    }
    if (t->kernel_name != NULL) {
      free(t->kernel_name);
    }
    t->kernel_name = (char*)malloc(nameLen+1);
    if (fread(t->kernel_name, nameLen, 1, t->file) != 1) {
      trace_last_error = "unable to read kernel name length";
      return 1;
    }
    t->kernel_name[nameLen] = '\0';
    t->new_kernel = 1;
    trace_last_error = NULL;
    return 0;
  }

  // Entry is an uncompressed record
  if (ch == 0xFF) {
    t->new_kernel = 0;
    uint64_t buf[3];
    if (fread(buf, 24, 1, t->file) != 1) {
      trace_last_error = "unable to read uncompressed record";
      return 1;
    }
    __trace_unpack(buf, &t->record);
    t->record.count = 1;
    trace_last_error = NULL;
    return 0;
  }

  // Entry is a compressed record
  if (ch == 0xFE) {
    t->new_kernel = 0;
    uint64_t buf[3];
    if (fread(buf, 24, 1, t->file) != 1) {
      trace_last_error = "unable to read compressed record";
      return 1;
    }
    __trace_unpack(buf, &t->record);

    // read count
    uint16_t countbuf;
    if (fread(&countbuf, 2, 1, t->file) != 1) {
      trace_last_error = "unable to read repetition count";
      return 1;
    }

    t->record.count = countbuf;
    trace_last_error = NULL;
    return 0;
  }

  trace_last_error = "invalid entry marker";
  return 1;
}

int trace_eof(trace_t *t) {
  return feof(t->file);
}

void trace_close(trace_t *t) {
  if (t->kernel_name) {
    free(t->kernel_name);
  }
  fclose(t->file);
  free(t);
}

/******************************************************************************
 * writer
 *****************************************************************************/

int trace_write_header(FILE *f, int version) {
  if (version == 2) {
    if (fwrite(v2, 10, 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
  } else if (version == 3) {
    if (fwrite(v3, 10, 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
  } else {
    trace_last_error = "invalid version";
    return 1;
  }
  trace_last_error = NULL;
  return 0;
}

int trace_write_kernel(FILE *f, const char* name) {
  uint8_t bufmarker = 0x00;
  uint16_t bufsize = strlen(name);
  if (fwrite(&bufmarker, 1, 1, f) < 1) {
    trace_last_error = "write error";
    return 1;
  }
  if (fwrite(&bufsize, 2, 1, f) < 1) {
    trace_last_error = "write error";
    return 1;
  }
  if (fwrite(name, bufsize, 1, f) < 1) {
    trace_last_error = "write error";
    return 1;
  }
  trace_last_error = NULL;
  return 0;
}

int trace_write_record(FILE *f, const trace_record_t *record) {
  uint8_t marker = record->count == 1 ? 0xFF : 0xFE;
  if (fwrite(&marker, 1, 1, f) < 1) {
    trace_last_error = "write error";
    return 1;
  }

  uint64_t buf[3];
  __trace_pack(record, buf);
  if (fwrite(buf, 24, 1, f) < 1) {
    trace_last_error = "write error";
    return 1;
  }

  if (record->count > 1) {
    if (fwrite(&record->count, 2, 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
  }
  trace_last_error = NULL;
  return 0;
}

#ifdef __cplusplus__
}
#endif

#endif
