#!/usr/bin/env python

from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def usage():
    eprint("usage: headerize.py variable-name filename")
    eprint("")
    eprint("creates a header file containing a global variable with the")
    eprint("name <varname> containing the contents of <filename>. All")
    eprint("control characters are escaped. Results are written to stdout.")

args = sys.argv

if (len(args) < 3):
    eprint("error: too few arguments")
    eprint("")
    usage()
    sys.exit(1)

with open(args[2], "rb") as f:
    # calculate size, as: seek to end, get position, seek back to start
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)

    # write escaped blob
    print("#pragma once")
    print("")
    #print("const char %s[%d] = " % (args[1], size+1))
    print("const unsigned char %s[] = {" % (args[1]))

    done = False
    while not done:
        sys.stdout.write("  ")
        for _ in range(20):
            byte = f.read(1)
            if byte == b"":
                done = True
                break
            sys.stdout.write("0x%02x," % ord(byte))
        sys.stdout.write("\n")
    print("};")

    #done = False
    #while not done:
    #    sys.stdout.write("  \"")
    #    for _ in range(20):
    #        byte = f.read(1)
    #        if byte == b"":
    #            done = True
    #            break
    #        sys.stdout.write("\\x%02x" % ord(byte))
    #    sys.stdout.write("\"")
    #    if not done:
    #        print("")
    #print(";")
