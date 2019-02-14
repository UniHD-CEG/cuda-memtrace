# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import sys
import platform

import lit.util
import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = 'Memtrace'
config.test_format = lit.formats.ShTest("O")
config.suffixes = ['.ll']
config.target_triple = ''
