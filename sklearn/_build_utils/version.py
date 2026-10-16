#!/usr/bin/env python3
# [MODULE DOC]
# HELPER: Build script that prints the scikit-learn version. At run/import time it
# reads ../__init__.py relative to this file, takes the first line starting with
# "__version__", splits on " = " and removes quotes, then prints the version string.
# Exceptions: StopIteration if no such line exists; IndexError if the assignment
# lacks the " = " spacing. Side effects: file read, stdout; the file handle is not
# explicitly closed.
"""Extract version number from __init__.py"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os

sklearn_init = os.path.join(os.path.dirname(__file__), "../__init__.py")

data = open(sklearn_init).readlines()
version_line = next(line for line in data if line.startswith("__version__"))

version = version_line.strip().split(" = ")[1].replace('"', "").replace("'", "")

print(version)
