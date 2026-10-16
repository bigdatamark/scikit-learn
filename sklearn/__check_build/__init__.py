# [PACKAGE DOC]
# Package sklearn.__check_build: imported early by sklearn/__init__.py to verify the
# compiled extensions exist. At import time it tries to import the compiled
# _check_build extension; on ImportError it calls raise_build_error, which raises a
# more informative ImportError. Exposes INPLACE_MSG / STANDARD_MSG message strings
# and raise_build_error.
"""Module to give helpful messages to the user that did not
compile scikit-learn properly.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os

INPLACE_MSG = """
It appears that you are importing a local scikit-learn source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform."""


def raise_build_error(e):
    # [FUNCTION DOC]
    # HELPER: Raises an ImportError explaining that scikit-learn was not built properly.
    # The message contains the original error e, the package directory path, a listing
    # of its files formatted three per line (names padded to 26 chars), build
    # instructions and a hint: INPLACE_MSG when the directory path is exactly the
    # relative "sklearn/__check_build" (importing from a source tree), else STANDARD_MSG.
    # Inputs: e - the original exception. Outputs: never returns.
    # Exceptions: always ImportError. Side effects: lists a directory.
    # Raise a comprehensible error and list the contents of the
    # directory to help debugging on the mailing list.
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    if local_dir == "sklearn/__check_build":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = INPLACE_MSG
    dir_content = list()
    for i, filename in enumerate(os.listdir(local_dir)):
        if (i + 1) % 3:
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + "\n")
    raise ImportError(
        """%s
___________________________________________________________________________
Contents of %s:
%s
___________________________________________________________________________
It seems that scikit-learn has not been built correctly.

If you have installed scikit-learn from source, please do not forget
to build the package before using it. For detailed instructions, see:
https://scikit-learn.org/dev/developers/development_setup.html#install-editable-version-of-scikit-learn
%s"""
        % (e, local_dir, "".join(dir_content).strip(), msg)
    )


try:
    from sklearn.__check_build._check_build import check_build  # noqa: F401
except ImportError as e:
    raise_build_error(e)
