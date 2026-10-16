# [MODULE DOC]
# Smoke test for sklearn.__check_build: verifies that the helper used to report a
# broken/uncompiled installation re-raises as ImportError.

"""
Smoke Test the check_build module
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from sklearn.__check_build import raise_build_error


def test_raise_build_error():
    # [FUNCTION DOC]
    # Verifies that raise_build_error, given an ImportError instance, raises an
    # ImportError (with its explanatory build message) via pytest.raises.
    with pytest.raises(ImportError):
        raise_build_error(ImportError())
