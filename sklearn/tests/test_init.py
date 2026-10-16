# [MODULE DOC]
# Top-level import smoke test. At import time it attempts a star-import of
# sklearn and stores any exception in the module global _top_import_error
# (None on success), because star-imports are not allowed inside functions.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Basic unittests to test functioning of module's top-level


try:
    from sklearn import *  # noqa: F403

    _top_import_error = None
except Exception as e:
    _top_import_error = e


def test_import_skl():
    # [FUNCTION DOC]
    # Verifies that the module-level star-import of sklearn succeeded, by asserting
    # the recorded _top_import_error is None.
    # Test either above import has failed for some reason
    # "import *" is discouraged outside of the module level, hence we
    # rely on setting up the variable above
    assert _top_import_error is None
