# [MODULE DOC]
# Tests that importing the deprecated enable_hist_gradient_boosting module warns.
"""Tests for making sure experimental imports work as expected."""

import textwrap

import pytest

from sklearn.utils._testing import assert_run_python_script_without_output
from sklearn.utils.fixes import _IS_WASM


@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_import_raises_warning():
    # [FUNCTION DOC]
    # Verifies that importing enable_hist_gradient_boosting raises a UserWarning
    # matching "it is not needed to import". Runs the import in a fresh Python
    # subprocess (module caching would otherwise hide the warning) via
    # assert_run_python_script_without_output, which also checks that the subprocess
    # output does not contain anything besides the given pattern. Marked xfail on WASM
    # where subprocesses cannot start.
    code = """
    import pytest
    with pytest.warns(UserWarning, match="it is not needed to import"):
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    """
    pattern = "it is not needed to import enable_hist_gradient_boosting anymore"
    assert_run_python_script_without_output(textwrap.dedent(code), pattern=pattern)
