# [MODULE DOC]
# HELPER: Private pytest utilities for the test suite. Importing it imports pytest.
"""Private testing utilities."""

import pytest

__all__ = ["xfail"]


def xfail(
    request: pytest.FixtureRequest, *, reason: str, strict: bool | None = None
) -> None:
    # [FUNCTION DOC]
    # HELPER: Marks the running test as expected-to-fail without stopping it, by adding
    #   a pytest xfail marker (with reason, and strict only if given) to request.node,
    #   so a pass becomes XPASS.
    # Inputs: request - pytest FixtureRequest; reason - str; strict - optional bool.
    # Outputs: None. Side effects: mutates the test node's markers.
    """
    XFAIL the currently running test.

    Unlike ``pytest.xfail``, allow rest of test to execute instead of immediately
    halting it, so that it may result in a XPASS.
    xref https://github.com/pandas-dev/pandas/issues/38902

    Parameters
    ----------
    request : pytest.FixtureRequest
        ``request`` argument of the test function.
    reason : str
        Reason for the expected failure.
    strict : bool, optional
        If True, the test will be marked as failed if it passes.
        If False, the test will be marked as passed if it fails.
        Default: ``xfail_strict`` value in ``pyproject.toml``, or False if absent.
    """
    if strict is not None:
        marker = pytest.mark.xfail(reason=reason, strict=strict)
    else:
        marker = pytest.mark.xfail(reason=reason)
    request.node.add_marker(marker)
