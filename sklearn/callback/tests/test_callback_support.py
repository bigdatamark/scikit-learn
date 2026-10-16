# [MODULE DOC]
# Tests for the callback support API of estimators (CallbackSupportMixin):
# set_callbacks validation and behavior, setup/teardown pairing when callbacks fail,
# grouping of teardown errors, and callbacks on estimators fitted from a plain
# function that has no callback support (sequential, threads, processes).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from sklearn.base import clone
from sklearn.callback.tests._utils import (
    FailingCallback,
    MaxIterEstimator,
    NotValidCallback,
    RecordingAutoPropagatedCallback,
    RecordingCallback,
)
from sklearn.utils.parallel import Parallel, delayed


@pytest.mark.parametrize(
    "callbacks",
    [
        [RecordingCallback()],
        [RecordingCallback(), RecordingAutoPropagatedCallback()],
    ],
)
def test_set_callbacks(callbacks):
    # [FUNCTION DOC]
    # Verifies set_callbacks stores the given callbacks in _skl_callbacks in order and
    # returns the estimator itself.
    # Inputs: callbacks - parametrized list of callback instances.
    """Sanity check for the `set_callbacks` method."""
    estimator = MaxIterEstimator()

    set_callbacks_return = estimator.set_callbacks(*callbacks)
    assert hasattr(estimator, "_skl_callbacks")

    assert estimator._skl_callbacks == callbacks

    assert set_callbacks_return is estimator


@pytest.mark.parametrize("callback", [None, NotValidCallback(), RecordingCallback])
def test_set_callbacks_error(callback):
    # [FUNCTION DOC]
    # Verifies set_callbacks raises TypeError for None, an object not following the
    # FitCallback protocol, and a callback class instead of an instance.
    # Inputs: callback - parametrized invalid value.
    """Check the error message when not passing a valid callback to `set_callbacks`."""
    estimator = MaxIterEstimator()

    with pytest.raises(
        TypeError,
        match="callbacks must be instances following the FitCallback protocol.",
    ):
        estimator.set_callbacks(callback)


@pytest.mark.parametrize(
    "fail_at", ["setup", "on_fit_task_begin", "on_fit_task_end", "teardown"]
)
def test_callback_error(fail_at):
    # [FUNCTION DOC]
    # Verifies that when a callback raises in any hook, fit propagates the ValueError and
    # setup and teardown were each still called exactly once.
    # Inputs: fail_at - parametrized hook name where the callback fails.
    """Check that a failing callback is properly teared down."""
    callback = FailingCallback(fail_at=fail_at)
    estimator = MaxIterEstimator().set_callbacks(callback)
    with pytest.raises(ValueError, match=f"Failing callback failed at {fail_at}"):
        estimator.fit()

    assert callback.count_hooks("setup") == 1
    assert callback.count_hooks("teardown") == 1


def test_teardown_matches_setup_calls_on_partial_setup_failure():
    # [FUNCTION DOC]
    # Verifies that when the first callback fails in setup, the second callback's setup
    # and teardown are never called while the failing one is still torn down.
    """Check that teardown only runs for callbacks that entered setup."""
    callback_1 = FailingCallback(fail_at="setup")
    callback_2 = RecordingCallback()

    estimator = MaxIterEstimator().set_callbacks(callback_1, callback_2)
    with pytest.raises(ValueError, match="Failing callback failed at setup"):
        estimator.fit()

    assert callback_1.count_hooks("setup") == 1
    assert callback_1.count_hooks("teardown") == 1

    # setup was never entered for this callback, so teardown should not be called.
    assert callback_2.count_hooks("setup") == 0
    assert callback_2.count_hooks("teardown") == 0


def test_multiple_teardown_errors_are_grouped():
    # [FUNCTION DOC]
    # Verifies that if several callbacks fail in teardown, all teardowns run and their
    # errors are raised together as one ExceptionGroup with two exceptions.
    """Check that all teardown are called even if some fail.

    All errors are raised together in one ExceptionGroup.
    """
    callback_1 = FailingCallback(fail_at="teardown")
    callback_2 = FailingCallback(fail_at="teardown")
    estimator = MaxIterEstimator().set_callbacks(callback_1, callback_2)

    with pytest.raises(
        ExceptionGroup, match="The following callback teardown errors occurred"
    ) as exc_info:
        estimator.fit()

    assert len(exc_info.value.exceptions) == 2
    assert callback_1.count_hooks("teardown") == 1
    assert callback_2.count_hooks("teardown") == 1


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("prefer", ["threads", "processes"])
@pytest.mark.parametrize(
    "Callback", [RecordingCallback, RecordingAutoPropagatedCallback]
)
def test_function_no_callback_support(n_jobs, prefer, Callback):
    # [FUNCTION DOC]
    # Verifies that when a callback-carrying estimator is cloned and fitted repeatedly
    # from a function without callback support (via joblib Parallel), every fit has an
    # independent context tree: setup/teardown n_fits times and n_fits*(1+max_iter) task
    # begins/ends, for both callback kinds and all backends.
    # Inputs: n_jobs, prefer, Callback - parametrized.
    """Check callbacks on estimators within function not supporting callbacks.

    Since the outer function does not support callbacks, there's no shared root context
    and the context trees of each sub-estimator are independent. As a result, the
    callback acts as a regular non-propagated callback: its on_fit_begin and on_fit_end
    are called once for each fit of the sub-estimator and the number of tasks is the sum
    of the number of tasks from all the sub-estimators.
    """

    def clone_and_fit(estimator):
        # [FUNCTION DOC]
        # HELPER: Clones the estimator (callbacks are kept by clone) and fits the clone.
        # Outputs: none.
        clone(estimator).fit()

    def func(estimator, n_fits, n_jobs, prefer):
        # [FUNCTION DOC]
        # HELPER: Runs clone_and_fit n_fits times through joblib Parallel with the given
        # n_jobs and prefer backend hint.
        # Outputs: none.
        Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(clone_and_fit)(estimator) for _ in range(n_fits)
        )

    n_fits, max_iter = 5, 7
    callback = Callback()
    estimator = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)

    func(estimator, n_fits, n_jobs, prefer)

    assert callback.count_hooks("setup") == n_fits
    # 1 root + max_iter leaves per fit
    assert callback.count_hooks("on_fit_task_begin") == n_fits * (1 + max_iter)
    assert callback.count_hooks("on_fit_task_end") == n_fits * (1 + max_iter)
    assert callback.count_hooks("teardown") == n_fits


def test_set_callback_empty():
    # [FUNCTION DOC]
    # Verifies that calling set_callbacks with no arguments deletes _skl_callbacks, and
    # that doing so again when the attribute is absent does not raise.
    """Check that setting no callbacks removes the `_skl_callbacks` attribute."""
    estimator = MaxIterEstimator()

    estimator.set_callbacks(RecordingCallback())
    assert hasattr(estimator, "_skl_callbacks")

    estimator.set_callbacks()
    assert not hasattr(estimator, "_skl_callbacks")

    # calling again doesn't raise
    estimator.set_callbacks()
    assert not hasattr(estimator, "_skl_callbacks")
