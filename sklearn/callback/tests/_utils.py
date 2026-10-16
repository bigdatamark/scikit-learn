# [MODULE DOC]
# HELPER: Test utilities for sklearn.callback: recording/failing callbacks and toy
# estimators and meta-estimators exercising the callback context API, plus the
# skip_callback_test_if_wasm pytest marker. Toy estimators sleep for
# computation_intensity seconds per iteration to emulate work.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import numpy as np
import pytest

from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.callback import CallbackSupportMixin, with_callbacks
from sklearn.callback._transport import open_listener, send
from sklearn.utils.fixes import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed

skip_callback_test_if_wasm = pytest.mark.skipif(
    _IS_WASM,
    reason="callback tests are skipped on WASM/Pyodide",
)


class RecordingCallback:
    # [CLASS DOC]
    # HELPER: FitCallback recording every hook call as a dict (name, estimator, context
    # and, for task hooks, kwargs) into self.record, delivered through a transport
    # listener so calls made in worker processes reach the main-process record.
    """A minimal callback used for smoke testing purposes.

    This callback keeps a record of the hooks called for introspection.

    This callback doesn't define `max_propagation_depth` and is therefore not an
    `AutoPropagatedCallback`: it should not be propagated to sub-estimators.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Creates the empty record and opens a listener (owned by self) appending to it.
        self.record = []
        self._listener_handle = open_listener(self.record.append, owner=self)

    def __getstate__(self):
        # [FUNCTION DOC]
        # HELPER: Pickle state with an empty record, to keep pickles small.
        # We never access the record from a worker so there's no need to ship it.
        # Emptying it avoids nesting the record, callback and context in each other
        # which would grow the pickle size exponentially.
        return {**self.__dict__, "record": []}

    def setup(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Sends a "setup" record entry.
        send(
            self._listener_handle,
            {"name": "setup", "estimator": estimator, "context": context},
        )

    def on_fit_task_begin(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Sends an "on_fit_task_begin" entry with all received keyword arguments.
        send(
            self._listener_handle,
            {
                "name": "on_fit_task_begin",
                "estimator": estimator,
                "context": context,
                "kwargs": {
                    "X": X,
                    "y": y,
                    "metadata": metadata,
                    "fitted_estimator": fitted_estimator,
                },
            },
        )

    def on_fit_task_end(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Sends an "on_fit_task_end" entry with all keyword arguments; returns None
        # (never stops fit).
        send(
            self._listener_handle,
            {
                "name": "on_fit_task_end",
                "estimator": estimator,
                "context": context,
                "kwargs": {
                    "X": X,
                    "y": y,
                    "metadata": metadata,
                    "fitted_estimator": fitted_estimator,
                },
            },
        )

    def teardown(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Sends a "teardown" record entry.
        send(
            self._listener_handle,
            {"name": "teardown", "estimator": estimator, "context": context},
        )

    def count_hooks(self, hook_name):
        # [FUNCTION DOC]
        # HELPER: Returns the number of recorded entries whose name equals hook_name.
        return len([rec for rec in self.record if rec["name"] == hook_name])


class RecordingAutoPropagatedCallback(RecordingCallback):
    # [CLASS DOC]
    # HELPER: RecordingCallback that is auto-propagated (max_propagation_depth None).
    """A minimal auto-propagated callback used for smoke testing purposes.

    This callback keeps a record of the hooks called for introspection.

    This callback defines `max_propagation_depth` and is therefore an
    `AutoPropagatedCallback`: it should be set on a top-level estimator and propagated
    to sub-estimators.
    """

    max_propagation_depth = None


class NotValidCallback:
    # [CLASS DOC]
    # HELPER: Object lacking teardown and on_fit_task_begin, so it fails the FitCallback
    # protocol check.
    """Invalid callback since it's missing methods from the protocol."""

    def setup(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: No-op; never called in tests.
        pass  # pragma: no cover

    def on_fit_task_end(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: No-op; never called in tests.
        pass  # pragma: no cover


class NotValidHookCallback(RecordingCallback):
    # [CLASS DOC]
    # HELPER: Callback whose on_fit_task_begin declares a keyword-only parameter not in
    # VALID_HOOK_PARAMS_OUT, triggering TypeError in CallbackContext._call_hooks.
    """Invalid callback since it has invalid parameters in the hooks signatures."""

    def on_fit_task_begin(self, estimator, context, *, not_valid_kwarg=None):
        # [FUNCTION DOC]
        # HELPER: No-op; never reached because dispatch raises first.
        pass  # pragma: no cover


class FailingCallback(RecordingCallback):
    # [CLASS DOC]
    # HELPER: RecordingCallback that raises ValueError in the hook named by fail_at
    # (after recording the call).
    """A callback that raises an error at some point."""

    def __init__(self, fail_at=None):
        # [FUNCTION DOC]
        # HELPER: Inputs: fail_at - None or one of "setup", "on_fit_task_begin",
        # "on_fit_task_end", "teardown".
        super().__init__()
        self.fail_at = fail_at

    def setup(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Records setup; raises ValueError if fail_at is "setup".
        super().setup(estimator, context)
        if self.fail_at == "setup":
            raise ValueError("Failing callback failed at setup")

    def on_fit_task_begin(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Records the call (without kwargs); raises ValueError if fail_at matches.
        super().on_fit_task_begin(estimator, context)
        if self.fail_at == "on_fit_task_begin":
            raise ValueError("Failing callback failed at on_fit_task_begin")

    def on_fit_task_end(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Records the call (without kwargs); raises ValueError if fail_at matches.
        super().on_fit_task_end(estimator, context)
        if self.fail_at == "on_fit_task_end":
            raise ValueError("Failing callback failed at on_fit_task_end")

    def teardown(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Records teardown; raises ValueError if fail_at is "teardown".
        super().teardown(estimator, context)
        if self.fail_at == "teardown":
            raise ValueError("Failing callback failed at teardown")


class StopFitCallback(RecordingCallback):
    # [CLASS DOC]
    # HELPER: RecordingCallback whose on_fit_task_end always requests stopping.
    """A callback with a `on_fit_task_end` hook returning True."""

    def on_fit_task_end(self, estimator, context):
        # [FUNCTION DOC]
        # HELPER: Records the call and returns True.
        super().on_fit_task_end(estimator, context)
        return True


class NotRequiredKwargsCallback(RecordingCallback):
    # [CLASS DOC]
    # HELPER: RecordingCallback whose on_fit_task_end requests only X and y, to check
    # that only declared kwargs are passed.
    """A callback with a `on_fit_task_end` not requiring all possible kwargs."""

    def on_fit_task_end(self, estimator, context, *, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Records the call with X and y only.
        super().on_fit_task_end(estimator, context, X=X, y=y)


class MaxIterEstimator(CallbackSupportMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Toy estimator with a fixed-length iteration loop.
    """A class that mimics the behavior of an estimator.

    The iterative part uses a loop with a max number of iterations known in advance.

    This estimator computes arbitrary predictions by averaging the feature
    values and multiplying the result by the number of iterations done
    in fit.
    """

    _parameter_constraints: dict = {}

    def __init__(self, max_iter=20, computation_intensity=0.001):
        # [FUNCTION DOC]
        # HELPER: Stores max_iter (int, 20) and computation_intensity (seconds, 0.001).
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X=None,
        y=None,
        *,
        sample_weight=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Root context with max_iter subtasks; for each iteration open a sequential
        # subcontext, call begin, sleep, call end with lazy reconstruction attributes
        # {n_iter_: i+1}, and break if a callback requested stopping. Sets n_iter_, calls the
        # root end hook with empty reconstruction attributes. sample_weight is forwarded as
        # metadata when given. Outputs: self.
        # KNOWN ISSUE: with max_iter=0 the loop variable is undefined and setting n_iter_
        # raises NameError.
        callback_ctx = self._init_callback_context(max_subtasks=self.max_iter)
        metadata = {"sample_weight": sample_weight} if sample_weight is not None else {}
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y, metadata=metadata)

        for i in range(self.max_iter):
            subcontext = callback_ctx.subcontext(task_name=f"iteration {i}")
            subcontext.call_on_fit_task_begin(
                estimator=self, X=X, y=y, metadata=metadata
            )

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.call_on_fit_task_end(
                estimator=self,
                X=X,
                y=y,
                metadata=metadata,
                reconstruction_attributes=lambda: {"n_iter_": i + 1},
            ):
                break

        self.n_iter_ = i + 1

        callback_ctx.call_on_fit_task_end(
            estimator=self,
            X=X,
            y=y,
            metadata=metadata,
            reconstruction_attributes={},
        )

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the row means of X times n_iter_.
        return np.mean(X, axis=1) * self.n_iter_


class WhileEstimator(CallbackSupportMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Toy estimator with an unknown number of iterations (max_subtasks None).
    """A class that mimics the behavior of an estimator.

    The iterative part uses a while loop with a number of iterations unknown in
    advance.
    """

    _parameter_constraints: dict = {}

    def __init__(self, computation_intensity=0.001):
        # [FUNCTION DOC]
        # HELPER: Stores computation_intensity.
        self.computation_intensity = computation_intensity

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Loops creating auto-numbered subcontexts until a callback requests stopping
        # or 21 iterations are done; brackets with root begin/end hooks. No reconstruction
        # attributes are passed. Outputs: self.
        callback_ctx = self._init_callback_context(max_subtasks=None)
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)

        i = 0
        while True:
            subcontext = callback_ctx.subcontext()
            subcontext.call_on_fit_task_begin(estimator=self, X=X, y=y)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.call_on_fit_task_end(estimator=self, X=X, y=y):
                break

            if i == 20:
                break

            i += 1

        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y)

        return self


class ThirdPartyEstimator(CallbackSupportMixin):
    # [CLASS DOC]
    # HELPER: Non-BaseEstimator class using only the public callback API (with_callbacks).
    """A class that mimics a third-party estimator with callback support only using
    public API.
    """

    def __init__(self, max_iter=20, computation_intensity=0.001):
        # [FUNCTION DOC]
        # HELPER: Stores max_iter and computation_intensity.
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    @with_callbacks
    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Same loop as MaxIterEstimator.fit without metadata or reconstruction
        # attributes; sets n_iter_ after the root end hook. Outputs: self.
        # KNOWN ISSUE: max_iter=0 leads to NameError on the loop variable.
        callback_ctx = self._init_callback_context(max_subtasks=self.max_iter)
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)

        for i in range(self.max_iter):
            subcontext = callback_ctx.subcontext()
            subcontext.call_on_fit_task_begin(estimator=self, X=X, y=y)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.call_on_fit_task_end(estimator=self, X=X, y=y):
                break

        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y)

        self.n_iter_ = i + 1

        return self


class ParentFitEstimator(MaxIterEstimator):
    # [CLASS DOC]
    # HELPER: Subclass whose fit delegates to MaxIterEstimator.fit (nested _fit_context).
    """A class that mimics an estimator using its parent fit method."""

    _parameter_constraints: dict = {}

    def __init__(self, max_iter=20, computation_intensity=0.001):
        # [FUNCTION DOC]
        # HELPER: Forwards to the parent __init__.
        super().__init__(max_iter, computation_intensity)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Calls the parent fit with X, y and returns its result.
        return super().fit(X, y)


class NoCallbackEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator without callback support (no set_callbacks), for warning tests.
    """A class that mimics an estimator without callback support."""

    def __init__(self, max_iter=20, computation_intensity=0.001):
        # [FUNCTION DOC]
        # HELPER: Stores max_iter and computation_intensity.
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Sleeps max_iter times; returns self.
        for i in range(self.max_iter):
            time.sleep(self.computation_intensity)  # Computation intensive task

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns zeros of length n_samples.
        return np.zeros(X.shape[0])


class MetaEstimator(CallbackSupportMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Toy meta-estimator with non-sequential outer tasks run in joblib Parallel and
    # inner tasks each fitting a clone of the sub-estimator.
    """A class that mimics the behavior of a meta-estimator.

    It has two levels of iterations. The outer level uses parallelism and the inner
    level is done in a function that is not a method of the class. That function must
    therefore receive the estimator and the callback context as arguments.
    """

    _parameter_constraints: dict = {}

    def __init__(
        self, estimator, n_outer=4, n_inner=3, n_jobs=None, prefer="processes"
    ):
        # [FUNCTION DOC]
        # HELPER: Stores estimator, n_outer (4), n_inner (3), n_jobs (None), prefer
        # ("processes", joblib backend preference).
        self.estimator = estimator
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_jobs = n_jobs
        self.prefer = prefer

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Creates a root context with n_outer non-sequential subtasks, then n_outer
        # "outer" subcontexts with explicit ids and n_inner children each, runs
        # _fit_subestimator for each in parallel, and brackets with root begin/end hooks.
        # Outputs: self.
        callback_ctx = self._init_callback_context(
            max_subtasks=self.n_outer, sequential_subtasks=False
        )
        metadata = {"sample_weight": sample_weight} if sample_weight is not None else {}
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y, metadata=metadata)

        outer_callback_contexts = [
            callback_ctx.subcontext(
                task_name="outer", task_id=i, max_subtasks=self.n_inner
            )
            for i in range(self.n_outer)
        ]

        Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(_fit_subestimator)(
                self,
                self.estimator,
                X=X,
                y=y,
                metadata=metadata,
                outer_callback_ctx=outer_callback_contexts[i],
            )
            for i in range(self.n_outer)
        )

        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y, metadata=metadata)

        return self


def _fit_subestimator(
    meta_estimator, inner_estimator, *, X, y, metadata, outer_callback_ctx
):
    # [FUNCTION DOC]
    # HELPER: For one outer task: call its begin hook, then n_inner times clone the inner
    # estimator, create an "inner" leaf subcontext, and inside propagate_callback_context
    # call begin, fit the clone (metadata passed as fit kwargs), call end; finally call
    # the outer end hook. Outputs: None; fitted clones are discarded.
    outer_callback_ctx.call_on_fit_task_begin(
        estimator=meta_estimator, X=X, y=y, metadata=metadata
    )

    for i in range(meta_estimator.n_inner):
        est = clone(inner_estimator)

        inner_ctx = outer_callback_ctx.subcontext(task_name="inner")
        with inner_ctx.propagate_callback_context(est):
            inner_ctx.call_on_fit_task_begin(
                estimator=meta_estimator, X=X, y=y, metadata=metadata
            )

            est.fit(X=X, y=y, **metadata)

            inner_ctx.call_on_fit_task_end(
                estimator=meta_estimator, X=X, y=y, metadata=metadata
            )

    outer_callback_ctx.call_on_fit_task_end(
        estimator=meta_estimator, X=X, y=y, metadata=metadata
    )


class HeterogeneousMetaEstimator(CallbackSupportMixin):
    # [CLASS DOC]
    # HELPER: Meta-estimator fitting a list of estimators sequentially; None entries are
    # skipped but still get a leaf task.
    """A meta-estimator that fits a list of estimators in order."""

    def __init__(self, estimators):
        # [FUNCTION DOC]
        # HELPER: Stores the list of estimators (entries may be None).
        self.estimators = estimators

    @with_callbacks
    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: One subtask per entry, named "fit <Class>" or "skip <i>"; non-None entries
        # are cloned and fitted inside propagate_callback_context with begin/end hooks.
        # Outputs: self.
        callback_ctx = self._init_callback_context(max_subtasks=len(self.estimators))
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)

        for i, est in enumerate(self.estimators):
            task_name = f"fit {est.__class__.__name__}" if est else f"skip {i}"
            subcontext = callback_ctx.subcontext(task_name=task_name)
            if est is not None:
                est = clone(est)
                with subcontext.propagate_callback_context(est):
                    subcontext.call_on_fit_task_begin(estimator=self, X=X, y=y)
                    est.fit(X, y)
                    subcontext.call_on_fit_task_end(estimator=self, X=X, y=y)
            else:
                subcontext.call_on_fit_task_begin(estimator=self, X=X, y=y)
                subcontext.call_on_fit_task_end(estimator=self, X=X, y=y)

        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y)

        return self


class NoSubtaskEstimator(CallbackSupportMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose fit has a single leaf root task.
    """A class mimicking an estimator without subtasks in fit."""

    @with_callbacks
    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Root context with no subtasks; calls begin (whose return value, the context,
        # is kept) then end. Outputs: self.
        callback_ctx = self._init_callback_context().call_on_fit_task_begin(
            estimator=self, X=X, y=y
        )

        # No task performed

        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y)

        return self
