# [MODULE DOC]
# Tests for CallbackContext (sklearn.callback._callback_context): building the task
# tree (subcontext, _add_child, _merge_with), propagating callbacks from a
# meta-estimator to sub-estimators, number of hook calls during fit, hook-call
# argument validation, lazy evaluation of hook kwargs, reconstruction of fitted
# estimators and the context repr. Uses the toy estimators/callbacks from
# sklearn.callback.tests._utils.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import numpy as np
import pytest

from sklearn.callback import CallbackSupportMixin, with_callbacks
from sklearn.callback._callback_context import (
    CallbackContext,
    _from_reconstruction_attributes,
    get_context_path,
)
from sklearn.callback.tests._utils import (
    MaxIterEstimator,
    MetaEstimator,
    NoCallbackEstimator,
    NoSubtaskEstimator,
    NotRequiredKwargsCallback,
    NotValidHookCallback,
    ParentFitEstimator,
    RecordingAutoPropagatedCallback,
    RecordingCallback,
    StopFitCallback,
    ThirdPartyEstimator,
)


def _make_callback_ctx(
    estimator, task_name="fit", task_id=0, max_subtasks=0, sequential_subtasks=True
):
    # [FUNCTION DOC]
    # HELPER: Builds a bare CallbackContext for an estimator via
    # CallbackContext._from_estimator, without registering any callbacks or hooks.
    # Inputs: estimator - estimator instance; task_name (default "fit"), task_id
    # (default 0), max_subtasks (default 0; None means unknown), sequential_subtasks
    # (default True).
    # Outputs: the new root CallbackContext.
    # Side effects: none beyond what _from_estimator does.
    """Helper function to create a callback context with default values.

    To be used instead of estimator._init_callback_context in tests that only check
    the context tree (without callbacks).
    """
    return CallbackContext._from_estimator(
        estimator, task_name, task_id, max_subtasks, sequential_subtasks
    )


def test_propagate_callback_context_autopropagated():
    # [FUNCTION DOC]
    # Verifies that propagate_callback_context only pushes auto-propagated callbacks of
    # the meta-estimator to the sub-estimator, sets _parent_callback_ctx inside the
    # context manager, and removes both attributes from the sub-estimator on exit.
    """Check that only auto-propagated callbacks are propagated."""
    not_propagated_callback = RecordingCallback()
    propagated_callback = RecordingAutoPropagatedCallback()

    estimator = MaxIterEstimator()
    metaestimator = MetaEstimator(estimator)
    metaestimator.set_callbacks(not_propagated_callback, propagated_callback)

    assert not hasattr(estimator, "_skl_callbacks")

    callback_ctx = _make_callback_ctx(metaestimator)
    with callback_ctx.propagate_callback_context(estimator):
        assert hasattr(estimator, "_parent_callback_ctx")
        assert not_propagated_callback not in estimator._skl_callbacks
        assert propagated_callback in estimator._skl_callbacks

    assert not hasattr(estimator, "_skl_callbacks")
    assert not hasattr(estimator, "_parent_callback_ctx")


def test_propagate_callback_context_clean_up():
    # [FUNCTION DOC]
    # Verifies that on exiting propagate_callback_context only the propagated callback is
    # removed: the sub-estimator's own callback list is restored to its original value and
    # _parent_callback_ctx is deleted.
    """Check that only propagated callbacks are removed on exit."""
    est_callback = RecordingCallback()
    estimator = MaxIterEstimator().set_callbacks(est_callback)

    meta_est_callback = RecordingAutoPropagatedCallback()
    metaestimator = MetaEstimator(estimator)
    metaestimator.set_callbacks(meta_est_callback)

    assert estimator._skl_callbacks == [est_callback]

    callback_ctx = _make_callback_ctx(metaestimator)
    with callback_ctx.propagate_callback_context(estimator):
        assert hasattr(estimator, "_parent_callback_ctx")
        assert est_callback in estimator._skl_callbacks
        assert meta_est_callback in estimator._skl_callbacks

    assert estimator._skl_callbacks == [est_callback]
    assert not hasattr(estimator, "_parent_callback_ctx")


def test_propagate_callback_context_no_callback():
    # [FUNCTION DOC]
    # Verifies that with no callbacks on the meta-estimator, propagation only sets
    # _parent_callback_ctx temporarily and never creates _skl_callbacks on either
    # estimator.
    """Check that no callback is propagated if there's no callback."""
    estimator = MaxIterEstimator()
    metaestimator = MetaEstimator(estimator)

    callback_ctx = _make_callback_ctx(metaestimator)
    assert len(callback_ctx._callbacks) == 0

    with callback_ctx.propagate_callback_context(estimator):
        assert hasattr(estimator, "_parent_callback_ctx")
        assert not hasattr(metaestimator, "_skl_callbacks")
        assert not hasattr(estimator, "_skl_callbacks")

    assert not hasattr(estimator, "_skl_callbacks")
    assert not hasattr(estimator, "_parent_callback_ctx")
    assert not hasattr(metaestimator, "_skl_callbacks")
    assert not hasattr(metaestimator, "_parent_callback_ctx")


def test_auto_propagated_callbacks():
    # [FUNCTION DOC]
    # Verifies that fitting a meta-estimator whose sub-estimator already holds an
    # auto-propagated callback raises TypeError with the expected message.
    """Check that it's not possible to set an auto-propagated callback on the
    sub-estimator of a meta-estimator.
    """
    estimator = MaxIterEstimator()
    estimator.set_callbacks(RecordingAutoPropagatedCallback())
    meta_estimator = MetaEstimator(estimator=estimator)

    match = (
        r"sub-estimator .*of a meta-estimator .*can't have auto-propagated callbacks"
    )
    with pytest.raises(TypeError, match=match):
        meta_estimator.fit(X=None, y=None)


def _make_task_tree(n_children, n_grandchildren):
    # [FUNCTION DOC]
    # HELPER: Builds a 3-level context tree on a MaxIterEstimator: a root with
    # n_children non-sequential children (explicit task_id i), each with n_grandchildren
    # sequential grandchildren (auto ids).
    # Inputs: n_children, n_grandchildren - non-negative ints.
    # Outputs: the root CallbackContext.
    # KNOWN ISSUE: the loop variable grandchild is assigned but never used (harmless).
    """Helper function to create a tree of tasks with a context for each task."""
    estimator = MaxIterEstimator()
    root = _make_callback_ctx(
        estimator, max_subtasks=n_children, sequential_subtasks=False
    )

    for i in range(n_children):
        child = root.subcontext(
            task_name="child task",
            task_id=i,
            max_subtasks=n_grandchildren,
        )

        for j in range(n_grandchildren):
            grandchild = child.subcontext(task_name="grandchild task")

    return root


def test_task_tree():
    # [FUNCTION DOC]
    # Verifies the structure from _make_task_tree(3, 5): parent links, path lengths per
    # depth, number of children equal to max_subtasks for every node, total node count
    # 1 + 3 + 15 via iteration, and that no node carries merge source names.
    """Check that the task tree is correctly built."""
    root = _make_task_tree(n_children=3, n_grandchildren=5)

    assert root.parent is None
    assert len(get_context_path(root)) == 1
    assert len(root._children_map) == 3

    for child in root._children_map.values():
        assert child.parent is root
        assert len(get_context_path(child)) == 2
        assert len(child._children_map) == 5

        for grandchild in child._children_map.values():
            assert grandchild.parent is child
            assert len(get_context_path(grandchild)) == 3
            assert len(grandchild._children_map) == 0

    # all _children_map lengths should match the max_subtasks attribute.
    for node in root:
        assert len(node._children_map) == node.max_subtasks

    # 1 root + 1 * 3 children + 1 * 3 * 5 grandchildren
    expected_n_nodes = np.sum(np.cumprod([1, 3, 5]))
    actual_n_nodes = sum(1 for _ in root)
    assert actual_n_nodes == expected_n_nodes

    # None of the nodes should have been merged with another node
    assert all(node.source_estimator_name is None for node in root)
    assert all(node.source_task_name is None for node in root)


def test_add_child():
    # [FUNCTION DOC]
    # Verifies _add_child: a child is registered, a duplicate task_id raises ValueError,
    # a distinct id is accepted, and exceeding max_subtasks (2) raises ValueError.
    """Sanity check for the `_add_child` method."""
    estimator = MaxIterEstimator()
    root = _make_callback_ctx(estimator, max_subtasks=2, sequential_subtasks=False)
    first_child = _make_callback_ctx(estimator, task_name="child task", task_id=0)

    root._add_child(first_child)
    assert root.max_subtasks == 2
    assert len(root._children_map) == 1
    assert first_child.task_id == 0

    # root already has a child with id 0
    second_child = _make_callback_ctx(estimator, task_name="child task", task_id=0)
    with pytest.raises(
        ValueError, match=r"Callback context .* already has a child with task_id=0"
    ):
        root._add_child(second_child)

    second_child.task_id = 1
    root._add_child(second_child)
    assert len(root._children_map) == 2

    # root can have at most 2 children
    third_child = _make_callback_ctx(estimator, task_name="child task", task_id=2)

    with pytest.raises(ValueError, match=r"Cannot add child to callback context"):
        root._add_child(third_child)


def test_merge_with():
    # [FUNCTION DOC]
    # Verifies that merging an inner estimator's root into a leaf of the outer tree
    # replaces the leaf under the same parent with the same task_id, keeps the inner
    # root's own max_subtasks/sequential_subtasks, and records the leaf's task and
    # estimator names as source_task_name/source_estimator_name.
    """Sanity check for the `_merge_with` method."""
    estimator = MaxIterEstimator()
    meta_estimator = MetaEstimator(estimator)
    outer_root = _make_callback_ctx(meta_estimator, max_subtasks=None)

    # Add a child task within the same estimator
    outer_child = outer_root.subcontext(
        task_name="child", max_subtasks=0, sequential_subtasks=True
    )

    # The root task of the inner estimator is merged with (and effectively replaces)
    # a leaf of the outer estimator because they correspond to the same formal task.
    inner_root = _make_callback_ctx(
        estimator, max_subtasks=2, sequential_subtasks=False
    )
    inner_root._merge_with(outer_child)

    assert inner_root.parent is outer_root
    assert inner_root.task_id == outer_child.task_id
    assert outer_child not in outer_root._children_map.values()
    assert inner_root in outer_root._children_map.values()

    # info concerning subtasks of the context are not inherited from other context
    assert inner_root.max_subtasks == 2
    assert not inner_root.sequential_subtasks

    # The name and estimator name of the tasks it was merged with are stored
    assert inner_root.source_task_name == outer_child.task_name
    assert inner_root.source_estimator_name == outer_child.estimator_name


def test_merge_with_error_not_leaf():
    # [FUNCTION DOC]
    # Verifies that _merge_with raises ValueError when the target context already has
    # children (is not a leaf).
    """Check that merging with a non-leaf node raises an error."""
    estimator = MaxIterEstimator()
    inner_root = _make_callback_ctx(estimator)

    meta_estimator = MetaEstimator(estimator)
    outer_root = _make_callback_ctx(meta_estimator, max_subtasks=None)

    # Add a child task within the same estimator
    outer_root.subcontext(task_name="child")

    with pytest.raises(ValueError, match=r"Cannot merge callback context"):
        inner_root._merge_with(outer_root)


@pytest.mark.parametrize(
    "estimator_class", [MaxIterEstimator, ThirdPartyEstimator, ParentFitEstimator]
)
def test_callback_ctx_removed_after_fit(estimator_class):
    # [FUNCTION DOC]
    # Verifies, for several toy estimator classes (parametrized), that the temporary
    # _callback_fit_ctx attribute is gone after fit.
    # Inputs: estimator_class - parametrized estimator class.
    """Check that the _callback_fit_ctx attribute gets removed after fit."""
    estimator = estimator_class().fit()
    assert not hasattr(estimator, "_callback_fit_ctx")


def test_inner_estimator_no_callback_support():
    # [FUNCTION DOC]
    # Verifies that a meta-estimator with an auto-propagated callback and a sub-estimator
    # lacking callback support fits without error but emits a UserWarning.
    """Check that meta estimators can have sub estimators without callback support.

    No error is raised when the sub-estimator does not support callbacks. If callbacks
    would be propagated, a warning is raised instead.
    """
    estimator = NoCallbackEstimator()
    meta_estimator = MetaEstimator(estimator)
    meta_estimator.set_callbacks(RecordingAutoPropagatedCallback())

    with pytest.warns(
        UserWarning,
        match="The estimator NoCallbackEstimator does not support callbacks.",
    ):
        meta_estimator.fit()


def test_estimator_without_subtask():
    # [FUNCTION DOC]
    # Smoke test: an estimator that never calls call_on_fit_task_end on subtasks can be
    # fitted with a RecordingCallback without errors.
    """Check that callback support works for an estimator without subtasks.

    This test is about verifying that an estimator that does not call its callback
    context's `call_on_fit_task_end` does not cause a problem.
    """
    estimator = NoSubtaskEstimator()
    estimator.set_callbacks(RecordingCallback())
    estimator.fit()


@pytest.mark.parametrize(
    "Callback", [RecordingAutoPropagatedCallback, RecordingCallback]
)
def test_callback_hooks_called(Callback):
    # [FUNCTION DOC]
    # Verifies hook call counts on a plain MaxIterEstimator: setup/teardown once, and
    # 1 + max_iter task begins/ends, for both callback kinds.
    # Inputs: Callback - parametrized callback class.
    """Check the number of callback hook calls in a regular estimator.

    For a regular estimator, it does not depend whether it's an autopropagated callback
    or not.
    """
    max_iter = 10
    callback = Callback()
    MaxIterEstimator(max_iter=max_iter).set_callbacks(callback).fit()
    assert callback.count_hooks("setup") == 1
    # 1 root + max_iter leaves
    assert callback.count_hooks("on_fit_task_begin") == 1 + max_iter
    assert callback.count_hooks("on_fit_task_end") == 1 + max_iter
    assert callback.count_hooks("teardown") == 1


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_meta_estimator_autopropagated_callback_hooks_called(n_jobs):
    # [FUNCTION DOC]
    # Verifies that an auto-propagated callback on MetaEstimator sees one setup/teardown
    # and task begin/end counts equal to the size of the full combined task tree
    # (1 + n_outer + n_outer*n_inner + n_outer*n_inner*max_iter), sequential and parallel.
    # Inputs: n_jobs - parametrized 1 or 2.
    """Check the number of callback hook calls in a meta-estimator.

    For an auto-propagated callback, setup and teardown are called only once,
    by the meta-estimator. To count the number of task ends, we need to aggregate the
    number of tasks from all the levels of the global task tree (which contains the
    task tree of the meta-estimator and the task trees of each sub-estimator).
    """

    n_outer, n_inner, max_iter = 2, 3, 5
    callback = RecordingAutoPropagatedCallback()
    MetaEstimator(
        MaxIterEstimator(max_iter=max_iter),
        n_outer=n_outer,
        n_inner=n_inner,
        n_jobs=n_jobs,
    ).set_callbacks(callback).fit()

    assert callback.count_hooks("setup") == 1
    expected_n_tasks = np.sum(np.cumprod([1, n_outer, n_inner, max_iter]))
    assert callback.count_hooks("on_fit_task_begin") == expected_n_tasks
    assert callback.count_hooks("on_fit_task_end") == expected_n_tasks
    assert callback.count_hooks("teardown") == 1


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_meta_estimator_callback_hooks_called(n_jobs):
    # [FUNCTION DOC]
    # Verifies that a non-propagated callback set on the sub-estimator gets
    # setup/teardown once per sub-estimator fit (n_outer*n_inner) and 1 + max_iter task
    # begins/ends per fit.
    # Inputs: n_jobs - parametrized 1 or 2.
    """Check the number of callback hook calls in a meta-estimator.

    For a non auto-propagated callback, setup and teardown are called once for
    each fit of the sub-estimator. The number of task ends is the sum of the number of
    task ends from all the sub-estimators.
    """
    n_outer, n_inner, max_iter = 2, 3, 5
    callback = RecordingCallback()
    est = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    MetaEstimator(est, n_outer=n_outer, n_inner=n_inner, n_jobs=n_jobs).fit()

    n_fits = n_outer * n_inner
    assert callback.count_hooks("setup") == n_fits
    # 1 root + max_iter leaves
    assert callback.count_hooks("on_fit_task_begin") == n_fits * (1 + max_iter)
    assert callback.count_hooks("on_fit_task_end") == n_fits * (1 + max_iter)
    assert callback.count_hooks("teardown") == n_fits


def test_autopropagation_to_callback_agnostic_subestimator():
    # [FUNCTION DOC]
    # Verifies that with a sub-estimator without callback support, the auto-propagated
    # callback only sees the meta-estimator's tree nodes (1 + n_outer + n_outer*n_inner)
    # and a UserWarning is emitted.
    """Check the number of hook calls when the sub-estimator doesn't support callbacks.

    The number of task begins and ends is just the number of nodes in the context tree
    of the meta-estimator.
    """
    n_outer, n_inner = 2, 3
    callback = RecordingAutoPropagatedCallback()
    meta_estimator = MetaEstimator(
        NoCallbackEstimator(), n_outer=n_outer, n_inner=n_inner
    ).set_callbacks(callback)

    with pytest.warns(
        UserWarning,
        match="The estimator NoCallbackEstimator does not support callbacks.",
    ):
        meta_estimator.fit()

    assert callback.count_hooks("setup") == 1
    expected_n_tasks = np.sum(np.cumprod([1, n_outer, n_inner]))
    assert callback.count_hooks("on_fit_task_begin") == expected_n_tasks
    assert callback.count_hooks("on_fit_task_end") == expected_n_tasks
    assert callback.count_hooks("teardown") == 1


# TODO(callbacks): should be a common test in a dev test suite instead of a check
# in the hook calls to avoid repeating the same check for each call of the same hook.
def test_hook_calling_invalid_kwargs_out():
    # [FUNCTION DOC]
    # Verifies that calling a hook on a callback whose hook signature has invalid
    # parameters raises TypeError.
    """Check that a callback with invalid kwargs in its signatures raises an error."""
    estimator = MaxIterEstimator()
    context = estimator.set_callbacks(NotValidHookCallback())._init_callback_context()
    msg = r"on_fit_task_begin .* has parameters that are not valid"
    with pytest.raises(TypeError, match=msg):
        context.call_on_fit_task_begin(estimator=estimator, X=1, y=2)


def test_hook_calling_return_value():
    # [FUNCTION DOC]
    # Verifies call_on_fit_task_end returns False when no callback requests a stop, and
    # True when a StopFitCallback is among the callbacks.
    """Check the return value of the hook calls."""
    estimator = MaxIterEstimator()
    context = estimator.set_callbacks(RecordingCallback())._init_callback_context()
    result = context.call_on_fit_task_end(estimator=estimator)
    # RecordingCallback.on_fit_task_end does not return a value (interpreted as False)
    assert result is False

    estimator.set_callbacks(RecordingCallback(), StopFitCallback())
    result = estimator._init_callback_context().call_on_fit_task_end(
        estimator=estimator
    )
    # StopFitCallback.on_fit_task_end returns True
    assert result is True


def test_hook_calling_lazy_evaluation():
    # [FUNCTION DOC]
    # Verifies that callable kwargs passed to hook calls are evaluated lazily: unused
    # ones are never evaluated, and one used by two callbacks is evaluated once. Counts
    # evaluations in a local dict through eval_kwarg.
    """Check lazy evaluation of kwargs.

    kwargs are not evaluated if no callback uses them.
    They are evaluated only once and passed to all callbacks that use them.
    """
    eval_counts = {"X": 0, "metadata": 0}

    def eval_kwarg(key):
        # [FUNCTION DOC]
        # HELPER: Increments the evaluation counter for key in the enclosing eval_counts
        # dict and returns 1.
        # Side effects: mutates the closed-over eval_counts (rebound by the outer test).
        eval_counts[key] += 1
        return 1

    # unused kwarg is not evaluated
    estimator = MaxIterEstimator()
    callback = NotRequiredKwargsCallback()
    context = estimator.set_callbacks(callback)._init_callback_context()
    context.call_on_fit_task_end(
        estimator=estimator,
        X=partial(eval_kwarg, "X"),
        metadata=partial(eval_kwarg, "metadata"),
    )
    assert eval_counts["X"] == 1
    assert eval_counts["metadata"] == 0

    # kwarg used twice is evaluated only once
    eval_counts = {"X": 0}
    estimator.set_callbacks(RecordingCallback(), RecordingCallback())
    context = estimator._init_callback_context()
    context.call_on_fit_task_begin(estimator=estimator, X=partial(eval_kwarg, "X"))
    assert eval_counts["X"] == 1


def test_hook_calling_lazy_evaluation_reconstruction_attributes():
    # [FUNCTION DOC]
    # Verifies that reconstruction_attributes is consumed by the context: callbacks
    # receive a fitted_estimator (a MaxIterEstimator with n_iter_ == 1) instead of the
    # raw reconstruction_attributes kwarg.
    """Check lazy evaluation of reconstruction_attributes.

    "reconstruction_attributes" is processed by the context and used to create a
    fitted estimator that is passed to the callback as "fitted_estimator".
    """
    estimator = MaxIterEstimator()
    callback = RecordingCallback()
    context = estimator.set_callbacks(callback)._init_callback_context()
    context.call_on_fit_task_end(
        estimator=estimator, reconstruction_attributes=lambda: {"n_iter_": 1}
    )
    assert "reconstruction_attributes" not in callback.record[-1]["kwargs"]
    assert "fitted_estimator" in callback.record[-1]["kwargs"]

    fitted_estimator = callback.record[-1]["kwargs"]["fitted_estimator"]
    assert isinstance(fitted_estimator, MaxIterEstimator)
    assert fitted_estimator.n_iter_ == 1


# TODO(callbacks): check that the reconstructed estimator can be used to predict
# while the original cannot.
def test_from_reconstruction_attributes():
    # [FUNCTION DOC]
    # Verifies _from_reconstruction_attributes returns a new instance of the same class
    # with identical params and the given fitted attribute set.
    """Test the _from_reconstruction_attributes helper function."""
    max_iter = 5
    estimator = MaxIterEstimator(max_iter=max_iter)
    reconstructed_est = _from_reconstruction_attributes(estimator, {"n_iter_": 2})
    assert isinstance(reconstructed_est, MaxIterEstimator)
    assert reconstructed_est is not estimator
    assert reconstructed_est.get_params() == estimator.get_params()
    assert reconstructed_est.n_iter_ == 2


def test_subcontext_task_id_ordering():
    # [FUNCTION DOC]
    # Verifies that with sequential_subtasks=True, subcontext assigns task ids 0..4 in
    # creation order.
    """Check that the task_id is automatically assigned in the correct order."""
    estimator = MaxIterEstimator()
    context = _make_callback_ctx(estimator, max_subtasks=5, sequential_subtasks=True)

    for i in range(5):
        subcontext = context.subcontext()
        assert subcontext.task_id == i


def test_subcontext_task_id_ordering_error():
    # [FUNCTION DOC]
    # Verifies that subcontext raises ValueError when a task_id is given for sequential
    # subtasks, and when it is missing for non-sequential subtasks.
    """Check errors raised when task_id and sequential_subtasks are inconsistent.

    - if sequential_subtasks is True, children task_ids must be left to None
    - if sequential_subtasks is False, children task_ids must be provided
    """
    estimator = MaxIterEstimator()
    context = _make_callback_ctx(estimator, max_subtasks=1, sequential_subtasks=True)
    with pytest.raises(
        ValueError,
        match=(
            "task_id for MaxIterEstimator child_task must be None if "
            "sequential_subtasks is True for fit."
        ),
    ):
        context.subcontext(task_name="child_task", task_id=0)

    context = _make_callback_ctx(estimator, max_subtasks=1, sequential_subtasks=False)
    with pytest.raises(
        ValueError,
        match=(
            "task_id for MaxIterEstimator child_task must be provided if "
            "sequential_subtasks is False for fit."
        ),
    ):
        context.subcontext(task_name="child_task")


def test_locally_defined_estimator():
    # [FUNCTION DOC]
    # Verifies that a class defined inside a function and using CallbackSupportMixin plus
    # with_callbacks gets exactly one call of each hook during fit.
    """Check registering a callback on a locally defined estimator."""

    class LocallyDefinedEstimator(CallbackSupportMixin):
        # [CLASS DOC]
        # HELPER: Minimal local estimator with callback support and a single root task.
        @with_callbacks
        def fit(self, X=None, y=None):
            # [FUNCTION DOC]
            # Creates the root callback context and calls the task begin and end hooks once.
            # Inputs: X, y - ignored. Outputs: self.
            callback_ctx = self._init_callback_context()
            callback_ctx.call_on_fit_task_begin(estimator=self)

            callback_ctx.call_on_fit_task_end(estimator=self)
            return self

    estimator = LocallyDefinedEstimator()
    callback = RecordingCallback()
    estimator.set_callbacks(callback)
    estimator.fit()
    assert callback.count_hooks("setup") == 1
    assert callback.count_hooks("on_fit_task_begin") == 1
    assert callback.count_hooks("on_fit_task_end") == 1
    assert callback.count_hooks("teardown") == 1


def test_callback_context_repr():
    # [FUNCTION DOC]
    # Verifies the exact repr string of a CallbackContext.
    """Smoke test for the repr of the callback context."""
    estimator = MaxIterEstimator()
    context = _make_callback_ctx(estimator, task_name="mytask", task_id=42)
    expected_repr = (
        "CallbackContext(estimator_name='MaxIterEstimator', task_name='mytask', "
        "task_id=42)"
    )
    assert repr(context) == expected_repr
