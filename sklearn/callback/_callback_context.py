# [MODULE DOC]
# Task-tree context objects passed to callback hooks, plus helpers to dispatch hooks,
# build snapshot "fitted" estimators and compute a context's ancestor path.
# Module state: VALID_HOOK_PARAMS_OUT lists the keyword-only hook parameters allowed;
# _cached_signature is an unbounded lru_cache around inspect.signature (it keeps
# references to the bound hook methods, and therefore callbacks, alive).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import copy
import functools
import inspect
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone

from sklearn.callback._base import AutoPropagatedCallback

# List of the parameters expected to be in the hooks signatures
VALID_HOOK_PARAMS_OUT = ["X", "y", "metadata", "fitted_estimator"]


_cached_signature = functools.lru_cache()(inspect.signature)


class CallbackContext:
    # [CLASS DOC]
    # Node of a tree of fit tasks. Private state: _callbacks (list shared by reference with
    # the estimator's _skl_callbacks and with all descendant contexts of that estimator),
    # _children_map (task_id -> child context, insertion ordered), _propagation_depth
    # (number of meta-estimator levels above), and optionally _propagated_callbacks (set
    # by propagate_callback_context). Not created via __init__ (constructors use __new__).
    """Task level context for the callbacks.

    This class is responsible for managing the callbacks and holding the tree structure
    of an estimator's tasks. Each instance corresponds to a task of the estimator.

    This class should not be instantiated directly, but through the
    `_init_callback_context` method of the estimator to create the root context or using
    the `subcontext` method of this class to create sub-contexts.

    These contexts are passed to the callback hooks to be able to keep track of the
    position of a task in the task tree from within the callbacks.

    Attributes
    ----------
    task_name : str
        The name of the task this context is responsible for.

    task_id : int
        The identifier of the task this context is responsible for. It uniquely
        identifies the task among its siblings.

    max_subtasks : int or None
        The maximum number of children tasks for this task. 0 means it's a leaf.
        None means the maximum number of subtasks is not known in advance.

    sequential_subtasks : bool
        Whether this context's subtasks are sequential. When True, children contexts'
        have consecutive integer task_ids starting from 0.

    estimator_name : str
        The name of the estimator that holds this context.

    parent : CallbackContext or None
        The parent context of this context. None if this context is the root.

    root_uuid : uuid.UUID instance
        The UUID of the root context. All contexts in the same task tree have the same
        root UUID that is used to identify the task tree itself.

    init_time : datetime.datetime
        The time when the context was initialised, in the UTC timezone.

    source_estimator_name : str or None
        The name of the estimator that holds the parent task this task was
        merged with. None if it was not merged with another context.

    source_task_name : str or None
        The task name of the parent task this task was merged with. None if it
        was not merged with another context.
    """

    @classmethod
    def _from_estimator(
        cls, estimator, task_name, task_id, max_subtasks, sequential_subtasks
    ):
        # [FUNCTION DOC]
        # HELPER: Builds the root context for an estimator's fit.
        # Inputs: as documented; estimator is not stored (avoids a reference cycle).
        # Steps: set all public attributes (new root_uuid, UTC init_time, no parent, empty
        # children). If the estimator carries _parent_callback_ctx (it is a sub-estimator
        # inside propagate_callback_context), merge into that parent leaf (see _merge_with)
        # and set depth to parent depth + 1; otherwise depth 0.
        # Outputs: new CallbackContext. Exceptions: ValueError from _merge_with.
        """Private constructor to create a root context.

        Parameters
        ----------
        estimator : estimator instance
            The estimator this context is responsible for.

        task_name : str
            The name of the root task.

        task_id : int
            Identifier for the root task.

        max_subtasks : int or None
            The maximum number of subtasks that can be children of the root task. None
            means the maximum number of subtasks is not known in advance. 0 means it's a
            leaf.

        sequential_subtasks : bool
            Whether the root context has sequential subtasks. If True, children contexts
            created via `subcontext` will have automatically assigned consecutive
            integer task_ids starting from 0.
        """
        new_ctx = cls.__new__(cls)

        # We don't store the estimator in the context to avoid circular references
        # because the estimator already holds a reference to the context.
        new_ctx._callbacks = getattr(estimator, "_skl_callbacks", [])
        new_ctx.estimator_name = estimator.__class__.__name__
        new_ctx.task_name = task_name
        new_ctx.task_id = task_id
        new_ctx.max_subtasks = max_subtasks
        new_ctx.sequential_subtasks = sequential_subtasks
        new_ctx.parent = None
        new_ctx.root_uuid = uuid.uuid4()
        new_ctx.init_time = datetime.now(timezone.utc)
        new_ctx._children_map = {}
        new_ctx.source_estimator_name = None
        new_ctx.source_task_name = None

        if hasattr(estimator, "_parent_callback_ctx"):
            # This context's task is the root task of the estimator which itself
            # corresponds to a leaf task of a meta-estimator. Both tasks actually
            # represent the same task so we merge both tasks into a single task,
            # attaching the task tree of the sub-estimator to the task tree of
            # the meta-estimator on the way.
            parent_ctx = estimator._parent_callback_ctx
            new_ctx._merge_with(parent_ctx)
            new_ctx._propagation_depth = parent_ctx._propagation_depth + 1
        else:
            new_ctx._propagation_depth = 0

        return new_ctx

    @classmethod
    def _from_parent(
        cls, parent_context, *, task_name, task_id, max_subtasks, sequential_subtasks
    ):
        # [FUNCTION DOC]
        # HELPER: Builds a child context of parent_context with the given task fields,
        # inheriting callbacks, estimator_name, depth and root_uuid; fresh init_time.
        # Outputs: new context, registered via parent_context._add_child (which sets parent).
        # Exceptions: ValueError from _add_child (duplicate id or too many children).
        """Private constructor to create a sub-context.

        Parameters
        ----------
        parent_context : `CallbackContext` instance
            The parent context of the new context.

        task_name : str
            The name of the task this context is responsible for.

        task_id : int
            The identifier of the task this context is responsible for.

        max_subtasks : int or None
            The maximum number of tasks that can be children of the task this context is
            responsible for. 0 means it's a leaf. None means the maximum number of
            subtasks is not known in advance.

        sequential_subtasks : bool
            Whether this context's subtasks are sequential. If True, children contexts
            created via `subcontext` will have automatically assigned consecutive
            integer task_ids starting from 0.
        """
        new_ctx = cls.__new__(cls)

        new_ctx._callbacks = parent_context._callbacks
        new_ctx.estimator_name = parent_context.estimator_name
        new_ctx._propagation_depth = parent_context._propagation_depth
        new_ctx.task_name = task_name
        new_ctx.task_id = task_id
        new_ctx.max_subtasks = max_subtasks
        new_ctx.sequential_subtasks = sequential_subtasks
        new_ctx.root_uuid = parent_context.root_uuid
        new_ctx.init_time = datetime.now(timezone.utc)
        new_ctx.parent = None
        new_ctx._children_map = {}
        new_ctx.source_estimator_name = None
        new_ctx.source_task_name = None

        # This task is a subtask of another task of a same estimator
        parent_context._add_child(new_ctx)

        return new_ctx

    def __iter__(self):
        # [FUNCTION DOC]
        # Yields this context then, recursively, each child subtree in insertion order
        # (pre-order DFS).
        """Pre-order depth-first traversal of the task tree."""
        yield self
        for context in self._children_map.values():
            yield from context

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "CallbackContext(estimator_name=..., task_name=..., task_id=...)".
        return (
            f"{self.__class__.__name__}("
            f"estimator_name={self.estimator_name!r}, "
            f"task_name={self.task_name!r}, "
            f"task_id={self.task_id})"
        )

    def _add_child(self, child_context):
        # [FUNCTION DOC]
        # HELPER: Registers child_context under its task_id and sets its parent to self.
        # Exceptions: ValueError if task_id already used, or if max_subtasks is not None and
        # the number of children already reached it.
        """Add `child_context` as a child of this context."""
        if child_context.task_id in self._children_map:
            raise ValueError(
                f"Callback context {self.task_name} of estimator "
                f"{self.estimator_name} already has a child with "
                f"task_id={child_context.task_id}."
            )

        if (
            self.max_subtasks is not None
            and len(self._children_map) >= self.max_subtasks
        ):
            raise ValueError(
                f"Cannot add child to callback context {self.task_name} of estimator "
                f"{self.estimator_name} because it already has its maximum "
                f"number of children ({self.max_subtasks})."
            )

        self._children_map[child_context.task_id] = child_context
        child_context.parent = self

    def _merge_with(self, other_context):
        # [FUNCTION DOC]
        # HELPER: Makes self (a sub-estimator root) replace other_context (a meta-estimator
        # leaf) in the tree: takes its parent, task_id and root_uuid, overwrites the parent's
        # children map entry, and records source_task_name/source_estimator_name.
        # Exceptions: ValueError if other_context is not a leaf (max_subtasks != 0).
        # KNOWN ISSUE: if other_context is itself a root (parent None, e.g. a meta-estimator
        # whose root task is a leaf) this raises AttributeError on the parent's children map.
        """Merge this context with `other_context`.

        This method is called on a sub-estimator's root task to merge it with a
        meta-estimator's leaf task. The sub-estimator's task tree is therefore attached
        to the meta-estimator's task tree. The root node of the sub-estimator's task
        tree and the leaf node of the meta-estimator's task tree are both represented
        by a single node in this combined task tree.
        """
        if other_context.max_subtasks != 0:
            raise ValueError(
                f"Cannot merge callback context (task {self.task_name!r} of estimator "
                f"{self.estimator_name}) with callback context "
                f"(task {other_context.task_name!r} of estimator "
                f"{other_context.estimator_name}) because the latter is not a leaf."
            )

        # Set the parent of the sub-estimator's root context to the parent of the
        # meta-estimator's leaf context
        self.parent = other_context.parent
        self.task_id = other_context.task_id
        self.root_uuid = other_context.root_uuid
        other_context.parent._children_map[self.task_id] = self

        # Keep information about the context it was merged with
        self.source_task_name = other_context.task_name
        self.source_estimator_name = other_context.estimator_name

    def subcontext(
        self, task_name="", task_id=None, max_subtasks=0, sequential_subtasks=True
    ):
        # [FUNCTION DOC]
        # Creates and attaches a child context.
        # Inputs: task_name (str, ""), task_id (int or None), max_subtasks (int/None, 0 = leaf),
        # sequential_subtasks (bool, True).
        # Behavior: when this context is sequential, task_id must be None and becomes the
        # current number of children; when not sequential, task_id is required.
        # Outputs: the new child CallbackContext.
        # Exceptions: ValueError for the task_id rule violations above, or from _add_child.
        """Create a context for a subtask of the current task.

        Parameters
        ----------
        task_name : str, default=""
            The name of the subtask.

        task_id : int or None, default=None
            An identifier of the subtask. It must be distinct from the task_ids of its
            siblings. If None, task_id is automatically set to the next available
            integer task_id.

        max_subtasks : int or None, default=0
            The maximum number of tasks that can be children of the subtask. 0 means
            it's a leaf. None means the maximum number of subtasks is not known in
            advance.

        sequential_subtasks : bool, default=True
            Whether the new context's subtasks are sequential. If True, children
            contexts of the new context, created via `subcontext`, will have
            automatically assigned consecutive integer task_ids starting from 0.
        """
        if self.sequential_subtasks and task_id is not None:
            raise ValueError(
                f"task_id for {self.estimator_name} {task_name} must be None if "
                f"sequential_subtasks is True for {self.task_name}."
            )
        if not self.sequential_subtasks and task_id is None:
            raise ValueError(
                f"task_id for {self.estimator_name} {task_name} must be provided "
                f"if sequential_subtasks is False for {self.task_name}."
            )
        if task_id is None:
            task_id = len(self._children_map)

        return CallbackContext._from_parent(
            parent_context=self,
            task_name=task_name,
            task_id=task_id,
            max_subtasks=max_subtasks,
            sequential_subtasks=sequential_subtasks,
        )

    def _call_hooks(self, estimator, hook_name, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Dispatches hook_name on each callback of this context and ORs results.
        # Inputs: estimator; hook_name (str); kwargs - candidate argument values (X, y,
        # metadata, reconstruction_attributes), each possibly a zero-arg callable for laziness.
        # Steps: for each callback not in _propagated_callbacks: read the hook's keyword-only
        # parameter names (cached signature); reject unknown names; for each requested name
        # compute its value once per call (calling it if callable). "fitted_estimator" is built
        # from reconstruction_attributes via _from_reconstruction_attributes (None if absent).
        # Call the hook with (estimator, self, requested kwargs) and OR its truthiness.
        # Outputs: bool. Exceptions: TypeError for invalid hook parameter names.
        # KNOWN ISSUE: the inline comment says only on_fit_task_end is skipped for propagated
        # callbacks, but every hook dispatched here is skipped for them. Also a value that is
        # legitimately callable (e.g. an estimator-like object with __call__) would be invoked.
        """Helper to call the hook of all callbacks with their respective arguments.

        Provide the right arguments to each hook by inspecting their signatures. Any
        value that is a callable is replaced by what it returns to allow lazy loading of
        the arguments.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling the callback hook.

        hook_name : str
            Name of the callback hook to call.

        **kwargs: dict
            Optional keyword arguments passed to the callback context.

        Returns
        -------
        result : bool
            True if any hook call returned True. False otherwise.
        """
        result = False

        # Keep a cache of the evaluated args to evaluate them only once.
        evaluated_args = {}

        for callback in self._callbacks:
            if callback in getattr(self, "_propagated_callbacks", []):
                # Only call the `on_fit_task_end` hook of callbacks that are not
                # propagated. For propagated callbacks, the hook will be called by the
                # sub-estimator's root context (both represent the same task).
                continue

            signature = _cached_signature(getattr(callback, hook_name))
            params_names = {
                p.name
                for p in signature.parameters.values()
                if p.kind == p.KEYWORD_ONLY
            }
            if diff := set(params_names) - set(VALID_HOOK_PARAMS_OUT):
                raise TypeError(
                    f"Hook {hook_name} of the callback {callback.__class__.__name__} "
                    f"has parameters that are not valid: {diff}. The valid parameters "
                    f"are: {VALID_HOOK_PARAMS_OUT}."
                )

            args_to_pass = {}
            for param_name in params_names:
                if param_name not in evaluated_args:
                    # Special case: "reconstruction_attributes" is not directly passed
                    # to the hook. A ready to predict/transform estimator is created
                    # from these attributes and passed to the hook as "fitted_estimator"
                    if param_name == "fitted_estimator":
                        attrs = kwargs.get("reconstruction_attributes", None)
                        attrs = attrs() if callable(attrs) else attrs
                        new_est = (
                            _from_reconstruction_attributes(estimator, attrs)
                            if attrs is not None
                            else None
                        )
                        evaluated_args["fitted_estimator"] = new_est
                    else:
                        val = kwargs.get(param_name, None)
                        val = val() if callable(val) else val
                        evaluated_args[param_name] = val

                args_to_pass[param_name] = evaluated_args[param_name]

            result |= bool(
                getattr(callback, hook_name)(estimator, self, **args_to_pass)
            )

        return result

    def call_on_fit_task_begin(
        self,
        *,
        estimator,
        X=None,
        y=None,
        metadata=None,
        reconstruction_attributes=None,
    ):
        # [FUNCTION DOC]
        # Dispatches on_fit_task_begin to callbacks (see _call_hooks); the hook results are
        # discarded. Inputs: keyword-only estimator, X, y, metadata, reconstruction_attributes
        # (values or zero-arg callables). Outputs: self (allows chaining/assignment).
        """Call the `on_fit_task_begin` hook of the callbacks.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling the callback hook.

        X : array-like or None, default=None
            The training data of the current task.

        y : array-like or None, default=None
            The training targets of the current task.

        metadata : dict or None, default=None
            A dictionary containing training metadata for the current task.

        reconstruction_attributes : dict or None, default=None
            A dictionary of the sufficient fitted attributes needed to construct a
            `fitted_estimator` from the current state of the estimator, i.e. an
            estimator instance ready to predict, transform, etc ... as if the fit had
            stopped at the beginning of this task. The `fitted_estimator` is the
            object that will be passed to the callbacks, if required.
        """
        self._call_hooks(
            estimator,
            hook_name="on_fit_task_begin",
            X=X,
            y=y,
            metadata=metadata,
            reconstruction_attributes=reconstruction_attributes,
        )
        return self

    def call_on_fit_task_end(
        self,
        *,
        estimator,
        X=None,
        y=None,
        metadata=None,
        reconstruction_attributes=None,
    ):
        # [FUNCTION DOC]
        # Dispatches on_fit_task_end (see _call_hooks). Inputs as call_on_fit_task_begin.
        # Outputs: bool, True if any callback requested stopping.
        """Call the `on_fit_task_end` hook of the callbacks.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling the callback hook.

        X : array-like or None, default=None
            The training data of the current task.

        y : array-like or None, default=None
            The training targets of the current task.

        metadata : dict or None, default=None
            A dictionary containing training metadata of the current task.

        reconstruction_attributes : dict or None, default=None
            A dictionary of the sufficient fitted attributes needed to construct a
            `fitted_estimator` from the current state of the estimator, i.e. an
            estimator instance ready to predict, transform, etc ... as if the fit had
            stopped at the end of this task. The `fitted_estimator` is the object
            that will be passed to the callbacks, if required.

        Returns
        -------
        stop : bool
            Whether or not to stop the current level of iterations at this end of this
            task.
        """
        return self._call_hooks(
            estimator,
            hook_name="on_fit_task_end",
            X=X,
            y=y,
            metadata=metadata,
            reconstruction_attributes=reconstruction_attributes,
        )

    @contextmanager
    def propagate_callback_context(self, sub_estimator):
        # [FUNCTION DOC]
        # Context manager wrapping a sub-estimator fit inside this (leaf) task.
        # Inputs: sub_estimator - estimator about to be fitted.
        # Steps: refuse sub-estimators that already own auto-propagated callbacks; set
        # sub_estimator._parent_callback_ctx to self (always, for tree merging); select this
        # context's auto-propagated callbacks whose max_propagation_depth is None or greater
        # than this depth; if the sub-estimator lacks set_callbacks, warn and propagate none;
        # otherwise record them in self._propagated_callbacks and append them to the
        # sub-estimator's callbacks. Yields None. On exit, remove the propagated callbacks
        # from the sub-estimator and delete _parent_callback_ctx.
        # Exceptions: TypeError for pre-existing auto-propagated callbacks.
        # Side effects: mutates sub_estimator and self; emits UserWarning as described.
        # KNOWN ISSUE: self._propagated_callbacks is never reset on exit, so it persists on
        # this context after the block.
        """Propagate the context and callbacks to a sub-estimator.

        Clear the propagated callbacks from the sub-estimator on exit.

        Only auto-propagated callbacks are propagated to the sub-estimator. An error is
        raised if the sub-estimator already holds auto-propagated callbacks.

        The sub-estimator receives this context as an attribute named
        `_parent_callback_ctx` so that the meta-estimator's task tree can be merged with
        the sub-estimator's one.

        Parameters
        ----------
        sub_estimator : estimator instance
            The estimator to propagate the callbacks and context to.
        """
        bad_callbacks = [
            callback.__class__.__name__
            for callback in getattr(sub_estimator, "_skl_callbacks", [])
            if isinstance(callback, AutoPropagatedCallback)
        ]
        if bad_callbacks:
            raise TypeError(
                f"The sub-estimator ({sub_estimator.__class__.__name__}) of a"
                f" meta-estimator ({self.estimator_name}) can't have"
                f" auto-propagated callbacks ({bad_callbacks})."
                " Register them directly on the meta-estimator."
            )

        # We store the parent context in the sub-estimator to be able to merge the task
        # trees of the sub-estimator and the meta-estimator. We want to link the task
        # trees even if there is no callback to propagate, as the sub-estimators might
        # have non auto-propagated callbacks, which would need to have access to the
        # whole tree.
        sub_estimator._parent_callback_ctx = self

        callbacks_to_propagate = [
            callback
            for callback in self._callbacks
            if isinstance(callback, AutoPropagatedCallback)
            and (
                callback.max_propagation_depth is None
                or self._propagation_depth < callback.max_propagation_depth
            )
        ]
        if callbacks_to_propagate and not hasattr(sub_estimator, "set_callbacks"):
            warnings.warn(
                f"The estimator {sub_estimator.__class__.__name__} does not support "
                f"callbacks. The callbacks attached to {self.estimator_name} will not "
                f"be propagated to this estimator."
            )
            callbacks_to_propagate = []

        if callbacks_to_propagate:
            self._propagated_callbacks = callbacks_to_propagate
            curr_callbacks = getattr(sub_estimator, "_skl_callbacks", [])
            sub_estimator.set_callbacks(*(curr_callbacks + callbacks_to_propagate))

        try:
            yield
        finally:
            if callbacks_to_propagate:
                kept_callbacks = [
                    cb
                    for cb in sub_estimator._skl_callbacks
                    if cb not in callbacks_to_propagate
                ]
                sub_estimator.set_callbacks(*kept_callbacks)
            del sub_estimator._parent_callback_ctx


def _from_reconstruction_attributes(estimator, reconstruction_attributes):
    # [FUNCTION DOC]
    # HELPER: Shallow-copies estimator and sets each key/value of reconstruction_attributes
    # on the copy as attributes. Outputs: the copy. The original is not modified, but
    # mutable attributes are shared with it.
    """Return a copy of the estimator as if it was fitted.

    Parameters
    ----------
    estimator : estimator instance
        The estimator from which to make a ready-to-be-evaluated copy.

    reconstruction_attributes : dict
        A dictionary containing the necessary attributes to create a working
        fitted estimator from this instance.

    Returns
    -------
    fitted_estimator : estimator instance
        The fitted copy of this estimator.
    """
    new_estimator = copy.copy(estimator)
    for key, val in reconstruction_attributes.items():
        setattr(new_estimator, key, val)
    return new_estimator


def get_context_path(context):
    # [FUNCTION DOC]
    # HELPER: Recursively walks parent links to return the list of contexts from the root
    # down to and including context. Outputs: list of CallbackContext.
    """Helper function to get the path from the root context down to a given context.

    Parameters
    ----------
    context : `CallbackContext` instance
        The context to get the path to.

    Returns
    -------
    list of `CallbackContext` instances
        The list of the ancestors (itself included) of the given context. The list is
        ordered from the root context to the given context.
    """
    return (
        [context]
        if context.parent is None
        else get_context_path(context.parent) + [context]
    )
