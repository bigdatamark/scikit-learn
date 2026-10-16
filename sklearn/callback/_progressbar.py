# [MODULE DOC]
# ProgressBar callback rendering nested rich progress bars for an estimator's task tree.
# Hook events are sent through the _transport module to a per-fit queue consumed by a
# RichProgressMonitor thread. Module state: _run_queues and _run_monitors, keyed by
# root_uuid (unpicklable objects kept out of the callback). The rich-dependent helper
# classes are only defined if rich imports successfully.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral
from queue import Queue
from threading import Thread

from sklearn.callback._callback_context import get_context_path
from sklearn.callback._transport import close_listener, open_listener, send
from sklearn.utils._optional_dependencies import check_rich_support
from sklearn.utils._param_validation import Interval, validate_params

# Per-fit local queues and monitors, keyed by the run's `root_uuid`. Both are not
# picklable so they live here rather than on the callback instance. Entries are added in
# `setup` and removed in `teardown`.
_run_queues = {}
_run_monitors = {}


class ProgressBar:
    # [CLASS DOC]
    # Auto-propagated FitCallback. Attributes: max_propagation_depth; _listener_handles
    # (root_uuid -> ListenerHandle) for fits in progress.
    """Callback that displays progress bars for each iterative step of an estimator.

    Parameters
    ----------
    max_propagation_depth : int, default=1
        The maximum depth of nested levels of estimators to display progress bars for.
        0 means that the progress of only the outermost estimator is displayed.
        If set to None, all levels are displayed.

    Notes
    -----
    This callback requires rich to be installed.
    """

    @validate_params(
        {"max_propagation_depth": [Interval(Integral, 0, None, closed="left"), None]},
        prefer_skip_nested_validation=True,
    )
    def __init__(self, max_propagation_depth=1):
        # [FUNCTION DOC]
        # Inputs: max_propagation_depth - int >= 0 or None (validated by validate_params).
        # Exceptions: ImportError (via check_rich_support) when rich is not installed;
        # InvalidParameterError for bad max_propagation_depth.
        check_rich_support("Progressbar")

        self.max_propagation_depth = max_propagation_depth

        # Handles to the main-process per-fit listeners, keyed by `root_uuid`.
        self._listener_handles = {}

    def setup(self, estimator, context):
        # [FUNCTION DOC]
        # Starts per-fit display: creates a Queue, opens a listener whose consumer is
        # queue.put, starts a RichProgressMonitor thread, and registers queue/monitor/handle
        # under context.root_uuid. Side effects: thread, socket, module/instance registries.
        # Lazily create the per-fit transport state. `setup` runs on the main
        # process for the common case where `ProgressBar` is registered on (or
        # auto-propagated from) the outermost estimator. In the degraded case
        # where the outer function does not support callbacks, this may run
        # inside a worker process; that worker will then open its own local
        # listener and the events will be process-local — the callback runs in
        # slightly degraded mode but does not crash.
        queue = Queue()
        # `queue.put` is the message consumer that `send` calls will use to forward
        # information to the rich monitor thread.
        self._listener_handles[context.root_uuid] = open_listener(queue.put)

        progress_monitor = RichProgressMonitor(queue=queue)
        progress_monitor.start()

        _run_queues[context.root_uuid] = queue
        _run_monitors[context.root_uuid] = progress_monitor

    def on_fit_task_begin(self, estimator, context):
        # [FUNCTION DOC]
        # Sends a "begin" event (path of task_ids from root, task name/id, max_subtasks,
        # estimator and source names) when the task is a non-leaf, a root, or a merged
        # sub-estimator root; leaf tasks send nothing. Exceptions: KeyError if setup was not
        # run for this root_uuid.
        # A new progress bar is created at the beginning of each task that is not a
        # leaf, except if it's also the root task of an estimator.
        if (
            context.max_subtasks != 0
            or context.parent is None
            or context.source_estimator_name is not None
        ):
            # We pass the minimal information to the queue that is necessary to create a
            # progress bar and not the context to avoid pickling the whole context tree.
            path = [ctx.task_id for ctx in get_context_path(context)]
            send(
                self._listener_handles[context.root_uuid],
                {
                    "event": "begin",
                    "path": path,
                    "task_name": context.task_name,
                    "task_id": context.task_id,
                    "max_subtasks": context.max_subtasks,
                    "estimator_name": context.estimator_name,
                    "source_estimator_name": context.source_estimator_name,
                    "source_task_name": context.source_task_name,
                },
            )

    def on_fit_task_end(self, estimator, context):
        # [FUNCTION DOC]
        # Sends an "end" event carrying only the task_id path from the root to this context.
        # Outputs: None (never requests stopping).
        # The path is enough to update the progress of the task and its ancestors.
        send(
            self._listener_handles[context.root_uuid],
            {
                "event": "end",
                "path": [ctx.task_id for ctx in get_context_path(context)],
            },
        )

    def teardown(self, estimator, context):
        # [FUNCTION DOC]
        # Ends the per-fit display: puts the None sentinel on the queue, joins the monitor
        # thread, closes and forgets the listener. Exceptions: KeyError if called twice.
        # Fit is finished. Signal that the queue won't receive any more tasks, close
        # the monitor thread and the listener.
        _run_queues.pop(context.root_uuid).put(None)
        _run_monitors.pop(context.root_uuid).join()
        close_listener(self._listener_handles.pop(context.root_uuid))


try:
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
    from rich.style import Style

    class _Progress(Progress):
        # [CLASS DOC]
        # HELPER: rich Progress subclass whose rendering shows tasks in the order given by the
        # _ordered_tasks attribute (empty table until it is set).
        # Custom Progress class to allow showing the tasks in a given order (given by
        # setting the _ordered_tasks attribute). In particular it allows to dynamically
        # create and insert tasks between existing tasks.
        def get_renderables(self):
            # [FUNCTION DOC]
            # HELPER: Yields a single table renderable built from _ordered_tasks.
            yield self.make_tasks_table(getattr(self, "_ordered_tasks", []))

    class _StyledColumnMixin:
        # [CLASS DOC]
        # HELPER: Mixin coloring a column's rendered text blue when the task is finished and
        # orange otherwise.
        """Apply finished/in-progress color style to rendered text."""

        def render(self, task):
            # [FUNCTION DOC]
            # HELPER: Renders via the next class in the MRO and sets the text style; returns text.
            text = super().render(task)
            text.style = "#29ABE2" if task.finished else "#F7931E"
            return text

    class _StyledTimeRemainingColumn(_StyledColumnMixin, TimeRemainingColumn):
        # [CLASS DOC]
        # HELPER: TimeRemainingColumn with finished/in-progress coloring.
        """Time column with color styling."""

    class _StyledPercentageColumn(_StyledColumnMixin, TextColumn):
        # [CLASS DOC]
        # HELPER: TextColumn (used for the percentage) with finished/in-progress coloring.
        """Percentage column with color styling."""

except ImportError:
    pass


class RichProgressMonitor(Thread):
    # [CLASS DOC]
    # Non-daemon thread consuming event dicts from queue and updating a rich display.
    # Attributes created in run: progress_ctx (_Progress) and root_rich_task (RichTask).
    """Thread monitoring the progress of an estimator with rich based display.

    The display is a list of nested rich tasks using `rich.Progress`. There is one for
    each non-leaf node in the task tree of the estimator.

    Parameters
    ----------
    queue : `queue.Queue` instance
        This thread will run until the queue is empty.
    """

    def __init__(self, *, queue):
        # [FUNCTION DOC]
        # Inputs: queue - Queue of event dicts terminated by a falsy sentinel (None).
        super().__init__()
        self.queue = queue

    def run(self):
        # [FUNCTION DOC]
        # Thread body: builds the _Progress with description, bar, percentage and remaining
        # time columns, then within the live display loops on queue.get until a falsy item,
        # dispatching "begin" events to _on_task_begin and all others to _on_task_end, and
        # does a final refresh. Side effects: mutates the event dicts (pops "event"); writes
        # to the terminal.
        self.progress_ctx = _Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                complete_style=Style(color="#F7931E"),
                finished_style=Style(color="#29ABE2"),
                pulse_style=Style(color="#F7931E"),
            ),
            _StyledPercentageColumn("{task.percentage:>3.0f}%"),
            _StyledTimeRemainingColumn(elapsed_when_finished=True),
        )

        # Holds the root of the tree of rich tasks (i.e. progress bars) that will be
        # created dynamically as the task tree of the estimator is traversed.
        self.root_rich_task = None

        with self.progress_ctx:
            while task_info := self.queue.get():
                if task_info.pop("event") == "begin":
                    self._on_task_begin(task_info)
                else:
                    self._on_task_end(task_info)

            self.progress_ctx.refresh()

    def _on_task_begin(self, task_info):
        # [FUNCTION DOC]
        # HELPER: Creates a RichTask at depth len(path)-1; depth 0 becomes root_rich_task,
        # otherwise it is stored in its parent (found by walking the path) under its task_id.
        # Then recomputes progress_ctx._ordered_tasks via pre-order traversal of the tree.
        # KNOWN ISSUE: assumes the parent bar exists; a begin event whose parent was a leaf
        # without a bar would hit None and raise AttributeError.
        """Create a progress bar for the task and update the list of ordered tasks."""
        path = task_info.pop("path")

        rich_task = RichTask(
            progress_ctx=self.progress_ctx, task_info=task_info, depth=len(path) - 1
        )
        if rich_task.depth == 0:
            self.root_rich_task = rich_task
        else:
            parent = self.root_rich_task.get_descendants(path)[-2]
            parent.children[path[-1]] = rich_task

        self.progress_ctx._ordered_tasks = [
            self.progress_ctx.tasks[task.id] for task in self.root_rich_task
        ]

    def _on_task_end(self, task_info):
        # [FUNCTION DOC]
        # HELPER: Handles an "end" event. Finds the task along path; if it has no bar (a
        # leaf), inserts a bar-less RichTask (total 0, counted as complete) under its parent;
        # otherwise marks it complete and shows its bar as 100%. Then for each ancestor with a
        # known total, from deepest to root, sets its completed count to progress * total.
        """Update the progress of the task and its ancestors recursively."""
        path = task_info.pop("path")
        *ancestors, task = self.root_rich_task.get_descendants(path)

        if task is None:
            # a leaf task of the estimator, no progress bar was created for it
            task = RichTask(self.progress_ctx, task_info, depth=len(ancestors))
            ancestors[-1].children[path[-1]] = task
        else:
            # Task is finished. Render the progress bar as 100% completed regardless of
            # its progress because a task may execute less tasks than its max_subtasks.
            task.completed = task.total
            self.progress_ctx.update(task.id, completed=1, total=1)

        for ancestor in reversed(ancestors):
            if ancestor.total is None:
                continue

            completed = ancestor.progress * ancestor.total
            self.progress_ctx.update(ancestor.id, completed=completed)


class RichTask:
    # [CLASS DOC]
    # HELPER: Node of the tree of progress bars mirroring the context tree. A node built
    # from an event without display fields (leaf end event) gets no id/progress bar.
    """A task, i.e. progressbar, in the tree of rich tasks.

    There is a rich task for each non-leaf task in the context tree of the estimator.

    Parameters
    ----------
    progress_ctx : `rich.Progress` instance
        The progress context to which this task belongs.

    task_info : dict
        Information about the task for which this rich task is created.

    depth : int
        The depth of the task in the tree of rich tasks.

    Attributes
    ----------
    completed : int
        The number of completed subtasks.

    total : int or None
        The total number of subtasks. None if the total number of subtasks is not known.

    progress : float
        The fraction, between 0 and 1, of the task that is completed.

    id : int
        The ID of the task in the Progress context.

    children : dict
        A mapping from the index of a child to the child node `{idx: RichTask}`.
        For a leaf, it's an empty dictionary.
    """

    def __init__(self, progress_ctx, task_info, *, depth):
        # [FUNCTION DOC]
        # Inputs: progress_ctx - rich Progress; task_info - event dict (max_subtasks is popped
        # from it, mutating the caller's dict); depth - int (keyword-only).
        # Side effects: if task_info still has keys, adds a rich task with a formatted
        # description and total=max_subtasks, storing its id.
        self.children = {}
        self.depth = depth
        self.completed = 0
        self.total = task_info.pop("max_subtasks", 0)

        if task_info:
            description = self._format_task_description(task_info)
            self.id = progress_ctx.add_task(description, total=self.total)

    @property
    def progress(self):
        # [FUNCTION DOC]
        # Fraction completed in [0, 1]: 1.0 if completed equals total (including 0 == 0 for
        # leaves), 0.0 if total is unknown, else the sum of children progress over total.
        """Return the fraction of the task that is completed."""
        if self.completed == self.total:
            return 1.0
        if self.total is None:
            return 0.0
        return sum(child.progress for child in self.children.values()) / self.total

    def _format_task_description(self, task_info):
        # [FUNCTION DOC]
        # HELPER: Returns the description: two spaces per depth level, optional
        # "source_estimator - source_task | " prefix, "estimator - task", and " #task_id" for
        # non-root bars.
        """Return a formatted description for the task."""
        indent = f"{'  ' * self.depth}"
        task_desc = f"{task_info['estimator_name']} - {task_info['task_name']}"
        id_mark = f" #{task_info['task_id']}" if self.depth > 0 else ""
        source_task_desc = (
            f"{task_info['source_estimator_name']} - {task_info['source_task_name']} | "
            if task_info["source_estimator_name"] is not None
            else ""
        )
        return f"{indent}{source_task_desc}{task_desc}{id_mark}"

    def get_descendants(self, path):
        # [FUNCTION DOC]
        # HELPER: Recursively follows path (task_ids; path[0] is this node) through children.
        # Outputs: list of nodes from self to the target; ends with None when the next child
        # is missing.
        """Return the descendants from this task along the given path."""
        if len(path) == 1:
            return [self]
        if path[1] not in self.children:
            return [self, None]
        return [self] + self.children[path[1]].get_descendants(path[1:])

    def __iter__(self):
        # [FUNCTION DOC]
        # Pre-order DFS yielding only nodes that have a bar (an id); a node without a bar
        # prunes its whole subtree.
        """Pre-order depth-first traversal, only including tasks with a progress bar."""
        if hasattr(self, "id"):
            yield self
            for child in self.children.values():
                yield from child
