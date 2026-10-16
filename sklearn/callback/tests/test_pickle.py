# [MODULE DOC]
# Pickling tests for the built-in callbacks (ProgressBar, ScoringMonitor) attached to
# estimators: round-trips before and after fit, refitting after unpickling in the
# same and in a fresh interpreter, and a guard ensuring the pickler never traverses
# containers mutated by listener threads. CALLBACK_FACTORIES parametrizes the tests.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Common pickle round-trip tests for callbacks.

These tests guard the contract that callbacks (and estimators they are attached to)
must be picklable, and that an estimator pickled after a successful fit can be
unpickled in a fresh Python interpreter.
"""

import functools
import io
import pickle
import re
import subprocess
import sys
import textwrap

import pytest

from sklearn.callback import ProgressBar, ScoringMonitor
from sklearn.callback._transport import _message_consumers
from sklearn.callback.tests._utils import MaxIterEstimator
from sklearn.datasets import make_regression


def _pb():
    # [FUNCTION DOC]
    # HELPER: Factory returning a new ProgressBar; skips the test if rich is missing.
    # Outputs: ProgressBar instance.
    pytest.importorskip("rich")
    return ProgressBar()


def _sm():
    # [FUNCTION DOC]
    # HELPER: Factory returning a new ScoringMonitor using the "r2" scorer.
    # Outputs: ScoringMonitor instance.
    return ScoringMonitor(scoring="r2")


CALLBACK_FACTORIES = [
    pytest.param(_pb, id="ProgressBar"),
    pytest.param(_sm, id="ScoringMonitor"),
]


@pytest.mark.parametrize("factory", CALLBACK_FACTORIES)
def test_estimator_with_callback_pickle_roundtrip_pre_fit(factory):
    # [FUNCTION DOC]
    # Verifies an unfitted estimator with one callback survives pickle dumps/loads with
    # its type and single callback preserved.
    # Inputs: factory - parametrized callback factory.
    """An estimator with the callback registered but not yet fitted is picklable."""
    estimator = MaxIterEstimator().set_callbacks(factory())
    restored = pickle.loads(pickle.dumps(estimator))
    assert type(restored) is type(estimator)
    assert len(restored._skl_callbacks) == 1


@pytest.mark.parametrize("factory", CALLBACK_FACTORIES)
def test_estimator_with_callback_pickle_roundtrip_post_fit(factory):
    # [FUNCTION DOC]
    # Same as the pre-fit round-trip but after fitting with max_iter=3.
    # Inputs: factory - parametrized callback factory.
    """An estimator with the callback registered and fitted is picklable."""
    callback = factory()
    estimator = MaxIterEstimator(max_iter=3).set_callbacks(callback)
    estimator.fit()
    restored = pickle.loads(pickle.dumps(estimator))
    assert type(restored) is type(estimator)
    assert len(restored._skl_callbacks) == 1


def test_callbacks_refit_after_pickle_in_same_process(capsys):
    # [FUNCTION DOC]
    # Verifies that an estimator with ProgressBar and ScoringMonitor, fitted, pickled and
    # unpickled in-process, can be refitted: the progress bar prints again (checked on
    # captured stdout) and the restored monitor holds 2 log entries whose first equals
    # the original one.
    # Inputs: capsys - pytest fixture capturing output.
    """An estimator with callbacks survives an in-process pickle round-trip.

    It also supports re-fitting after being unpickled and the callbacks accumulate new
    data from the re-fit.
    """
    pytest.importorskip("rich")

    X, y = make_regression(n_samples=30, n_features=2, random_state=0)

    sm = ScoringMonitor(scoring="r2")
    estimator = MaxIterEstimator(max_iter=3).set_callbacks(ProgressBar(), sm)
    estimator.fit(X=X, y=y)

    captured = capsys.readouterr()
    assert re.search(r"MaxIterEstimator - fit", captured.out)
    assert re.search(r"100%", captured.out)

    original_logs = sm.get_logs(select="all")
    assert len(original_logs) == 1

    restored = pickle.loads(pickle.dumps(estimator))
    restored.fit(X=X, y=y)

    captured = capsys.readouterr()
    assert re.search(r"MaxIterEstimator - fit", captured.out)
    assert re.search(r"100%", captured.out)

    restored_logs = restored._skl_callbacks[1].get_logs(select="all")
    assert len(restored_logs) == 2
    assert restored_logs[0].data == original_logs[0].data


def test_callbacks_refit_after_load_in_fresh_process(tmp_path, capsys):
    # [FUNCTION DOC]
    # Verifies that an estimator pickled to tmp_path can be loaded and refitted in a new
    # Python subprocess; the subprocess script asserts the monitor logs, and the parent
    # checks the subprocess stdout for the progress bar output.
    # Inputs: tmp_path, capsys - pytest fixtures.
    # Side effects: writes est.pkl in tmp_path; spawns a subprocess (timeout 120 s).
    # KNOWN ISSUE: the subprocess return code and stderr are not checked, so a failing
    # assertion inside the script only surfaces if it happens before the progress bar
    # output is printed; the log assertions after refit can fail silently.
    """An estimator with callbacks survives unpickling in a fresh interpreter.

    It also supports re-fitting after being unpickled and the callbacks accumulate new
    data from the re-fit.
    """
    pytest.importorskip("rich")

    X, y = make_regression(n_samples=20, n_features=3, random_state=0)

    sm = ScoringMonitor(scoring="r2")
    estimator = MaxIterEstimator(max_iter=3).set_callbacks(ProgressBar(), sm)
    estimator.fit(X=X, y=y)

    captured = capsys.readouterr()
    assert re.search(r"MaxIterEstimator - fit", captured.out)
    assert re.search(r"100%", captured.out)

    original_logs = sm.get_logs(select="all")
    assert len(original_logs) == 1

    pkl_path = tmp_path / "est.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(estimator, f)

    load_script = textwrap.dedent(
        f"""
        import pickle
        from sklearn.callback import ScoringMonitor
        from sklearn.datasets import make_regression

        with open({str(pkl_path)!r}, "rb") as f:
            est = pickle.load(f)

        X, y = make_regression(n_samples=20, n_features=3, random_state=1)
        est.fit(X=X, y=y)

        restored_logs = est._skl_callbacks[1].get_logs(select="all")
        assert len(restored_logs) == 2
        assert restored_logs[0].data == {original_logs[0].data}
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", load_script], capture_output=True, timeout=120
    )

    stdout = result.stdout.decode()
    assert re.search(r"MaxIterEstimator - fit", stdout)
    assert re.search(r"100%", stdout)


class _TraversalRecorder(pickle.Pickler):
    # [CLASS DOC]
    # HELPER: Pickler writing to an in-memory buffer that records the id of every object
    # it visits via persistent_id.
    # Attributes: walked_through - set of object ids encountered.
    """A pickler that records the id of every object it walks through.

    `persistent_id` is called for every object the pickler encounters, so an object is
    recorded whichever path leads to it, not only when it is a direct attribute of the
    object being pickled,
    see https://docs.python.org/3/library/pickle.html#pickle.Pickler.persistent_id.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the pickler on a fresh BytesIO with the highest protocol and an empty
        # walked_through set.
        super().__init__(io.BytesIO(), protocol=pickle.HIGHEST_PROTOCOL)
        self.walked_through = set()

    def persistent_id(self, obj):
        # [FUNCTION DOC]
        # Records id(obj) in walked_through and returns None so obj is pickled normally.
        # Inputs: obj - object being pickled. Outputs: None.
        self.walked_through.add(id(obj))
        return None  # pickle `obj` as usual


def _checked(hook):
    # [FUNCTION DOC]
    # HELPER: Wraps a callback hook method so that, before delegating, it verifies the
    # callback can be pickled without walking through containers that listener threads
    # mutate.
    # Inputs: hook - unbound hook function.
    # Outputs: wrapper with the same signature metadata (functools.wraps).
    """Wrap a callback hook so that it first checks the callback it is called on.

    The check is that the pickler does not walk through the objects that listener
    threads mutate. They are found through the registered consumers: a consumer is
    normally a method bound to the container it fills, e.g. `self._log.append` for
    ScoringMonitor or `queue.put` for ProgressBar, so the container is what it is bound
    to. A consumer bound to nothing, e.g. a closure, is skipped, since there is then no
    way to tell what it mutates.
    """

    # preserve the signature of the hook because callbacks are validated against it
    @functools.wraps(hook)
    def checked_hook(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Snapshots the registered message consumers, collects the objects they are bound to
        # (skipping unbound consumers), pickles self with _TraversalRecorder, asserts none of
        # those containers were visited, then calls the original hook.
        # Inputs: self - callback; args/kwargs forwarded. Outputs: the hook's return value.
        # Exceptions: AssertionError if an offending container was traversed.
        # snapshot because another thread may register a listener concurrently
        consumers = list(_message_consumers.values())
        # the containers are held, not just their ids, which could be reused once freed
        watched = [c.__self__ for c in consumers if hasattr(c, "__self__")]

        recorder = _TraversalRecorder()
        recorder.dump(self)

        offenders = recorder.walked_through & {id(container) for container in watched}
        assert not offenders, (
            f"Pickling {self.__class__.__name__} walks through a container that a"
            " listener thread mutates concurrently. Keep it away from the pickler,"
            " either by handing over a copy of it in the callback's `__getstate__`, or"
            " by storing it outside of the callback instance."
        )

        return hook(self, *args, **kwargs)

    return checked_hook


@pytest.mark.parametrize("factory", CALLBACK_FACTORIES)
def test_listener_state_is_not_walked_by_the_pickler(factory, monkeypatch):
    # [FUNCTION DOC]
    # Monkeypatches the callback class's on_fit_task_begin/on_fit_task_end with _checked
    # wrappers and fits an estimator, so the pickling check runs while listeners are up.
    # Inputs: factory - parametrized; monkeypatch - fixture.
    # Side effects: patches the callback class for the duration of the test.
    """Check that pickling a callback never traverses state its listener mutates.

    An estimator carrying a callback can be pickled by a background thread, e.g. loky's
    queue feeder dispatching a task to a worker, while the listener thread of that same
    callback mutates the callback's state as messages come in. Pickling a container that
    another thread mutates breaks the dump, which joblib reports as "Could not pickle
    the task to send it to the workers".

    The check runs from a hook, i.e. while the listeners are up, which is when such a
    dispatch would happen.
    """
    callback = factory()

    for hook_name in ("on_fit_task_begin", "on_fit_task_end"):
        hook = getattr(callback.__class__, hook_name)
        monkeypatch.setattr(callback.__class__, hook_name, _checked(hook))

    MaxIterEstimator(max_iter=3).set_callbacks(callback).fit()
