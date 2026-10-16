# [MODULE DOC]
# Tests for the global configuration API: get_config, set_config and
# config_context (defaults, nesting, exception safety, thread/process safety,
# and the SciPy version requirement of array_api_dispatch).

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils.fixes import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed


def test_config_context():
    # [FUNCTION DOC]
    # Verifies the full default config dict, that calling config_context without
    # "with" changes nothing, that entering it overrides values only inside the block,
    # that None means "keep current value" in nested contexts, that set_config inside
    # a context is reverted on exit, and that positional or unknown keyword arguments
    # raise TypeError.
    # KNOWN ISSUE: the expected default dict is hard-coded, so this test must be
    # updated whenever a new config key is added.
    assert get_config() == {
        "assume_finite": False,
        "working_memory": 1024,
        "print_changed_only": True,
        "display": "diagram",
        "array_api_dispatch": False,
        "pairwise_dist_chunk_size": 256,
        "enable_cython_pairwise_dist": True,
        "transform_output": "default",
        "enable_metadata_routing": False,
        "skip_parameter_validation": False,
        "sparse_interface": "spmatrix",
    }

    # Not using as a context manager affects nothing
    config_context(assume_finite=True)
    assert get_config()["assume_finite"] is False

    with config_context(assume_finite=True):
        assert get_config() == {
            "assume_finite": True,
            "working_memory": 1024,
            "print_changed_only": True,
            "display": "diagram",
            "array_api_dispatch": False,
            "pairwise_dist_chunk_size": 256,
            "enable_cython_pairwise_dist": True,
            "transform_output": "default",
            "enable_metadata_routing": False,
            "skip_parameter_validation": False,
            "sparse_interface": "spmatrix",
        }
    assert get_config()["assume_finite"] is False

    with config_context(assume_finite=True):
        with config_context(assume_finite=None):
            assert get_config()["assume_finite"] is True

        assert get_config()["assume_finite"] is True

        with config_context(assume_finite=False):
            assert get_config()["assume_finite"] is False

            with config_context(assume_finite=None):
                assert get_config()["assume_finite"] is False

                # global setting will not be retained outside of context that
                # did not modify this setting
                set_config(assume_finite=True)
                assert get_config()["assume_finite"] is True

            assert get_config()["assume_finite"] is False

        assert get_config()["assume_finite"] is True

    assert get_config() == {
        "assume_finite": False,
        "working_memory": 1024,
        "print_changed_only": True,
        "display": "diagram",
        "array_api_dispatch": False,
        "pairwise_dist_chunk_size": 256,
        "enable_cython_pairwise_dist": True,
        "transform_output": "default",
        "enable_metadata_routing": False,
        "skip_parameter_validation": False,
        "sparse_interface": "spmatrix",
    }

    # No positional arguments
    with pytest.raises(TypeError):
        config_context(True)

    # No unknown arguments
    with pytest.raises(TypeError):
        config_context(do_something_else=True).__enter__()


def test_config_context_exception():
    # [FUNCTION DOC]
    # Verifies that the config is restored to assume_finite False when an exception
    # (ValueError) escapes a config_context block.
    assert get_config()["assume_finite"] is False
    try:
        with config_context(assume_finite=True):
            assert get_config()["assume_finite"] is True
            raise ValueError()
    except ValueError:
        pass
    assert get_config()["assume_finite"] is False


def test_set_config():
    # [FUNCTION DOC]
    # Verifies set_config toggles assume_finite, that None leaves the current value
    # unchanged, and that an unknown keyword raises TypeError.
    # Side effects: mutates global config but ends with assume_finite False again.
    assert get_config()["assume_finite"] is False
    set_config(assume_finite=None)
    assert get_config()["assume_finite"] is False
    set_config(assume_finite=True)
    assert get_config()["assume_finite"] is True
    set_config(assume_finite=None)
    assert get_config()["assume_finite"] is True
    set_config(assume_finite=False)
    assert get_config()["assume_finite"] is False

    # No unknown arguments
    with pytest.raises(TypeError):
        set_config(do_something_else=True)


def set_assume_finite(assume_finite, sleep_duration):
    # [FUNCTION DOC]
    # HELPER: Worker used by the thread-safety tests. Enters config_context with the
    # given assume_finite, sleeps sleep_duration seconds, then reads the config.
    # Inputs: assume_finite - bool; sleep_duration - float seconds.
    # Outputs: the assume_finite value observed inside the context (bool).
    # Side effects: blocks the calling thread for sleep_duration.
    """Return the value of assume_finite after waiting `sleep_duration`."""
    with config_context(assume_finite=assume_finite):
        time.sleep(sleep_duration)
        return get_config()["assume_finite"]


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_config_threadsafe_joblib(backend):
    # [FUNCTION DOC]
    # Runs set_assume_finite four times through joblib Parallel (n_jobs=2) with the
    # loky, multiprocessing and threading backends, interleaving short and long sleeps,
    # and verifies each job sees its own value, i.e. config is not shared across
    # concurrently running workers.
    """Test that the global config is threadsafe with all joblib backends.
    Two jobs are spawned and sets assume_finite to two different values.
    When the job with a duration 0.1s completes, the assume_finite value
    should be the same as the value passed to the function. In other words,
    it is not influenced by the other job setting assume_finite to True.
    """
    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    items = Parallel(backend=backend, n_jobs=2)(
        delayed(set_assume_finite)(assume_finite, sleep_dur)
        for assume_finite, sleep_dur in zip(assume_finites, sleep_durations)
    )

    assert items == [False, True, False, True]


@pytest.mark.xfail(_IS_WASM, reason="cannot start threads")
def test_config_threadsafe():
    # [FUNCTION DOC]
    # Same check as test_config_threadsafe_joblib but using a ThreadPoolExecutor with
    # two workers directly; xfailed on WASM where threads cannot be started.
    """Uses threads directly to test that the global config does not change
    between threads. Same test as `test_config_threadsafe_joblib` but with
    `ThreadPoolExecutor`."""

    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    with ThreadPoolExecutor(max_workers=2) as e:
        items = [
            output
            for output in e.map(set_assume_finite, assume_finites, sleep_durations)
        ]

    assert items == [False, True, False, True]


def test_config_array_api_dispatch_error_scipy(monkeypatch):
    # [FUNCTION DOC]
    # Monkeypatches the SciPy version seen by sklearn.utils._array_api to 1.13.0 and
    # verifies that enabling array_api_dispatch via config_context or set_config
    # raises ImportError mentioning that SciPy 1.14.0 or newer is required.
    """Check error when SciPy is too old"""
    monkeypatch.setattr(sklearn.utils._array_api.scipy, "__version__", "1.13.0")

    with pytest.raises(ImportError, match="SciPy must be 1.14.0 or newer"):
        with config_context(array_api_dispatch=True):
            pass

    with pytest.raises(ImportError, match="SciPy must be 1.14.0 or newer"):
        set_config(array_api_dispatch=True)
