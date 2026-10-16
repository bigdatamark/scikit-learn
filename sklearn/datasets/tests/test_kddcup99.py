# [MODULE DOC]
# Tests for sklearn.datasets.fetch_kddcup99 (percent10 mode only). The loader comes from
# the fetch_kddcup99_fxt fixture, which skips unless data is cached or downloads are
# enabled. Covers subsets, return_X_y, as_frame, shuffling, missing pandas and corrupted
# cache handling.
"""Test  kddcup99 loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs).

Only 'percent10' mode is tested, as the full data
is too big to use in unit-testing.
"""

from functools import partial

import pytest

from sklearn.datasets.tests.test_common import (
    check_as_frame,
    check_pandas_dependency_message,
    check_return_X_y,
)


@pytest.mark.parametrize("as_frame", [True, False])
@pytest.mark.parametrize(
    "subset, n_samples, n_features",
    [
        (None, 494021, 41),
        ("SA", 100655, 41),
        ("SF", 73237, 4),
        ("http", 58725, 3),
        ("smtp", 9571, 3),
    ],
)
def test_fetch_kddcup99_percent10(
    fetch_kddcup99_fxt, as_frame, subset, n_samples, n_features
):
    # [FUNCTION DOC]
    # Parametrized over subset (None, SA, SF, http, smtp) with expected sample/feature
    # counts and over as_frame. Verifies data and target shapes, the frame having one
    # extra column (the target) when as_frame is True, and the DESCR RST label.
    data = fetch_kddcup99_fxt(subset=subset, as_frame=as_frame)
    assert data.data.shape == (n_samples, n_features)
    assert data.target.shape == (n_samples,)
    if as_frame:
        assert data.frame.shape == (n_samples, n_features + 1)
    assert data.DESCR.startswith(".. _kddcup99_dataset:")


def test_fetch_kddcup99_return_X_y(fetch_kddcup99_fxt):
    # [FUNCTION DOC]
    # Verifies via check_return_X_y that return_X_y=True on the smtp subset returns the
    # same data and target as the Bunch.
    fetch_func = partial(fetch_kddcup99_fxt, subset="smtp")
    data = fetch_func()
    check_return_X_y(data, fetch_func)


def test_fetch_kddcup99_as_frame(fetch_kddcup99_fxt):
    # [FUNCTION DOC]
    # Runs the shared check_as_frame helper on the default fetch to verify
    # DataFrame/Series outputs are consistent with the array outputs.
    bunch = fetch_kddcup99_fxt()
    check_as_frame(bunch, fetch_kddcup99_fxt)


def test_fetch_kddcup99_shuffle(fetch_kddcup99_fxt):
    # [FUNCTION DOC]
    # Verifies that shuffle=True on the SA subset (random_state=0) keeps the same set of
    # target labels and the same data and target shapes as the unshuffled fetch.
    dataset = fetch_kddcup99_fxt(
        random_state=0,
        subset="SA",
        percent10=True,
    )
    dataset_shuffled = fetch_kddcup99_fxt(
        random_state=0,
        subset="SA",
        shuffle=True,
        percent10=True,
    )
    assert set(dataset["target"]) == set(dataset_shuffled["target"])
    assert dataset_shuffled.data.shape == dataset.data.shape
    assert dataset_shuffled.target.shape == dataset.target.shape


def test_pandas_dependency_message(fetch_kddcup99_fxt, hide_available_pandas):
    # [FUNCTION DOC]
    # Verifies via check_pandas_dependency_message that as_frame=True raises the
    # expected ImportError when pandas is hidden.
    check_pandas_dependency_message(fetch_kddcup99_fxt)


def test_corrupted_file_error_message(fetch_kddcup99_fxt, tmp_path):
    # [FUNCTION DOC]
    # Writes junk bytes into a fake cached samples file under tmp_path/kddcup99_10-py3
    # and verifies the fetcher raises OSError telling the user to delete the cache
    # directory.
    # Side effects: writes files in tmp_path only.
    """Check that a nice error message is raised when cache is corrupted."""
    kddcup99_dir = tmp_path / "kddcup99_10-py3"
    kddcup99_dir.mkdir()
    samples_path = kddcup99_dir / "samples"

    with samples_path.open("wb") as f:
        f.write(b"THIS IS CORRUPTED")

    msg = (
        "The cache for fetch_kddcup99 is invalid, please "
        f"delete {kddcup99_dir} and run the fetch_kddcup99 again"
    )

    with pytest.raises(OSError, match=msg):
        fetch_kddcup99_fxt(data_home=str(tmp_path))
