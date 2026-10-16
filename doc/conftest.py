# [MODULE DOC]
# pytest configuration for running doctests in the .rst documentation. Skips
# dataset-dependent pages when data is not cached or network tests are disabled,
# skips pages needing pandas, matplotlib or cupy when missing, forces the
# matplotlib agg backend, clears doctest globals, and skips all doctests on
# NumPy < 2 or SciPy < 1.14 because their reprs differ.
import os
from os import environ
from os.path import exists, join

import pytest
from _pytest.doctest import DoctestItem

from sklearn.datasets import get_data_home
from sklearn.datasets._base import _pkl_filepath
from sklearn.datasets._twenty_newsgroups import CACHE_NAME
from sklearn.utils._testing import SkipTest, check_skip_network
from sklearn.utils.fixes import np_base_version, parse_version, sp_version


def setup_labeled_faces():
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest unless the LFW data folder exists in the data home.
    data_home = get_data_home()
    if not exists(join(data_home, "lfw_home")):
        raise SkipTest("Skipping dataset loading doctests")


def setup_rcv1():
    # [FUNCTION DOC]
    # HELPER: Skips when network tests are disabled (check_skip_network) or when the
    # RCV1 folder is not in the data home.
    check_skip_network()
    # skip the test in rcv1.rst if the dataset is not already loaded
    rcv1_dir = join(get_data_home(), "RCV1")
    if not exists(rcv1_dir):
        raise SkipTest("Download RCV1 dataset to run this test.")


def setup_twenty_newsgroups():
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest unless the 20 newsgroups pickle cache exists.
    cache_path = _pkl_filepath(get_data_home(), CACHE_NAME)
    if not exists(cache_path):
        raise SkipTest("Skipping dataset loading doctests")


def setup_working_with_text_data():
    # [FUNCTION DOC]
    # HELPER: Skips when network tests are disabled or the 20 newsgroups cache is
    # missing (for the text data tutorial).
    check_skip_network()
    cache_path = _pkl_filepath(get_data_home(), CACHE_NAME)
    if not exists(cache_path):
        raise SkipTest("Skipping dataset loading doctests")


def setup_loading_other_datasets():
    # [FUNCTION DOC]
    # HELPER: Skips when pandas is missing or SKLEARN_SKIP_NETWORK_TESTS is not "0"
    # (default "1", i.e. skip).
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise SkipTest("Skipping loading_other_datasets.rst, pandas not installed")

    # checks SKLEARN_SKIP_NETWORK_TESTS to see if test should run
    run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
    if not run_network_tests:
        raise SkipTest(
            "Skipping loading_other_datasets.rst, tests can be "
            "enabled by setting SKLEARN_SKIP_NETWORK_TESTS=0"
        )


def setup_compose():
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest if pandas cannot be imported (compose.rst).
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise SkipTest("Skipping compose.rst, pandas not installed")


def setup_impute():
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest if pandas cannot be imported (impute.rst).
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise SkipTest("Skipping impute.rst, pandas not installed")


def setup_grid_search():
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest if pandas cannot be imported (grid_search.rst).
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise SkipTest("Skipping grid_search.rst, pandas not installed")


def setup_preprocessing():
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest if pandas cannot be imported (preprocessing.rst).
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise SkipTest("Skipping preprocessing.rst, pandas not installed")


def skip_if_matplotlib_not_installed(fname):
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest naming the file's basename if matplotlib is missing.
    # Input: fname path string.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        basename = os.path.basename(fname)
        raise SkipTest(f"Skipping doctests for {basename}, matplotlib not installed")


def skip_if_cupy_not_installed(fname):
    # [FUNCTION DOC]
    # HELPER: Raises SkipTest naming the file's basename if cupy is missing.
    # Input: fname path string.
    try:
        import cupy  # noqa: F401
    except ImportError:
        basename = os.path.basename(fname)
        raise SkipTest(f"Skipping doctests for {basename}, cupy not installed")


def pytest_runtest_setup(item):
    # [FUNCTION DOC]
    # pytest hook run before each item: normalizes the item path to forward slashes
    # and calls the matching setup_* helper based on the file suffix, then applies the
    # matplotlib skip for partial_dependence.rst/tree.rst and the cupy skip for
    # array_api.rst. Exceptions: SkipTest from helpers.
    # KNOWN ISSUE: datasets/index.rst always enters the first branch, so the
    # "or is_index" conditions of the later elif branches are unreachable and index.rst
    # only gets the labeled faces check.
    fname = item.fspath.strpath
    # normalize filename to use forward slashes on Windows for easier handling
    # later
    fname = fname.replace(os.sep, "/")

    is_index = fname.endswith("datasets/index.rst")
    if fname.endswith("datasets/labeled_faces.rst") or is_index:
        setup_labeled_faces()
    elif fname.endswith("datasets/rcv1.rst") or is_index:
        setup_rcv1()
    elif fname.endswith("datasets/twenty_newsgroups.rst") or is_index:
        setup_twenty_newsgroups()
    elif fname.endswith("modules/compose.rst") or is_index:
        setup_compose()
    elif fname.endswith("datasets/loading_other_datasets.rst"):
        setup_loading_other_datasets()
    elif fname.endswith("modules/impute.rst"):
        setup_impute()
    elif fname.endswith("modules/grid_search.rst"):
        setup_grid_search()
    elif fname.endswith("modules/preprocessing.rst"):
        setup_preprocessing()

    rst_files_requiring_matplotlib = [
        "modules/partial_dependence.rst",
        "modules/tree.rst",
    ]
    for each in rst_files_requiring_matplotlib:
        if fname.endswith(each):
            skip_if_matplotlib_not_installed(fname)

    if fname.endswith("array_api.rst"):
        skip_if_cupy_not_installed(fname)


def pytest_configure(config):
    # [FUNCTION DOC]
    # pytest hook selecting the matplotlib "agg" backend if matplotlib is installed.
    # Side effects: global matplotlib backend change.
    # Use matplotlib agg backend during the tests including doctests
    try:
        import matplotlib

        matplotlib.use("agg")
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    # [FUNCTION DOC]
    # pytest hook after collection: sets an empty globals dict for each doctest item
    # and, when NumPy < 2 or SciPy < 1.14, adds a skip marker to all doctest items
    # (the SciPy reason wins if both apply). Side effects: mutates collected items.
    """Called after collect is completed.

    Parameters
    ----------
    config : pytest config
    items : list of collected items
    """
    skip_doctests = False
    if np_base_version < parse_version("2"):
        # TODO: configure numpy to output scalar arrays as regular Python scalars
        # once possible to improve readability of the tests docstrings.
        # https://numpy.org/neps/nep-0051-scalar-representation.html#implementation
        reason = "Due to NEP 51 numpy scalar repr has changed in numpy 2"
        skip_doctests = True

    if sp_version < parse_version("1.14"):
        reason = "Scipy sparse matrix repr has changed in scipy 1.14"
        skip_doctests = True

    # Normally doctest has the entire module's scope. Here we set globs to an empty dict
    # to remove the module's scope:
    # https://docs.python.org/3/library/doctest.html#what-s-the-execution-context
    for item in items:
        if isinstance(item, DoctestItem):
            item.dtest.globs = {}

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)
