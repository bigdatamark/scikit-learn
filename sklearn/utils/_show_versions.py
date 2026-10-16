# [MODULE DOC]
# Implements sklearn.show_versions: prints platform, Python, dependency versions,
# OpenMP build status and threadpoolctl runtime information for bug reports.
"""
Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import platform
import sys

from threadpoolctl import threadpool_info

from sklearn import __version__
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled


def _get_sys_info():
    # [FUNCTION DOC]
    # HELPER: Returns a dict with keys python (sys.version with newlines replaced by
    # spaces), executable (sys.executable) and machine (platform.platform()).
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    # [FUNCTION DOC]
    # HELPER: Returns a dict with 'sklearn' mapped to the scikit-learn version and each
    # of a fixed list of dependencies (pip, setuptools, numpy, scipy, Cython, pandas,
    # matplotlib, joblib, threadpoolctl, narwhals) mapped to its installed distribution
    # version from importlib.metadata, or None when not installed. Does not import them.
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    deps = [
        "pip",
        "setuptools",
        "numpy",
        "scipy",
        "Cython",
        "pandas",
        "matplotlib",
        "joblib",
        "threadpoolctl",
        "narwhals",
    ]

    deps_info = {
        "sklearn": __version__,
    }

    from importlib.metadata import PackageNotFoundError, version

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions():
    # [FUNCTION DOC]
    # Prints, to stdout, a 'System:' section, a 'Python dependencies:' section
    # (right-aligned keys), whether scikit-learn was built with OpenMP, and for each
    # threadpoolctl-detected library its info key/value pairs separated by blank lines.
    # Outputs: None. Side effects: stdout output; threadpool_info inspects loaded
    # native libraries.
    """Print useful debugging information.

    .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn import show_versions
    >>> show_versions()  # doctest: +SKIP
    """

    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))

    print(
        "\n{k}: {stat}".format(
            k="Built with OpenMP", stat=_openmp_parallelism_enabled()
        )
    )

    # show threadpoolctl results
    threadpool_results = threadpool_info()
    if threadpool_results:
        print()
        print("threadpoolctl info:")

        for i, result in enumerate(threadpool_results):
            for key, val in result.items():
                print(f"{key:>15}: {val}")
            if i != len(threadpool_results) - 1:
                print()
