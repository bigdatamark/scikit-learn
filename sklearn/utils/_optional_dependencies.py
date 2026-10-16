# [MODULE DOC]
# HELPER: Lazy checks for optional third-party dependencies (matplotlib, pandas,
# rich) that raise an informative ImportError naming the calling feature.
# No import-time side effects; the optional packages are only imported on call.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


def check_matplotlib_support(caller_name):
    # [FUNCTION DOC]
    # HELPER: Tries to import matplotlib.
    # Inputs: caller_name - str naming the feature, used in the error message.
    # Outputs: None.
    # Exceptions: ImportError (chained from the original) with install hint if
    # matplotlib is missing.
    # Side effects: imports matplotlib into sys.modules when available.
    """Raise ImportError with detailed error message if mpl is not installed.

    Plot utilities like any of the Display's plotting functions should lazily import
    matplotlib and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires matplotlib.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{caller_name} requires matplotlib. You can install matplotlib with "
            "`pip install matplotlib`"
        ) from e


def check_pandas_support(caller_name):
    # [FUNCTION DOC]
    # HELPER: Tries to import pandas and returns the module object.
    # Inputs: caller_name - str naming the feature, used in the error message.
    # Outputs: the pandas module.
    # Exceptions: ImportError (chained) if pandas is missing.
    # Side effects: imports pandas.
    # KNOWN ISSUE: docstring says "Plot utilities like fetch_openml"; fetch_openml is a
    # dataset loader, not a plot utility.
    """Raise ImportError with detailed error message if pandas is not installed.

    Plot utilities like :func:`fetch_openml` should lazily import
    pandas and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires pandas.

    Returns
    -------
    pandas
        The pandas package.
    """
    try:
        import pandas

        return pandas
    except ImportError as e:
        raise ImportError(f"{caller_name} requires pandas.") from e


def check_rich_support(caller_name):
    # [FUNCTION DOC]
    # HELPER: Tries to import rich.
    # Inputs: caller_name - str naming the feature, used in the error message.
    # Outputs: None.
    # Exceptions: ImportError (chained) if rich is missing.
    # Side effects: imports rich when available.
    """Raise ImportError with detailed error message if rich is not installed.

    Caller should lazily import rich and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires rich.
    """
    try:
        import rich  # noqa: F401
    except ImportError as e:
        raise ImportError(f"{caller_name} requires rich.") from e
