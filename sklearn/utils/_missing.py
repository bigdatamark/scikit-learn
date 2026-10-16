# [MODULE DOC]
# HELPER: Small utilities to detect missing-value markers (float NaN scalars and
# pandas.NA) without failing on non-numeric inputs. No import-time side effects.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import numbers
from contextlib import suppress


def is_scalar_nan(x):
    # [FUNCTION DOC]
    # Returns True only for real, non-integral scalars that are NaN (Python float,
    # numpy floating). Integers, bools, strings, None and containers return False.
    # Inputs: x - any object.
    # Outputs: bool.
    # Exceptions: none; non-numeric types short-circuit before math.isnan is called.
    # Side effects: none.
    """Test if x is NaN.

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').

    Parameters
    ----------
    x : any type
        Any scalar value.

    Returns
    -------
    bool
        Returns true if x is NaN, and false otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils._missing import is_scalar_nan
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    return (
        not isinstance(x, numbers.Integral)
        and isinstance(x, numbers.Real)
        and math.isnan(x)
    )


def is_pandas_na(x):
    # [FUNCTION DOC]
    # Returns True when x is the pandas.NA singleton (identity check). If pandas
    # cannot be imported, returns False.
    # Inputs: x - any object.
    # Outputs: bool.
    # Exceptions: ImportError from pandas is suppressed.
    # Side effects: may import pandas lazily.
    """Test if x is pandas.NA.

    We intentionally do not use this function to return `True` for `pd.NA` in
    `is_scalar_nan`, because estimators that support `pd.NA` are the exception
    rather than the rule at the moment. When `pd.NA` is more universally
    supported, we may reconsider this decision.

    Parameters
    ----------
    x : any type
        The input value to test.

    Returns
    -------
    boolean
        True if `x` is `pandas.NA`, False otherwise.
    """
    with suppress(ImportError):
        from pandas import NA

        return x is NA

    return False
