# [MODULE DOC]
# Implements the set_output API: transformers' transform/fit_transform outputs are
# wrapped into a dataframe container (pandas or polars) according to per-estimator
# or global configuration. Defines the container adapter protocol, the pandas and
# polars adapters, a registry (ADAPTERS_MANAGER, populated with both adapters at
# import time; the libraries themselves are imported lazily), config lookup,
# output wrapping helpers and the _SetOutputMixin used by TransformerMixin.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from functools import wraps
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import issparse

from sklearn._config import get_config
from sklearn.utils._available_if import available_if
from sklearn.utils.fixes import parse_version


def check_library_installed(library):
    # [FUNCTION DOC]
    # HELPER: Imports and returns the named module.
    # Inputs: library - str module name. Outputs: module.
    # Exceptions: ImportError with a set_output-specific message when not installed.
    """Check library is installed."""
    try:
        return importlib.import_module(library)
    except ImportError as exc:
        raise ImportError(
            f"Setting output container to '{library}' requires {library} to be"
            " installed"
        ) from exc


def get_columns(columns):
    # [FUNCTION DOC]
    # HELPER: Resolves column names. If columns is callable (typically
    # get_feature_names_out) it is called; an AttributeError whose message mentions
    # 'does not provide get_feature_names_out' yields None, other AttributeErrors are
    # re-raised. Non-callables are returned unchanged.
    # Outputs: array-like of names or None.
    if callable(columns):
        try:
            return columns()
        except AttributeError as e:
            if "does not provide get_feature_names_out" in str(e):
                return None
            else:
                raise
    return columns


@runtime_checkable
class ContainerAdapterProtocol(Protocol):
    # [CLASS DOC]
    # Runtime-checkable typing Protocol describing what an output container adapter
    # must offer: a container_lib name and create_container, is_supported_container,
    # rename_columns and hstack methods. Bodies are docstring-only stubs.
    container_lib: str

    def create_container(self, X_output, X_original, columns, inplace=False):
        # [FUNCTION DOC]
        # Protocol stub; see docstring. Returns None if called directly.
        # KNOWN ISSUE: the protocol declares inplace default False while both concrete
        # adapters default to True.
        """Create container from `X_output` with additional metadata.

        Parameters
        ----------
        X_output : {ndarray, dataframe}
            Data to wrap.

        X_original : {ndarray, dataframe}
            Original input dataframe. This is used to extract the metadata that should
            be passed to `X_output`, e.g. pandas row index.

        columns : callable, ndarray, or None
            The column names or a callable that returns the column names. The
            callable is useful if the column names require some computation. If `None`,
            then no columns are passed to the container's constructor.

        inplace : bool, default=False
            Whether or not we intend to modify `X_output` in-place. However, it does
            not guarantee that we return the same object if the in-place operation
            is not possible.

        Returns
        -------
        wrapped_output : container_type
            `X_output` wrapped into the container type.
        """

    def is_supported_container(self, X):
        # [FUNCTION DOC]
        # Protocol stub; see docstring.
        """Return True if X is a supported container.

        Parameters
        ----------
        X : container
            Container to be checked.

        Returns
        -------
        is_supported_container : bool
            True if X is a supported container.
        """

    def rename_columns(self, X, columns):
        # [FUNCTION DOC]
        # Protocol stub; see docstring.
        """Rename columns in `X`.

        Parameters
        ----------
        X : container
            Container which columns is updated.

        columns : ndarray of str
            Columns to update the `X`'s columns with.

        Returns
        -------
        updated_container : container
            Container with new names.
        """

    def hstack(self, Xs):
        # [FUNCTION DOC]
        # Protocol stub; see docstring.
        """Stack containers horizontally (column-wise).

        Parameters
        ----------
        Xs : list of containers
            List of containers to stack.

        Returns
        -------
        stacked_Xs : container
            Stacked containers.
        """


class PandasAdapter:
    # [CLASS DOC]
    # Adapter producing pandas DataFrames (container_lib 'pandas').
    container_lib = "pandas"

    def create_container(self, X_output, X_original, columns, inplace=True):
        # [FUNCTION DOC]
        # Resolves columns via get_columns. If not inplace or the output is not already
        # a
        # DataFrame, builds a new DataFrame (copying when not inplace) reusing the index
        # of
        # X_output if it is a DataFrame, else of X_original if it is a DataFrame/Series,
        # else a default index. Then, if columns is not None, overwrites the column
        # labels.
        # Inputs: X_output - ndarray or DataFrame; X_original - original input; columns
        # -
        # callable, array or None; inplace - bool, default True.
        # Outputs: DataFrame. Exceptions: ImportError without pandas; pandas length
        # mismatch errors if columns has the wrong size.
        # Side effects: when inplace and X_output is a DataFrame, its columns are
        # mutated.
        pd = check_library_installed("pandas")
        columns = get_columns(columns)

        if not inplace or not isinstance(X_output, pd.DataFrame):
            # In all these cases, we need to create a new DataFrame

            # Unfortunately, we cannot use `getattr(container, "index")`
            # because `list` exposes an `index` attribute.
            if isinstance(X_output, pd.DataFrame):
                index = X_output.index
            elif isinstance(X_original, (pd.DataFrame, pd.Series)):
                index = X_original.index
            else:
                index = None

            # We don't pass columns here because it would intend columns selection
            # instead of renaming.
            X_output = pd.DataFrame(X_output, index=index, copy=not inplace)

        if columns is not None:
            return self.rename_columns(X_output, columns)
        return X_output

    def is_supported_container(self, X):
        # [FUNCTION DOC]
        # Returns True iff X is a pandas DataFrame. Exceptions: ImportError without
        # pandas.
        pd = check_library_installed("pandas")
        return isinstance(X, pd.DataFrame)

    def rename_columns(self, X, columns):
        # [FUNCTION DOC]
        # Assigns columns directly to X.columns (allows duplicates) and returns X.
        # Side effects: mutates X.
        # we cannot use `rename` since it takes a dictionary and at this stage we have
        # potentially duplicate column names in `X`
        X.columns = columns
        return X

    def hstack(self, Xs):
        # [FUNCTION DOC]
        # Concatenates the DataFrames column-wise with pandas concat. Outputs:
        # DataFrame.
        pd = check_library_installed("pandas")
        return pd.concat(Xs, axis=1)


class PolarsAdapter:
    # [CLASS DOC]
    # Adapter producing polars DataFrames (container_lib 'polars').
    container_lib = "polars"

    def create_container(self, X_output, X_original, columns, inplace=True):
        # [FUNCTION DOC]
        # Resolves columns (ndarray converted to list). If not inplace or X_output is
        # not
        # a polars DataFrame, returns a new DataFrame built row-oriented with columns as
        # schema (None lets polars name them). Otherwise renames columns when given.
        # X_original is ignored (polars has no index).
        # Outputs: polars DataFrame. Exceptions: ImportError without polars.
        # Side effects: mutates X_output's column names in the inplace path.
        pl = check_library_installed("polars")
        columns = get_columns(columns)
        columns = columns.tolist() if isinstance(columns, np.ndarray) else columns

        if not inplace or not isinstance(X_output, pl.DataFrame):
            # In all these cases, we need to create a new DataFrame
            return pl.DataFrame(X_output, schema=columns, orient="row")

        if columns is not None:
            return self.rename_columns(X_output, columns)
        return X_output

    def is_supported_container(self, X):
        # [FUNCTION DOC]
        # Returns True iff X is a polars DataFrame.
        pl = check_library_installed("polars")
        return isinstance(X, pl.DataFrame)

    def rename_columns(self, X, columns):
        # [FUNCTION DOC]
        # Assigns columns to X.columns and returns X. Side effects: mutates X.
        # we cannot use `rename` since it takes a dictionary and at this stage we have
        # potentially duplicate column names in `X`
        X.columns = columns
        return X

    def hstack(self, Xs):
        # [FUNCTION DOC]
        # Concatenates horizontally with polars concat, using how='horizontal_extend'
        # for
        # polars >= 1.42.1 and 'horizontal' otherwise. Outputs: polars DataFrame.
        pl = check_library_installed("polars")

        # "horizontal" is deprecated in favor of "horizontal_extend" ahead of
        # polars 2.0, see https://github.com/pola-rs/polars/issues/27927
        how = "horizontal"
        if parse_version(pl.__version__) >= parse_version("1.42.1"):
            how = "horizontal_extend"

        return pl.concat(Xs, how=how)


class ContainerAdaptersManager:
    # [CLASS DOC]
    # HELPER: Registry mapping container library name to adapter instance.
    def __init__(self):
        # [FUNCTION DOC]
        # Initialises an empty adapters dict.
        self.adapters = {}

    @property
    def supported_outputs(self):
        # [FUNCTION DOC]
        # Property: set of valid output config values, 'default' plus registered names.
        return {"default"} | set(self.adapters)

    def register(self, adapter):
        # [FUNCTION DOC]
        # Registers adapter under its container_lib, replacing any previous one.
        # Side effects: mutates adapters.
        self.adapters[adapter.container_lib] = adapter


ADAPTERS_MANAGER = ContainerAdaptersManager()
ADAPTERS_MANAGER.register(PandasAdapter())
ADAPTERS_MANAGER.register(PolarsAdapter())


def _get_adapter_from_container(container):
    # [FUNCTION DOC]
    # HELPER: Returns the adapter registered for the top-level package of the
    # container's class module (e.g. 'pandas').
    # Exceptions: ValueError listing available adapters if none is registered.
    """Get the adapter that knows how to handle such container.

    See :class:`sklearn.utils._set_output.ContainerAdapterProtocol` for more
    details.
    """
    module_name = container.__class__.__module__.split(".")[0]
    try:
        return ADAPTERS_MANAGER.adapters[module_name]
    except KeyError as exc:
        available_adapters = list(ADAPTERS_MANAGER.adapters.keys())
        raise ValueError(
            "The container does not have a registered adapter in scikit-learn. "
            f"Available adapters are: {available_adapters} while the container "
            f"provided is: {container!r}."
        ) from exc


def _get_container_adapter(method, estimator=None):
    # [FUNCTION DOC]
    # HELPER: Returns the adapter matching the dense output config for method and
    # estimator, or None when the config is 'default'.
    # Exceptions: ValueError from _get_output_config for invalid configs.
    """Get container adapter."""
    dense_config = _get_output_config(method, estimator)["dense"]
    try:
        return ADAPTERS_MANAGER.adapters[dense_config]
    except KeyError:
        return None


def _get_output_config(method, estimator=None):
    # [FUNCTION DOC]
    # HELPER: Looks up the output container for method: the estimator's
    # _sklearn_output_config entry if present, else the global '<method>_output'
    # config.
    # Outputs: dict with key 'dense'.
    # Exceptions: ValueError if the value is not a supported output; KeyError if the
    # global config lacks the key for an unknown method.
    """Get output config based on estimator and global configuration.

    Parameters
    ----------
    method : {"transform"}
        Estimator's method for which the output container is looked up.

    estimator : estimator instance or None
        Estimator to get the output configuration from. If `None`, check global
        configuration is used.

    Returns
    -------
    config : dict
        Dictionary with keys:

        - "dense": specifies the dense container for `method`. This can be
          `"default"` or `"pandas"`.
    """
    est_sklearn_output_config = getattr(estimator, "_sklearn_output_config", {})
    if method in est_sklearn_output_config:
        dense_config = est_sklearn_output_config[method]
    else:
        dense_config = get_config()[f"{method}_output"]

    supported_outputs = ADAPTERS_MANAGER.supported_outputs
    if dense_config not in supported_outputs:
        raise ValueError(
            f"output config must be in {sorted(supported_outputs)}, got {dense_config}"
        )

    return {"dense": dense_config}


def _wrap_data_with_container(method, data_to_wrap, original_input, estimator):
    # [FUNCTION DOC]
    # HELPER: Returns data_to_wrap unchanged when the config is 'default' or the
    # estimator is not auto-wrap configured. Otherwise wraps it with the configured
    # adapter, passing original_input for index and estimator.get_feature_names_out as
    # the (lazily evaluated) column names.
    # Exceptions: ValueError if data_to_wrap is a scipy sparse matrix; ValueError for
    # invalid config.
    """Wrap output with container based on an estimator's or global config.

    Parameters
    ----------
    method : {"transform"}
        Estimator's method to get container output for.

    data_to_wrap : {ndarray, dataframe}
        Data to wrap with container.

    original_input : {ndarray, dataframe}
        Original input of function.

    estimator : estimator instance
        Estimator with to get the output configuration from.

    Returns
    -------
    output : {ndarray, dataframe}
        If the output config is "default" or the estimator is not configured
        for wrapping return `data_to_wrap` unchanged.
        If the output config is "pandas", return `data_to_wrap` as a pandas
        DataFrame.
    """
    output_config = _get_output_config(method, estimator)

    if output_config["dense"] == "default" or not _auto_wrap_is_configured(estimator):
        return data_to_wrap

    dense_config = output_config["dense"]
    if issparse(data_to_wrap):
        raise ValueError(
            "The transformer outputs a scipy sparse matrix. "
            "Try to set the transformer output to a dense array or disable "
            f"{dense_config.capitalize()} output with set_output(transform='default')."
        )

    adapter = ADAPTERS_MANAGER.adapters[dense_config]
    return adapter.create_container(
        data_to_wrap,
        original_input,
        columns=estimator.get_feature_names_out,
    )


def _wrap_method_output(f, method):
    # [FUNCTION DOC]
    # HELPER: Decorates transform-like method f so its output is wrapped per the
    # config key method. Outputs: the wrapper function (metadata copied from f).
    """Wrapper used by `_SetOutputMixin` to automatically wrap methods."""

    @wraps(f)
    def wrapped(self, X, *args, **kwargs):
        # [FUNCTION DOC]
        # Calls f; if the result is a tuple only its first element is wrapped (rest
        # kept,
        # namedtuple types rebuilt via _make); otherwise the whole result is wrapped,
        # using
        # X as the original input.
        data_to_wrap = f(self, X, *args, **kwargs)
        if isinstance(data_to_wrap, tuple):
            # only wrap the first output for cross decomposition
            return_tuple = (
                _wrap_data_with_container(method, data_to_wrap[0], X, self),
                *data_to_wrap[1:],
            )
            # Support for namedtuples `_make` is a documented API for namedtuples:
            # https://docs.python.org/3/library/collections.html#collections.somenamedtuple._make
            if hasattr(type(data_to_wrap), "_make"):
                return type(data_to_wrap)._make(return_tuple)
            return return_tuple

        return _wrap_data_with_container(method, data_to_wrap, X, self)

    return wrapped


def _auto_wrap_is_configured(estimator):
    # [FUNCTION DOC]
    # HELPER: True iff the estimator has get_feature_names_out and 'transform' is among
    # its _sklearn_auto_wrap_output_keys (empty set if absent). Also used as the
    # availability check for set_output.
    """Return True if estimator is configured for auto-wrapping the transform method.

    `_SetOutputMixin` sets `_sklearn_auto_wrap_output_keys` to `set()` if auto wrapping
    is manually disabled.
    """
    auto_wrap_output_keys = getattr(estimator, "_sklearn_auto_wrap_output_keys", set())
    return (
        hasattr(estimator, "get_feature_names_out")
        and "transform" in auto_wrap_output_keys
    )


class _SetOutputMixin:
    # [CLASS DOC]
    # Mixin that, at subclass creation, wraps transform/fit_transform so outputs
    # follow set_output / global config, and offers set_output.
    """Mixin that dynamically wraps methods to return container based on config.

    Currently `_SetOutputMixin` wraps `transform` and `fit_transform` and configures
    it based on `set_output` of the global configuration.

    `set_output` is only defined if `get_feature_names_out` is defined and
    `auto_wrap_output_keys` is the default value.
    """

    def __init_subclass__(cls, auto_wrap_output_keys=("transform",), **kwargs):
        # [FUNCTION DOC]
        # Inputs: auto_wrap_output_keys - tuple of config keys (default ('transform',))
        # or None to disable wrapping; other kwargs forwarded to super.
        # Behaviour: None sets _sklearn_auto_wrap_output_keys to an empty set.
        # Otherwise,
        # for transform and fit_transform (both mapped to key 'transform'), if the class
        # has
        # the method and the key is requested, records the key; if the method is defined
        # in the class's own __dict__, replaces it with the wrapping decorator
        # (inherited
        # methods are already wrapped by their defining class).
        # Exceptions: ValueError if auto_wrap_output_keys is neither None nor a tuple.
        # Side effects: sets class attributes and replaces methods on cls.
        super().__init_subclass__(**kwargs)

        # Dynamically wraps `transform` and `fit_transform` and configure it's
        # output based on `set_output`.
        if not (
            isinstance(auto_wrap_output_keys, tuple) or auto_wrap_output_keys is None
        ):
            raise ValueError("auto_wrap_output_keys must be None or a tuple of keys.")

        if auto_wrap_output_keys is None:
            cls._sklearn_auto_wrap_output_keys = set()
            return

        # Mapping from method to key in configurations
        method_to_key = {
            "transform": "transform",
            "fit_transform": "transform",
        }
        cls._sklearn_auto_wrap_output_keys = set()

        for method, key in method_to_key.items():
            if not hasattr(cls, method) or key not in auto_wrap_output_keys:
                continue
            cls._sklearn_auto_wrap_output_keys.add(key)

            # Only wrap methods defined by cls itself
            if method not in cls.__dict__:
                continue
            wrapped_method = _wrap_method_output(getattr(cls, method), key)
            setattr(cls, method, wrapped_method)

    @available_if(_auto_wrap_is_configured)
    def set_output(self, *, transform=None):
        # [FUNCTION DOC]
        # Available only when _auto_wrap_is_configured. Stores transform in the
        # instance's
        # _sklearn_output_config (created if missing); None leaves config untouched.
        # Outputs: self. Side effects: mutates self. The value is validated lazily at
        # transform time, not here.
        """Set output container.

        Refer to the :ref:`user guide <df_output_transform>` for more details
        and :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py` for an
        example on how to use the API.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `"polars"`: Polars output
            - `None`: Transform configuration is unchanged

            .. versionadded:: 1.4
                `"polars"` option was added.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if transform is None:
            return self

        if not hasattr(self, "_sklearn_output_config"):
            self._sklearn_output_config = {}

        self._sklearn_output_config["transform"] = transform
        return self


def _safe_set_output(estimator, *, transform=None):
    # [FUNCTION DOC]
    # Calls set_output on estimator if it can transform (has transform, or has
    # fit_transform and transform is not None); otherwise does nothing.
    # Outputs: result of set_output (the estimator) or None when skipped.
    # Exceptions: ValueError if the estimator should be configured but lacks set_output.
    # KNOWN ISSUE: the docstring says the estimator is returned, but None is returned
    # when the estimator cannot transform.
    """Safely call estimator.set_output and error if it not available.

    This is used by meta-estimators to set the output for child estimators.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance.

    transform : {"default", "pandas", "polars"}, default=None
        Configure output of the following estimator's methods:

        - `"transform"`
        - `"fit_transform"`

        If `None`, this operation is a no-op.

    Returns
    -------
    estimator : estimator instance
        Estimator instance.
    """
    set_output_for_transform = hasattr(estimator, "transform") or (
        hasattr(estimator, "fit_transform") and transform is not None
    )
    if not set_output_for_transform:
        # If estimator can not transform, then `set_output` does not need to be
        # called.
        return

    if not hasattr(estimator, "set_output"):
        raise ValueError(
            f"Unable to configure output for {estimator} because `set_output` "
            "is not available."
        )
    return estimator.set_output(transform=transform)
