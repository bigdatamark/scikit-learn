# [MODULE DOC]
# Kernel library for Gaussian process regression and classification. Defines the
# Hyperparameter spec namedtuple, the abstract Kernel base class (parameter
# introspection, log-space theta/bounds handling, operator overloading for +, * and **),
# mixins (normalized, stationary, generic-object), composite kernels (CompoundKernel,
# Sum, Product, Exponentiation) and concrete kernels (ConstantKernel, WhiteKernel, RBF,
# Matern, RationalQuadratic, ExpSineSquared, DotProduct, PairwiseKernel).
# Hyperparameters are discovered by scanning for attributes named "hyperparameter_*";
# theta is the log of the concatenated non-fixed hyperparameter values. Side effects at
# import: wraps inspect.signature in an LRU cache (maxsize 32) stored in the
# module-level name signature, used by Kernel.get_params.
"""A set of kernels that can be combined by operators and used in Gaussian processes."""

# Kernels for Gaussian process regression and classification.
#
# The kernels in this module allow kernel-engineering, i.e., they can be
# combined via the "+" and "*" operators or be exponentiated with a scalar
# via "**". These sum and product expressions can also contain scalar values,
# which are automatically converted to a constant kernel.
#
# All kernels allow (analytic) gradient-based hyperparameter optimization.
# The space of hyperparameters can be specified by giving lower und upper
# boundaries for the value of each hyperparameter (the search space is thus
# rectangular). Instead of specifying bounds, hyperparameters can also be
# declared to be "fixed", which causes these hyperparameters to be excluded from
# optimization.


# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Note: this module is strongly inspired by the kernel module of the george
#       package.

import inspect
import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import _num_samples

# Cache constructor signature inspection for kernels as it empirically
# proves to account for 15% or more of the total grid-search time of GP
# model on small to medium data.
signature = lru_cache(maxsize=32)(inspect.signature)


def _check_length_scale(X, length_scale):
    # [FUNCTION DOC]
    # HELPER: Validates a length_scale against the data X and normalizes it.
    # Inputs: X - 2D array (n_samples, n_features); length_scale - scalar or 1D
    # array-like.
    # Outputs: length_scale squeezed and cast to float (0-d for isotropic, 1D
    # otherwise).
    # Exceptions: ValueError if the squeezed length_scale has more than one dimension,
    # or if it is 1D and its length differs from X.shape[1].
    # Side effects: none.
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class Hyperparameter(
    namedtuple(
        "Hyperparameter", ("name", "value_type", "bounds", "n_elements", "fixed")
    )
):
    # [CLASS DOC]
    # Immutable namedtuple describing one kernel hyperparameter (name, value_type,
    # bounds, n_elements, fixed). Uses empty __slots__ so instances stay as compact as
    # plain tuples.
    """A kernel hyperparameter's specification in form of a namedtuple.

    .. versionadded:: 0.18

    Attributes
    ----------
    name : str
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds

    value_type : str
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.

    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the hyperparameter's value
        cannot be changed.

    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.

    fixed : bool, default=None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the "fixed" is
        derived based on the given bounds.

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import ConstantKernel
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import Hyperparameter
    >>> X, y = make_friedman2(n_samples=50, noise=0, random_state=0)
    >>> kernel = ConstantKernel(constant_value=1.0,
    ...    constant_value_bounds=(0.0, 10.0))

    We can access each hyperparameter:

    >>> for hyperparameter in kernel.hyperparameters:
    ...    print(hyperparameter)
    Hyperparameter(name='constant_value', value_type='numeric',
    bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)

    >>> params = kernel.get_params()
    >>> for key in sorted(params): print(f"{key} : {params[key]}")
    constant_value : 1.0
    constant_value_bounds : (0.0, 10.0)
    """

    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None):
        # [FUNCTION DOC]
        # Constructs the tuple, normalizing bounds. Unless bounds is the string "fixed",
        # bounds is converted to a 2D array; for vector-valued hyperparameters
        # (n_elements > 1) a single bounds row is repeated n_elements times.
        # Inputs: name (str), value_type (str), bounds (pair, array of pairs, or
        # "fixed"), n_elements (int, default 1), fixed (bool or None; None means derive
        # from bounds).
        # Outputs: new Hyperparameter instance; fixed is True only if bounds == "fixed"
        # when fixed was None.
        # Exceptions: ValueError if n_elements > 1 and bounds has neither 1 nor
        # n_elements rows.
        if not isinstance(bounds, str) or bounds != "fixed":
            bounds = np.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError(
                        "Bounds on %s should have either 1 or "
                        "%d dimensions. Given are %d"
                        % (name, n_elements, bounds.shape[0])
                    )

        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        return super().__new__(cls, name, value_type, bounds, n_elements, fixed)

    # This is mainly a testing utility to check that two hyperparameters
    # are equal.
    def __eq__(self, other):
        # [FUNCTION DOC]
        # Field-wise equality used mainly in tests; bounds compared element-wise with
        # all().
        # Inputs: other - another Hyperparameter-like object. Outputs: bool.
        # KNOWN ISSUE: overriding __eq__ without __hash__ makes instances unhashable,
        # and an other object lacking these attributes raises AttributeError instead of
        # returning False.
        return (
            self.name == other.name
            and self.value_type == other.value_type
            and np.all(self.bounds == other.bounds)
            and self.n_elements == other.n_elements
            and self.fixed == other.fixed
        )


class Kernel(metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base class for all kernels. Subclasses must implement __call__, diag and
    # is_stationary; hyperparameters are exposed through properties named
    # "hyperparameter_<name>" returning Hyperparameter objects, and constructor
    # arguments must be stored as same-named attributes so get_params/set_params work.
    """Base class for all kernels.

    .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import Kernel, RBF
    >>> import numpy as np
    >>> class CustomKernel(Kernel):
    ...     def __init__(self, length_scale=1.0):
    ...         self.length_scale = length_scale
    ...     def __call__(self, X, Y=None):
    ...         if Y is None:
    ...             Y = X
    ...         return np.inner(X, X if Y is None else Y) ** 2
    ...     def diag(self, X):
    ...         return np.ones(X.shape[0])
    ...     def is_stationary(self):
    ...         return True
    >>> kernel = CustomKernel(length_scale=2.0)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(kernel(X))
    [[ 25 121]
     [121 625]]
    """

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # Collects constructor parameters by inspecting the (cached) signature of the
        # class __init__, skipping self and **kwargs, and reading each as an attribute
        # of self.
        # Inputs: deep - bool, default True (accepted for API compatibility).
        # Outputs: dict mapping parameter name to current value.
        # Exceptions: RuntimeError if __init__ declares *args.
        # KNOWN ISSUE: deep is ignored here; nested parameters are only expanded by
        # subclasses (KernelOperator, Exponentiation) that override this method.
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init_sign = signature(cls.__init__)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError(
                "scikit-learn kernels should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s doesn't follow this convention." % (cls,)
            )
        for arg in args:
            params[arg] = getattr(self, arg)

        return params

    def set_params(self, **params):
        # [FUNCTION DOC]
        # Sets parameters, supporting nested "<component>__<param>" keys which are
        # forwarded to the sub-kernel's set_params (only the first "__" is split on).
        # Inputs: **params - parameter names and values. Outputs: self.
        # Exceptions: ValueError if a top-level name is not among get_params(deep=True)
        # keys.
        # Side effects: mutates self (setattr) or nested kernels in place. Returns
        # immediately if params is empty.
        """Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        # [FUNCTION DOC]
        # Returns a deep clone (sklearn.base.clone) of the kernel with theta replaced.
        # Inputs: theta - ndarray (n_dims,) of log-transformed non-fixed
        # hyperparameters.
        # Outputs: new Kernel. Side effects: none on self.
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        # [FUNCTION DOC]
        # Number of non-fixed hyperparameter entries, i.e. the length of theta. Outputs:
        # int.
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        # [FUNCTION DOC]
        # Lists all Hyperparameter objects by reading every attribute of self whose name
        # starts with "hyperparameter_" (in dir() order, i.e. alphabetical). Outputs:
        # list.
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        return r

    @property
    def theta(self):
        # [FUNCTION DOC]
        # Getter: concatenates the current values of all non-fixed hyperparameters (in
        # the order of hyperparameters) and returns their natural log as a 1D array;
        # empty array if none.
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        # [FUNCTION DOC]
        # Setter: walks non-fixed hyperparameters in order, consuming n_elements entries
        # of theta for each, exponentiating them and assigning via set_params.
        # Inputs: theta - 1D array of log-values.
        # Exceptions: ValueError if theta length differs from the number of entries
        # consumed (checked only after all hyperparameters are processed; too-short
        # theta may raise an IndexError first).
        # Side effects: mutates hyperparameter attributes of self (or nested kernels).
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = np.exp(
                    theta[i : i + hyperparameter.n_elements]
                )
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (i, len(theta))
            )
        self.set_params(**params)

    @property
    def bounds(self):
        # [FUNCTION DOC]
        # Returns log of the vertically stacked bounds of non-fixed hyperparameters,
        # shape (n_dims, 2); empty array if all are fixed.
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [
            hyperparameter.bounds
            for hyperparameter in self.hyperparameters
            if not hyperparameter.fixed
        ]
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def __add__(self, b):
        # [FUNCTION DOC]
        # Implements kernel + b; non-Kernel b is wrapped in ConstantKernel. Outputs: Sum
        # kernel.
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        # [FUNCTION DOC]
        # Implements b + kernel (reflected); non-Kernel b wrapped in ConstantKernel.
        # Outputs: Sum.
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        # [FUNCTION DOC]
        # Implements kernel * b; non-Kernel b wrapped in ConstantKernel. Outputs:
        # Product.
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        # [FUNCTION DOC]
        # Implements b * kernel (reflected); non-Kernel b wrapped in ConstantKernel.
        # Outputs: Product kernel.
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        # [FUNCTION DOC]
        # Implements kernel ** b. Inputs: b - scalar exponent. Outputs: Exponentiation
        # kernel.
        return Exponentiation(self, b)

    def __eq__(self, b):
        # [FUNCTION DOC]
        # Equality: same concrete type and every parameter from get_params equal
        # (compared with any() over element-wise inequality, so arrays work). Outputs:
        # bool.
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        # [FUNCTION DOC]
        # Default repr: class name followed by the theta values formatted with 3
        # significant digits. Outputs: str.
        return "{0}({1})".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.theta))
        )

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Abstract: evaluates k(X, Y) (and optionally the gradient w.r.t. log
        # hyperparameters). Must be implemented by subclasses.
        """Evaluate the kernel."""

    @abstractmethod
    def diag(self, X):
        # [FUNCTION DOC]
        # Abstract: returns the diagonal of k(X, X) as a 1D array of length n_samples_X.
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """

    @abstractmethod
    def is_stationary(self):
        # [FUNCTION DOC]
        # Abstract: returns whether the kernel is stationary (depends only on X - Y).
        """Returns whether the kernel is stationary."""

    @property
    def requires_vector_input(self):
        # [FUNCTION DOC]
        # Whether the kernel needs fixed-length feature vectors as input. Base default:
        # True.
        """Returns whether the kernel is defined on fixed-length feature
        vectors or generic objects. Defaults to True for backward
        compatibility."""
        return True

    def _check_bounds_params(self):
        # [FUNCTION DOC]
        # Called after fitting a GP: for each non-fixed hyperparameter dimension, checks
        # whether the current theta is numerically close (np.isclose) to its log lower
        # or upper bound.
        # Outputs: None.
        # Side effects: emits a ConvergenceWarning per dimension close to a bound (lower
        # bound checked first; at most one warning per dimension).
        """Called after fitting to warn if bounds may have been too tight."""
        list_close = np.isclose(self.bounds, np.atleast_2d(self.theta).T)
        idx = 0
        for hyp in self.hyperparameters:
            if hyp.fixed:
                continue
            for dim in range(hyp.n_elements):
                if list_close[idx, 0]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified lower "
                        "bound %s. Decreasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][0]),
                        ConvergenceWarning,
                    )
                elif list_close[idx, 1]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified upper "
                        "bound %s. Increasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][1]),
                        ConvergenceWarning,
                    )
                idx += 1


class NormalizedKernelMixin:
    # [CLASS DOC]
    # Mixin for kernels with k(x, x) = 1, providing a constant-one diag.
    """Mixin for kernels which are normalized: k(X, X)=1.

    .. versionadded:: 0.18
    """

    def diag(self, X):
        # [FUNCTION DOC]
        # Returns an array of ones of length X.shape[0].
        # Inputs: X - array with a shape attribute. Outputs: float ndarray
        # (n_samples_X,).
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.ones(X.shape[0])


class StationaryKernelMixin:
    # [CLASS DOC]
    # Mixin marking a kernel as stationary.
    """Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
    """

    def is_stationary(self):
        # [FUNCTION DOC]
        # Always returns True.
        """Returns whether the kernel is stationary."""
        return True


class GenericKernelMixin:
    # [CLASS DOC]
    # Mixin for kernels that accept generic objects (sequences, graphs) instead of
    # vectors.
    """Mixin for kernels which operate on generic objects such as variable-
    length sequences, trees, and graphs.

    .. versionadded:: 0.22
    """

    @property
    def requires_vector_input(self):
        # [FUNCTION DOC]
        # Always returns False (kernel does not require vector input).
        """Whether the kernel works only on fixed-length feature vectors."""
        return False


class CompoundKernel(Kernel):
    # [CLASS DOC]
    # Kernel holding a list of sub-kernels; evaluation stacks each sub-kernel's output
    # along a new last axis instead of combining them. Used internally (e.g. multi-class
    # GPC).
    """Kernel which is composed of a set of other kernels.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernels : list of Kernels
        The other kernels

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import WhiteKernel
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> from sklearn.gaussian_process.kernels import CompoundKernel
    >>> kernel = CompoundKernel(
    ...     [WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    >>> print(kernel.bounds)
    [[-11.51292546  11.51292546]
     [-11.51292546  11.51292546]]
    >>> print(kernel.n_dims)
    2
    >>> print(kernel.theta)
    [1.09861229 0.69314718]
    """

    def __init__(self, kernels):
        # [FUNCTION DOC]
        # Stores the list of kernels. Inputs: kernels - list of Kernel instances.
        self.kernels = kernels

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # Returns {"kernels": self.kernels}; deep is ignored (sub-kernel params not
        # expanded).
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return dict(kernels=self.kernels)

    @property
    def theta(self):
        # [FUNCTION DOC]
        # Getter: concatenation of each sub-kernel's theta, in list order.
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        # [FUNCTION DOC]
        # Setter: splits theta into consecutive equal-size chunks, one per sub-kernel.
        # Inputs: theta - 1D array.
        # Side effects: sets theta on each sub-kernel.
        # KNOWN ISSUE: reads self.k1.n_dims but CompoundKernel has no k1 attribute, so
        # this raises AttributeError; it also assumes all sub-kernels have the same
        # n_dims.
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k_dims = self.k1.n_dims
        for i, kernel in enumerate(self.kernels):
            kernel.theta = theta[i * k_dims : (i + 1) * k_dims]

    @property
    def bounds(self):
        # [FUNCTION DOC]
        # Vertical stack of each sub-kernel's bounds, shape (total n_dims, 2).
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return np.vstack([kernel.bounds for kernel in self.kernels])

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Evaluates every sub-kernel on (X, Y) and stacks the results.
        # Inputs: X, Y (Y may be None), eval_gradient - bool.
        # Outputs: K of shape (n_X, n_Y, n_kernels); if eval_gradient, also the gradient
        # with each sub-kernel's gradient given a trailing axis and concatenated along
        # axis 3, shape (n_X, n_X, n_dims, n_kernels) (requires equal n_dims across
        # sub-kernels).
        # Exceptions: propagated from sub-kernels (e.g. ValueError for gradient with Y
        # given).
        """Return the kernel k(X, Y) and optionally its gradient.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of the
            kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y, n_kernels)
            Kernel k(X, Y)

        K_gradient : ndarray of shape \
                (n_samples_X, n_samples_X, n_dims, n_kernels), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K = []
            K_grad = []
            for kernel in self.kernels:
                K_single, K_grad_single = kernel(X, Y, eval_gradient)
                K.append(K_single)
                K_grad.append(K_grad_single[..., np.newaxis])
            return np.dstack(K), np.concatenate(K_grad, 3)
        else:
            return np.dstack([kernel(X, Y, eval_gradient) for kernel in self.kernels])

    def __eq__(self, b):
        # [FUNCTION DOC]
        # Equality: same type, same number of sub-kernels and pairwise-equal sub-kernels
        # in order.
        if type(self) != type(b) or len(self.kernels) != len(b.kernels):
            return False
        return np.all(
            [self.kernels[i] == b.kernels[i] for i in range(len(self.kernels))]
        )

    def is_stationary(self):
        # [FUNCTION DOC]
        # True only if all sub-kernels are stationary.
        """Returns whether the kernel is stationary."""
        return np.all([kernel.is_stationary() for kernel in self.kernels])

    @property
    def requires_vector_input(self):
        # [FUNCTION DOC]
        # True if any sub-kernel requires vector input.
        """Returns whether the kernel is defined on discrete structures."""
        return np.any([kernel.requires_vector_input for kernel in self.kernels])

    def diag(self, X):
        # [FUNCTION DOC]
        # Stacks the diagonals of each sub-kernel and transposes. Outputs: (n_samples_X,
        # n_kernels).
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X, n_kernels)
            Diagonal of kernel k(X, X)
        """
        return np.vstack([kernel.diag(X) for kernel in self.kernels]).T


class KernelOperator(Kernel):
    # [CLASS DOC]
    # Base class for binary kernel operators combining k1 and k2. Exposes nested
    # parameters and hyperparameters with "k1__" / "k2__" prefixes and theta as k1's
    # followed by k2's.
    """Base class for all kernel operators.

    .. versionadded:: 0.18
    """

    def __init__(self, k1, k2):
        # [FUNCTION DOC]
        # Stores the two operand kernels. Inputs: k1, k2 - Kernel instances.
        self.k1 = k1
        self.k2 = k2

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # Returns {"k1", "k2"} plus, if deep, each operand's params prefixed by
        # "k1__"/"k2__".
        # Inputs: deep - bool, default True. Outputs: dict.
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(("k1__" + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(("k2__" + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        # [FUNCTION DOC]
        # Builds Hyperparameter copies of k1's and k2's hyperparameters with names
        # prefixed "k1__" / "k2__". Outputs: list of Hyperparameter.
        # Note: fixed is re-derived from bounds rather than copied from the operand.
        """Returns a list of all hyperparameter."""
        r = [
            Hyperparameter(
                "k1__" + hyperparameter.name,
                hyperparameter.value_type,
                hyperparameter.bounds,
                hyperparameter.n_elements,
            )
            for hyperparameter in self.k1.hyperparameters
        ]

        for hyperparameter in self.k2.hyperparameters:
            r.append(
                Hyperparameter(
                    "k2__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self):
        # [FUNCTION DOC]
        # Getter: k1.theta followed by k2.theta.
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        # [FUNCTION DOC]
        # Setter: first k1.n_dims entries go to k1.theta, the rest to k2.theta.
        # Side effects: mutates both operand kernels.
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        # [FUNCTION DOC]
        # Stack of k1.bounds and k2.bounds; if either is empty, returns the other alone.
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        if self.k1.bounds.size == 0:
            return self.k2.bounds
        if self.k2.bounds.size == 0:
            return self.k1.bounds
        return np.vstack((self.k1.bounds, self.k2.bounds))

    def __eq__(self, b):
        # [FUNCTION DOC]
        # Equality: same type and operands equal in either order (treats the operator as
        # commutative, which also applies to subclasses).
        if type(self) != type(b):
            return False
        return (self.k1 == b.k1 and self.k2 == b.k2) or (
            self.k1 == b.k2 and self.k2 == b.k1
        )

    def is_stationary(self):
        # [FUNCTION DOC]
        # True if both operands are stationary.
        """Returns whether the kernel is stationary."""
        return self.k1.is_stationary() and self.k2.is_stationary()

    @property
    def requires_vector_input(self):
        # [FUNCTION DOC]
        # True if either operand requires vector input.
        # KNOWN ISSUE: docstring says "Returns whether the kernel is stationary", which
        # does not match what is computed.
        """Returns whether the kernel is stationary."""
        return self.k1.requires_vector_input or self.k2.requires_vector_input


class Sum(KernelOperator):
    # [CLASS DOC]
    # Sum of two kernels: k(X, Y) = k1(X, Y) + k2(X, Y).
    """The `Sum` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{sum}(X, Y) = k_1(X, Y) + k_2(X, Y)

    Note that the `__add__` magic method is overridden, so
    `Sum(RBF(), RBF())` is equivalent to using the + operator
    with `RBF() + RBF()`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the sum-kernel

    k2 : Kernel
        The second base-kernel of the sum-kernel

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, Sum, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Sum(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 + RBF(length_scale=1)
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Evaluates k1 + k2 on (X, Y).
        # Outputs: K (n_X, n_Y); if eval_gradient, also the gradients of k1 and k2
        # stacked along the last axis (derivative of a sum is the sum's per-parameter
        # derivatives).
        # Exceptions: propagated from operands.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
                default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
            return K1 + K2, np.dstack((K1_gradient, K2_gradient))
        else:
            return self.k1(X, Y) + self.k2(X, Y)

    def diag(self, X):
        # [FUNCTION DOC]
        # Sum of the operands' diagonals. Outputs: (n_samples_X,).
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(X) + self.k2.diag(X)

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "<k1> + <k2>".
        return "{0} + {1}".format(self.k1, self.k2)


class Product(KernelOperator):
    # [CLASS DOC]
    # Product of two kernels: k(X, Y) = k1(X, Y) * k2(X, Y).
    """The `Product` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{prod}(X, Y) = k_1(X, Y) * k_2(X, Y)

    Note that the `__mul__` magic method is overridden, so
    `Product(RBF(), RBF())` is equivalent to using the * operator
    with `RBF() * RBF()`.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the product-kernel

    k2 : Kernel
        The second base-kernel of the product-kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RBF, Product,
    ...            ConstantKernel)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Product(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 * RBF(length_scale=1)
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Evaluates k1 * k2 element-wise on (X, Y).
        # Outputs: K (n_X, n_Y); if eval_gradient, also the gradient by the product
        # rule: k1's gradient times K2 stacked with k2's gradient times K1 along the
        # last axis.
        # Exceptions: propagated from operands.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
            return K1 * K2, np.dstack(
                (K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis])
            )
        else:
            return self.k1(X, Y) * self.k2(X, Y)

    def diag(self, X):
        # [FUNCTION DOC]
        # Product of the operands' diagonals. Outputs: (n_samples_X,).
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(X) * self.k2.diag(X)

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "<k1> * <k2>".
        return "{0} * {1}".format(self.k1, self.k2)


class Exponentiation(Kernel):
    # [CLASS DOC]
    # Raises a base kernel to a fixed scalar power: k(X, Y) = kernel(X, Y) ** exponent.
    # The exponent is not a hyperparameter; nested names use the "kernel__" prefix.
    """The Exponentiation kernel takes one base kernel and a scalar parameter
    :math:`p` and combines them via

    .. math::
        k_{exp}(X, Y) = k(X, Y) ^p

    Note that the `__pow__` magic method is overridden, so
    `Exponentiation(RBF(), 2)` is equivalent to using the ** operator
    with `RBF() ** 2`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : Kernel
        The base kernel

    exponent : float
        The exponent for the base kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RationalQuadratic,
    ...            Exponentiation)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Exponentiation(RationalQuadratic(), exponent=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.419
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([635.5]), array([0.559]))
    """

    def __init__(self, kernel, exponent):
        # [FUNCTION DOC]
        # Stores base kernel and exponent. Inputs: kernel - Kernel; exponent - float.
        self.kernel = kernel
        self.exponent = exponent

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # Returns {"kernel", "exponent"} plus, if deep, base kernel params prefixed
        # "kernel__".
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(("kernel__" + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        # [FUNCTION DOC]
        # Copies of the base kernel's hyperparameters renamed with the "kernel__"
        # prefix.
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(
                Hyperparameter(
                    "kernel__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self):
        # [FUNCTION DOC]
        # Getter: delegates to the base kernel's theta.
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        # [FUNCTION DOC]
        # Setter: assigns theta to the base kernel. Side effects: mutates the base
        # kernel.
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        # [FUNCTION DOC]
        # Delegates to the base kernel's bounds.
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.kernel.bounds

    def __eq__(self, b):
        # [FUNCTION DOC]
        # Equality: same type, equal base kernels and equal exponents.
        if type(self) != type(b):
            return False
        return self.kernel == b.kernel and self.exponent == b.exponent

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Evaluates the base kernel and raises it element-wise to exponent.
        # Outputs: K (n_X, n_Y); if eval_gradient, also the chain-rule gradient: the
        # base gradient multiplied by exponent * K_base ** (exponent - 1).
        # Side effects: the base kernel's gradient array is modified in place (it is
        # freshly created by the base kernel, so no caller-visible effect).
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K, K_gradient = self.kernel(X, Y, eval_gradient=True)
            K_gradient *= self.exponent * K[:, :, np.newaxis] ** (self.exponent - 1)
            return K**self.exponent, K_gradient
        else:
            K = self.kernel(X, Y, eval_gradient=False)
            return K**self.exponent

    def diag(self, X):
        # [FUNCTION DOC]
        # Base kernel's diagonal raised to exponent. Outputs: (n_samples_X,).
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.kernel.diag(X) ** self.exponent

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "<kernel> ** <exponent>".
        return "{0} ** {1}".format(self.kernel, self.exponent)

    def is_stationary(self):
        # [FUNCTION DOC]
        # Delegates to the base kernel.
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()

    @property
    def requires_vector_input(self):
        # [FUNCTION DOC]
        # Delegates to the base kernel.
        """Returns whether the kernel is defined on discrete structures."""
        return self.kernel.requires_vector_input


class ConstantKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    # [CLASS DOC]
    # Kernel returning the same value constant_value for every pair of inputs; works on
    # generic objects (only the number of samples is used).
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = RBF() + ConstantKernel(constant_value=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3696
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([606.1]), array([0.248]))
    """

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        # [FUNCTION DOC]
        # Stores constant_value (float, default 1.0) and constant_value_bounds (pair or
        # "fixed", default (1e-5, 1e5)).
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for constant_value using constant_value_bounds.
        return Hyperparameter("constant_value", "numeric", self.constant_value_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Builds an (n_X, n_Y) array filled with constant_value (dtype from
        # constant_value).
        # Inputs: X, Y (Y None means Y = X), eval_gradient - bool.
        # Outputs: K; if eval_gradient, also the gradient w.r.t. log(constant_value),
        # which equals constant_value everywhere, shape (n_X, n_X, 1), or an empty (n_X,
        # n_X, 0) array when the hyperparameter is fixed.
        # Exceptions: ValueError if eval_gradient is True and Y is given.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = np.full(
            (_num_samples(X), _num_samples(Y)),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    np.full(
                        (_num_samples(X), _num_samples(X), 1),
                        self.constant_value,
                        dtype=np.array(self.constant_value).dtype,
                    ),
                )
            else:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
        else:
            return K

    def diag(self, X):
        # [FUNCTION DOC]
        # Array of length n_samples(X) filled with constant_value.
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(
            _num_samples(X),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns sqrt(constant_value) formatted to 3 significant digits followed by
        # "**2".
        return "{0:.3g}**2".format(np.sqrt(self.constant_value))


class WhiteKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    # [CLASS DOC]
    # White-noise kernel: noise_level on the diagonal of k(X, X), zero elsewhere and
    # zero for any cross-evaluation k(X, Y) with Y given (even if X and Y share
    # samples).
    """White kernel.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise of the signal as independently and identically
    normally-distributed. The parameter noise_level equals the variance of this
    noise.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    noise_level : float, default=1.0
        Parameter controlling the noise level (variance)

    noise_level_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'noise_level'.
        If set to "fixed", 'noise_level' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0, 592.1 ]), array([316.6, 316.6]))
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        # [FUNCTION DOC]
        # Stores noise_level (float, default 1.0) and noise_level_bounds (default (1e-5,
        # 1e5)).
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_noise_level(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for noise_level using noise_level_bounds.
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # If Y is None, returns noise_level times the identity of size n_X; otherwise an
        # all-zero (n_X, n_Y) array.
        # Outputs: K; if eval_gradient, also gradient noise_level times identity with a
        # trailing axis (n_X, n_X, 1), or empty (n_X, n_X, 0) if fixed.
        # Exceptions: ValueError if eval_gradient is True and Y is given.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * np.eye(_num_samples(X))
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (
                        K,
                        self.noise_level * np.eye(_num_samples(X))[:, :, np.newaxis],
                    )
                else:
                    return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        # [FUNCTION DOC]
        # Array of length n_samples(X) filled with noise_level.
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(
            _num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype
        )

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "WhiteKernel(noise_level=<3 sig. digits>)" (uses actual class name).
        return "{0}(noise_level={1:.3g})".format(
            self.__class__.__name__, self.noise_level
        )


class RBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    # [CLASS DOC]
    # Squared-exponential kernel exp(-0.5 * squared Euclidean distance of
    # X/length_scale), isotropic (scalar length_scale) or anisotropic (one length scale
    # per feature).
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354, 0.03228, 0.1322],
           [0.7906, 0.0652, 0.1441]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        # [FUNCTION DOC]
        # Stores length_scale (float or array, default 1.0) and length_scale_bounds.
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        # [FUNCTION DOC]
        # True if length_scale is iterable with more than one element.
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for length_scale; vector-valued with n_elements equal to
        # the length of length_scale when anisotropic.
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Scales X (and Y) by length_scale, computes squared Euclidean distances (pdist
        # then squareform with unit diagonal when Y is None; cdist otherwise) and
        # returns exp(-0.5 * dists).
        # Inputs: X - array (made at least 2D), Y - optional array, eval_gradient -
        # bool.
        # Outputs: K (n_X, n_Y); if eval_gradient, also the gradient w.r.t. log
        # length_scale: empty (n_X, n_X, 0) if fixed; K times the squared scaled
        # distance with a trailing axis for isotropic (or single-element) length scales;
        # for anisotropic, per-feature squared scaled differences times K, shape (n_X,
        # n_X, n_features).
        # Exceptions: ValueError if eval_gradient with Y given; ValueError from
        # _check_length_scale for incompatible length_scale.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "RBF(length_scale=...)" listing all length scales when anisotropic,
        # otherwise the single value; 3 significant digits.
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )


class Matern(RBF):
    # [CLASS DOC]
    # Matern kernel, an RBF generalization with smoothness parameter nu (not optimized).
    # Uses closed forms for nu in {0.5, 1.5, 2.5, inf} and the Bessel-function formula
    # otherwise.
    """Matern kernel.

    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is given by:

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)



    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import Matern
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8513, 0.0368, 0.1117],
            [0.8086, 0.0693, 0.1220]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        # [FUNCTION DOC]
        # Stores length_scale, length_scale_bounds (via RBF.__init__) and nu (default
        # 1.5).
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Computes Euclidean distances d between X/length_scale and Y/length_scale
        # (condensed pdist when Y is None) and evaluates: nu=0.5 exp(-d); nu=1.5
        # (1+sqrt3 d)exp(-sqrt3 d); nu=2.5 (1+sqrt5 d+5d^2/3)exp(-sqrt5 d); nu=inf
        # exp(-d^2/2); otherwise the general formula 2^(1-nu)/Gamma(nu) * (sqrt(2nu)
        # d)^nu * K_nu(sqrt(2nu) d), with exact zero distances bumped to machine epsilon
        # to avoid NaN. When Y is None the result is expanded to a square matrix with
        # unit diagonal.
        # Outputs: K (n_X, n_Y); if eval_gradient, also the gradient w.r.t. log
        # length_scale: empty if fixed; analytic per-feature (anisotropic) or summed
        # (isotropic) formulas for the four special nu values using squared scaled
        # differences D; for general nu, a numerical forward-difference gradient via
        # _approx_fprime with step 1e-10.
        # Exceptions: ValueError if eval_gradient with Y given; ValueError from
        # _check_length_scale.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="euclidean")
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
            else:
                D = squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                divide_result = np.zeros_like(D)
                np.divide(
                    D,
                    denominator,
                    out=divide_result,
                    where=denominator != 0,
                )
                K_gradient = K[..., np.newaxis] * divide_result
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    # [FUNCTION DOC]
                    # HELPER: Evaluates a clone of this kernel with the given theta on
                    # the captured X, Y; used for numerical differentiation. Inputs:
                    # theta - 1D array. Outputs: kernel matrix.
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "Matern(length_scale=..., nu=...)" with all length scales when
        # anisotropic.
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
            )
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu
            )


class RationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    # [CLASS DOC]
    # Rational quadratic kernel (1 + d^2 / (2 alpha l^2))^(-alpha), isotropic only.
    """Rational Quadratic kernel.

    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length scales. It is
    parameterized by a length scale parameter :math:`l>0` and a scale
    mixture parameter :math:`\\alpha>0`. Only the isotropic variant
    where length_scale :math:`l` is a scalar is supported at the moment.
    The kernel is given by:

    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}

    where :math:`\\alpha` is the scale mixture parameter, :math:`l` is
    the length scale of the kernel and :math:`d(\\cdot,\\cdot)` is the
    Euclidean distance.
    For advice on how to set the parameters, see e.g. [1]_.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    alpha : float > 0, default=1.0
        Scale mixture parameter

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'alpha'.
        If set to "fixed", 'alpha' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RationalQuadratic
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8881, 0.0566, 0.05518],
            [0.8678, 0.0707 , 0.0614]])
    """

    def __init__(
        self,
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-5, 1e5),
        alpha_bounds=(1e-5, 1e5),
    ):
        # [FUNCTION DOC]
        # Stores length_scale, alpha and their bounds (defaults 1.0, 1.0, (1e-5, 1e5)
        # each).
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds

    @property
    def hyperparameter_length_scale(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for length_scale (always scalar).
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_alpha(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for alpha.
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Computes squared Euclidean distances, base = 1 + d^2 / (2 alpha l^2) and K =
        # base to the power -alpha (unit diagonal forced when Y is None).
        # Outputs: K (n_X, n_Y); if eval_gradient, also gradients w.r.t. log alpha and
        # log length_scale stacked in that order (alphabetical, matching
        # hyperparameters/theta): d log l: d^2 K / (l^2 base); d log alpha: K (-alpha
        # log(base) + d^2 / (2 l^2 base)); each replaced by an empty slice if that
        # hyperparameter is fixed.
        # Exceptions: AttributeError if length_scale has more than one element;
        # ValueError if eval_gradient with Y given.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if len(np.atleast_1d(self.length_scale)) > 1:
            raise AttributeError(
                "RationalQuadratic kernel only supports isotropic version, "
                "please use a single scalar for length_scale"
            )
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric="sqeuclidean"))
            tmp = dists / (2 * self.alpha * self.length_scale**2)
            base = 1 + tmp
            K = base**-self.alpha
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric="sqeuclidean")
            K = (1 + dists / (2 * self.alpha * self.length_scale**2)) ** -self.alpha

        if eval_gradient:
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = dists * K / (self.length_scale**2 * base)
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # l is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))

            # gradient with respect to alpha
            if not self.hyperparameter_alpha.fixed:
                alpha_gradient = K * (
                    -self.alpha * np.log(base)
                    + dists / (2 * self.length_scale**2 * base)
                )
                alpha_gradient = alpha_gradient[:, :, np.newaxis]
            else:  # alpha is kept fixed
                alpha_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((alpha_gradient, length_scale_gradient))
        else:
            return K

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "RationalQuadratic(alpha=..., length_scale=...)" with 3 significant
        # digits.
        return "{0}(alpha={1:.3g}, length_scale={2:.3g})".format(
            self.__class__.__name__, self.alpha, self.length_scale
        )


class ExpSineSquared(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    # [CLASS DOC]
    # Periodic kernel exp(-2 sin^2(pi d / p) / l^2), isotropic only.
    r"""Exp-Sine-Squared kernel (aka periodic kernel).

    The ExpSineSquared kernel allows one to model functions which repeat
    themselves exactly. It is parameterized by a length scale
    parameter :math:`l>0` and a periodicity parameter :math:`p>0`.
    Only the isotropic variant where :math:`l` is a scalar is
    supported at the moment. The kernel is given by:

    .. math::
        k(x_i, x_j) = \text{exp}\left(-
        \frac{ 2\sin^2(\pi d(x_i, x_j)/p) }{ l^ 2} \right)

    where :math:`l` is the length scale of the kernel, :math:`p` the
    periodicity of the kernel and :math:`d(\cdot,\cdot)` is the
    Euclidean distance.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------

    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    periodicity : float > 0, default=1.0
        The periodicity of the kernel.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    periodicity_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'periodicity'.
        If set to "fixed", 'periodicity' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import ExpSineSquared
    >>> X, y = make_friedman2(n_samples=50, noise=0, random_state=0)
    >>> kernel = ExpSineSquared(length_scale=1, periodicity=1)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.0144
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([425.6, 457.5]), array([0.3894, 0.3467]))
    """

    def __init__(
        self,
        length_scale=1.0,
        periodicity=1.0,
        length_scale_bounds=(1e-5, 1e5),
        periodicity_bounds=(1e-5, 1e5),
    ):
        # [FUNCTION DOC]
        # Stores length_scale, periodicity and their bounds (defaults 1.0, 1.0, (1e-5,
        # 1e5)).
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds
        self.periodicity_bounds = periodicity_bounds

    @property
    def hyperparameter_length_scale(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for length_scale.
        """Returns the length scale"""
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_periodicity(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for periodicity.
        return Hyperparameter("periodicity", "numeric", self.periodicity_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Computes Euclidean distances d, arg = pi d / periodicity and K = exp(-2
        # (sin(arg) / length_scale)^2).
        # Outputs: K (n_X, n_Y); if eval_gradient, also gradients w.r.t. log
        # length_scale (4 sin^2(arg) K / l^2) and log periodicity (4 arg cos(arg)
        # sin(arg) K / l^2) stacked in that order; empty slices for fixed
        # hyperparameters.
        # Exceptions: ValueError if eval_gradient with Y given.
        # KNOWN ISSUE: diagonal is not forced to exactly 1 (unlike
        # RBF/RationalQuadratic), though sin(0) = 0 makes it 1 anyway; does not validate
        # that length_scale is scalar.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric="euclidean"))
            arg = np.pi * dists / self.periodicity
            sin_of_arg = np.sin(arg)
            K = np.exp(-2 * (sin_of_arg / self.length_scale) ** 2)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric="euclidean")
            K = np.exp(
                -2 * (np.sin(np.pi / self.periodicity * dists) / self.length_scale) ** 2
            )

        if eval_gradient:
            cos_of_arg = np.cos(arg)
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = 4 / self.length_scale**2 * sin_of_arg**2 * K
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # length_scale is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
            # gradient with respect to p
            if not self.hyperparameter_periodicity.fixed:
                periodicity_gradient = (
                    4 * arg / self.length_scale**2 * cos_of_arg * sin_of_arg * K
                )
                periodicity_gradient = periodicity_gradient[:, :, np.newaxis]
            else:  # p is kept fixed
                periodicity_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((length_scale_gradient, periodicity_gradient))
        else:
            return K

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "ExpSineSquared(length_scale=..., periodicity=...)".
        return "{0}(length_scale={1:.3g}, periodicity={2:.3g})".format(
            self.__class__.__name__, self.length_scale, self.periodicity
        )


class DotProduct(Kernel):
    # [CLASS DOC]
    # Non-stationary linear kernel sigma_0^2 + x_i . x_j.
    r"""Dot-Product kernel.

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting :math:`N(0, 1)` priors on the coefficients
    of :math:`x_d (d = 1, . . . , D)` and a prior of :math:`N(0, \sigma_0^2)`
    on the bias. The DotProduct kernel is invariant to a rotation of
    the coordinates about the origin, but not translations.
    It is parameterized by a parameter sigma_0 :math:`\sigma`
    which controls the inhomogenity of the kernel. For :math:`\sigma_0^2 =0`,
    the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous. The kernel is given by

    .. math::
        k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.

    See [1]_, Chapter 4, Section 4.2, for further details regarding the
    DotProduct kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    sigma_0 : float >= 0, default=1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogeneous.

    sigma_0_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'sigma_0'.
        If set to "fixed", 'sigma_0' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0, 592.1]), array([316.6, 316.6]))
    """

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)):
        # [FUNCTION DOC]
        # Stores sigma_0 (default 1.0) and sigma_0_bounds (default (1e-5, 1e5)).
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    @property
    def hyperparameter_sigma_0(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for sigma_0.
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Computes the inner product matrix of X with Y (or with itself) plus sigma_0^2.
        # Outputs: K (n_X, n_Y); if eval_gradient, also gradient w.r.t. log sigma_0,
        # constant 2 sigma_0^2, shape (n_X, n_X, 1), or empty if fixed.
        # Exceptions: ValueError if eval_gradient with Y given.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            K = np.inner(X, X) + self.sigma_0**2
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            K = np.inner(X, Y) + self.sigma_0**2

        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                K_gradient = np.empty((K.shape[0], K.shape[1], 1))
                K_gradient[..., 0] = 2 * self.sigma_0**2
                return K, K_gradient
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        # [FUNCTION DOC]
        # Row-wise squared norms of X plus sigma_0^2. Outputs: (n_samples_X,).
        # Note: unlike __call__, X is not passed through atleast_2d, so X must already
        # be 2D.
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        return np.einsum("ij,ij->i", X, X) + self.sigma_0**2

    def is_stationary(self):
        # [FUNCTION DOC]
        # Always False.
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "DotProduct(sigma_0=...)".
        return "{0}(sigma_0={1:.3g})".format(self.__class__.__name__, self.sigma_0)


# adapted from scipy/optimize/optimize.py for functions with 2d output
def _approx_fprime(xk, f, epsilon, args=()):
    # [FUNCTION DOC]
    # HELPER: Forward finite-difference Jacobian for a function returning a 2D array
    # (adapted from scipy).
    # Inputs: xk - 1D point; f - callable taking xk plus args returning a 2D array;
    # epsilon - step size; args - extra positional arguments (tuple).
    # Outputs: ndarray (f0.shape[0], f0.shape[1], len(xk)) where slice k is (f(xk +
    # epsilon e_k) - f(xk)) / epsilon.
    # Side effects: calls f len(xk) + 1 times.
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


class PairwiseKernel(Kernel):
    # [CLASS DOC]
    # Wraps sklearn.metrics.pairwise kernels; only gamma is a hyperparameter, gradients
    # are computed numerically.
    """Wrapper for kernels in sklearn.metrics.pairwise.

    A thin wrapper around the functionality of the kernels in
    sklearn.metrics.pairwise.

    Note: Evaluation of eval_gradient is not analytic but numeric and all
          kernels support only isotropic distances. The parameter gamma is
          considered to be a hyperparameter and may be optimized. The other
          kernel parameters are set directly at initialization and are kept
          fixed.

    .. versionadded:: 0.18

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter gamma of the pairwise kernel specified by metric. It should
        be positive.

    gamma_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'gamma'.
        If set to "fixed", 'gamma' cannot be changed during
        hyperparameter tuning.

    metric : {"linear", "additive_chi2", "chi2", "poly", "polynomial", \
              "rbf", "laplacian", "sigmoid", "cosine"} or callable, \
              default="linear"
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    pairwise_kernels_kwargs : dict, default=None
        All entries of this dict (if any) are passed as keyword arguments to
        the pairwise kernel function.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import PairwiseKernel
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = PairwiseKernel(metric='rbf')
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8880, 0.05663, 0.05532],
           [0.8676, 0.07073, 0.06165]])
    """

    def __init__(
        self,
        gamma=1.0,
        gamma_bounds=(1e-5, 1e5),
        metric="linear",
        pairwise_kernels_kwargs=None,
    ):
        # [FUNCTION DOC]
        # Stores gamma (default 1.0), gamma_bounds, metric (default "linear") and
        # pairwise_kernels_kwargs (default None meaning no extra kwargs).
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        self.metric = metric
        self.pairwise_kernels_kwargs = pairwise_kernels_kwargs

    @property
    def hyperparameter_gamma(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for gamma.
        return Hyperparameter("gamma", "numeric", self.gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Evaluates pairwise_kernels(X, Y, metric, gamma, filter_params=True, plus extra
        # kwargs) after making X at least 2D.
        # Outputs: K (n_X, n_Y); if eval_gradient, also the gradient w.r.t. log gamma:
        # empty (n_X, n_X, 0) if fixed, otherwise a numerical gradient via
        # _approx_fprime (step 1e-10) of the same pairwise kernel evaluated at gamma =
        # exp(theta).
        # Exceptions: propagated from pairwise_kernels.
        # KNOWN ISSUE: unlike other kernels, eval_gradient with Y given does not raise;
        # the returned gradient then has shape (n_X, n_Y, 1) and the fixed case returns
        # (n_X, n_X, 0), which is inconsistent.
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        pairwise_kernels_kwargs = self.pairwise_kernels_kwargs
        if self.pairwise_kernels_kwargs is None:
            pairwise_kernels_kwargs = {}

        X = np.atleast_2d(X)
        K = pairwise_kernels(
            X,
            Y,
            metric=self.metric,
            gamma=self.gamma,
            filter_params=True,
            **pairwise_kernels_kwargs,
        )
        if eval_gradient:
            if self.hyperparameter_gamma.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # approximate gradient numerically
                def f(gamma):  # helper function
                    # [FUNCTION DOC]
                    # HELPER: Evaluates the pairwise kernel on the captured X, Y with
                    # gamma = exp(gamma) (input is log gamma). Outputs: kernel matrix.
                    return pairwise_kernels(
                        X,
                        Y,
                        metric=self.metric,
                        gamma=np.exp(gamma),
                        filter_params=True,
                        **pairwise_kernels_kwargs,
                    )

                return K, _approx_fprime(self.theta, f, 1e-10)
        else:
            return K

    def diag(self, X):
        # [FUNCTION DOC]
        # Slow diagonal: applies the full kernel to each row of X individually and
        # flattens the resulting 1x1 matrices. Outputs: (n_samples_X,).
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # We have to fall back to slow way of computing diagonal
        return np.apply_along_axis(self, 1, X).ravel()

    def is_stationary(self):
        # [FUNCTION DOC]
        # True only when metric is "rbf".
        """Returns whether the kernel is stationary."""
        return self.metric in ["rbf"]

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "PairwiseKernel(gamma=..., metric=...)".
        return "{0}(gamma={1}, metric={2})".format(
            self.__class__.__name__, self.gamma, self.metric
        )
