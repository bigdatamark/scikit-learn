# [MODULE DOC]
# Shared infrastructure for ensemble estimators: BaseEnsemble (homogeneous ensembles
# built by cloning one base estimator), _BaseHeterogeneousEnsemble (ensembles of named,
# user-provided estimators such as voting/stacking), and helpers to fit an estimator in
# a job, seed nested random_state parameters, and split estimators across jobs.
"""Base class for ensemble-based estimators."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import effective_n_jobs

from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.utils import Bunch, check_random_state
from sklearn.utils._tags import get_tags
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import _routing_enabled
from sklearn.utils.metaestimators import _BaseComposition


def _fit_single_estimator(
    estimator, X, y, fit_params, message_clsname=None, message=None
):
    # [FUNCTION DOC]
    # HELPER: Fits one estimator, optionally printing elapsed time, and returns it.
    # Behavior: when metadata routing is disabled and fit_params contains sample_weight,
    # only sample_weight is passed to fit (other fit_params are dropped); otherwise all
    # fit_params are forwarded as keywords.
    # Inputs: estimator - unfitted estimator; X, y - training data; fit_params - dict;
    # message_clsname, message - strings for the elapsed-time printer (None disables it).
    # Outputs: the same estimator object, now fitted.
    # Exceptions: re-raises TypeError with a clearer message when the estimator's fit does
    # not accept sample_weight; other TypeErrors propagate unchanged.
    # Side effects: fits (mutates) estimator; may print timing to stdout.
    """Private function used to fit an estimator within a job."""
    # TODO(SLEP6): remove if-condition for unrouted sample_weight when metadata
    # routing can't be disabled.
    if not _routing_enabled() and "sample_weight" in fit_params:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=fit_params["sample_weight"])
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        estimator.__class__.__name__
                    )
                ) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y, **fit_params)
    return estimator


def _set_random_states(estimator, random_state=None):
    # [FUNCTION DOC]
    # HELPER: Assigns a fresh int32 seed, drawn from random_state, to every parameter named
    # random_state or ending with "__random_state" in the deep parameter list.
    # Keys are processed in sorted order so the seeds are deterministic for a given input.
    # Inputs: estimator - object with get_params/set_params; random_state - int,
    # RandomState or None.
    # Outputs: None. Side effects: mutates estimator via set_params; advances the RNG.
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for homogeneous ensembles. Subclasses must fill estimators_ during fit
    # using _make_estimator. Supports len(), indexing and iteration over estimators_.
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=10,
        estimator_params=tuple(),
    ):
        # [FUNCTION DOC]
        # Stores estimator, n_estimators and estimator_params unchanged (no validation, no
        # estimator instantiation). Abstract: subclasses must override.
        # Set parameters
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        # [FUNCTION DOC]
        # HELPER: Sets estimator_ to the user-supplied estimator, or to default when None.
        # Inputs: default - fallback estimator instance or None. Outputs: None.
        # Side effects: sets self.estimator_. No type validation is performed.
        """Check the base estimator.

        Sets the `estimator_` attributes.
        """
        if self.estimator is not None:
            self.estimator_ = self.estimator
        else:
            self.estimator_ = default

    def _make_estimator(self, append=True, random_state=None):
        # [FUNCTION DOC]
        # HELPER: Clones estimator_, sets on it the ensemble attributes named in
        # estimator_params, optionally reseeds its random_state parameters, and optionally
        # appends it to estimators_.
        # Inputs: append - bool, default True; random_state - int/RandomState/None.
        # Outputs: the new unfitted estimator.
        # Exceptions: AttributeError if estimators_ does not exist and append is True.
        # Side effects: may mutate self.estimators_ and advance random_state.
        """Make and configure a copy of the `estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        # [FUNCTION DOC]
        # Returns the number of fitted estimators in estimators_ (AttributeError if unfitted).
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        # [FUNCTION DOC]
        # Returns estimators_[index]; supports any index type the list accepts (int, slice).
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        # [FUNCTION DOC]
        # Returns an iterator over estimators_.
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)


def _partition_estimators(n_estimators, n_jobs):
    # [FUNCTION DOC]
    # HELPER: Splits n_estimators as evenly as possible across jobs.
    # Behavior: n_jobs is the effective job count capped at n_estimators; each job gets
    # the floor share, and the first (n_estimators mod n_jobs) jobs get one extra.
    # Inputs: n_estimators - int > 0; n_jobs - int or None (joblib semantics).
    # Outputs: tuple (n_jobs, list of per-job counts, list of n_jobs+1 cumulative start
    # offsets beginning at 0).
    # Exceptions: ZeroDivisionError if n_estimators is 0 (n_jobs becomes 0).
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class _BaseHeterogeneousEnsemble(
    MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta
):
    # [CLASS DOC]
    # Abstract base for ensembles of named heterogeneous estimators (list of (name,
    # estimator) pairs, entries may be "drop"). Provides validation, nested parameter
    # get/set via _BaseComposition and tag aggregation.
    """Base class for heterogeneous ensemble of learners.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    @property
    def named_estimators(self):
        # [FUNCTION DOC]
        # Property returning a Bunch mapping each name to its (unfitted, as given) estimator
        # from the estimators parameter, including "drop" entries.
        """Dictionary to access any fitted sub-estimators by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return Bunch(**dict(self.estimators))

    @abstractmethod
    def __init__(self, estimators):
        # [FUNCTION DOC]
        # Stores the estimators list unchanged. Abstract: subclasses must override.
        self.estimators = estimators

    def _validate_estimators(self):
        # [FUNCTION DOC]
        # HELPER: Validates the estimators parameter.
        # Checks: non-empty list of tuples/lists whose first item is a string; names are valid
        # and unique (via _validate_names); at least one entry is not "drop"; every non-drop
        # estimator is a classifier when self is a classifier, else a regressor.
        # Outputs: tuple (names tuple, estimators tuple).
        # Exceptions: ValueError for each failed check above.
        if len(self.estimators) == 0 or not all(
            isinstance(item, (tuple, list)) and isinstance(item[0], str)
            for item in self.estimators
        ):
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a "
                "non-empty list of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        has_estimator = any(est != "drop" for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        is_estimator_type = is_classifier if is_classifier(self) else is_regressor

        for est in estimators:
            if est != "drop" and not is_estimator_type(est):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        est.__class__.__name__, is_estimator_type.__name__[3:]
                    )
                )

        return names, estimators

    def set_params(self, **params):
        # [FUNCTION DOC]
        # Sets parameters, including replacing or dropping named sub-estimators, via
        # _BaseComposition._set_params on the "estimators" attribute.
        # Outputs: self. Side effects: mutates self and possibly sub-estimators.
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("estimators", **params)
        return self

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # Returns constructor parameters plus, when deep, each named sub-estimator and its
        # parameters prefixed by "<name>__".
        """
        Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("estimators", deep=deep)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags: allow_nan and sparse input tags are True only if every non-drop
        # sub-estimator supports them.
        # Exceptions: any error from malformed estimators is swallowed, leaving parent tags.
        # KNOWN ISSUE: the inline comment claims the tags are assumed False on failure, but the
        # code simply keeps whatever the parent tags hold.
        tags = super().__sklearn_tags__()
        try:
            tags.input_tags.allow_nan = all(
                get_tags(est[1]).input_tags.allow_nan if est[1] != "drop" else True
                for est in self.estimators
            )
            tags.input_tags.sparse = all(
                get_tags(est[1]).input_tags.sparse if est[1] != "drop" else True
                for est in self.estimators
            )
        except Exception:
            # If `estimators` does not comply with our API (list of tuples) then it will
            # fail. In this case, we assume that `allow_nan` and `sparse` are False but
            # the parameter validation will raise an error during `fit`.
            pass  # pragma: no cover
        return tags
