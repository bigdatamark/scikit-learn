# [PACKAGE DOC]
# Top-level sklearn package. Defines __version__, re-exports clone, get_config,
# set_config, config_context and show_versions, and lazily exposes the submodules in
# _submodules through module-level __getattr__/__dir__ (PEP 562), so subpackages
# are imported on first attribute access.
# Import-time side effects: creates the "sklearn" logger; sets the environment
# variables KMP_DUPLICATE_LIB_OK=True and KMP_INIT_AT_FORK=FALSE if unset; imports
# __check_build (raises a helpful ImportError if extensions are not built) and
# _distributor_init before anything linked to OpenMP.
"""Configure global settings and get information about the working environment."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Machine learning module for Python
# ==================================
#
# sklearn is a Python module integrating classical machine
# learning algorithms in the tightly-knit world of scientific Python
# packages (numpy, scipy, matplotlib).
#
# It aims to provide simple and efficient solutions to learning problems
# that are accessible to everybody and reusable in various contexts:
# machine-learning as a versatile tool for science and engineering.
#
# See https://scikit-learn.org for complete documentation.

import importlib as _importlib
import logging
import os
import random

from sklearn._config import config_context, get_config, set_config

logger = logging.getLogger(__name__)


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "1.9.1"


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subscription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# `_distributor_init` allows distributors to run custom init code.
# For instance, for the Windows wheel, this is used to pre-load the
# vcomp shared library runtime for OpenMP embedded in the sklearn/.libs
# sub-folder.
# It is necessary to do this prior to importing show_versions as the
# later is linked to the OpenMP runtime to make it possible to introspect
# it and importing it first would fail if the OpenMP dll cannot be found.
from sklearn import __check_build, _distributor_init  # noqa: E402 F401
from sklearn.base import clone  # noqa: E402
from sklearn.utils._show_versions import show_versions  # noqa: E402

_submodules = [
    "calibration",
    "callback",
    "cluster",
    "covariance",
    "cross_decomposition",
    "datasets",
    "decomposition",
    "dummy",
    "ensemble",
    "exceptions",
    "experimental",
    "externals",
    "feature_extraction",
    "feature_selection",
    "frozen",
    "gaussian_process",
    "inspection",
    "isotonic",
    "kernel_approximation",
    "kernel_ridge",
    "linear_model",
    "manifold",
    "metrics",
    "mixture",
    "model_selection",
    "multiclass",
    "multioutput",
    "naive_bayes",
    "neighbors",
    "neural_network",
    "pipeline",
    "preprocessing",
    "random_projection",
    "semi_supervised",
    "svm",
    "tree",
    "discriminant_analysis",
    "impute",
    "compose",
]

__all__ = _submodules + [
    # Non-modules:
    "clone",
    "get_config",
    "set_config",
    "config_context",
    "show_versions",
]


def __dir__():
    # [FUNCTION DOC]
    # Module-level __dir__ hook: returns __all__ (submodule names plus re-exported
    # functions) for dir(sklearn) and tab completion.
    return __all__


def __getattr__(name):
    # [FUNCTION DOC]
    # Module-level __getattr__ hook (called only for names not found normally): imports
    # and returns sklearn.<name> for names in _submodules, otherwise returns the module
    # global of that name.
    # Exceptions: AttributeError for unknown names.
    # Side effects: importing a submodule.
    # KNOWN ISSUE: the globals() fallback is effectively dead, since existing globals
    # never reach __getattr__.
    if name in _submodules:
        return _importlib.import_module(f"sklearn.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'sklearn' has no attribute '{name}'")


def setup_module(module):
    # [FUNCTION DOC]
    # Pytest module-level setup hook: seeds numpy's global RNG and Python's random
    # module with SKLEARN_SEED from the environment, or a random int32-range value if
    # unset, and prints the seed.
    # Inputs: module - the test module (unused).
    # Exceptions: ValueError if SKLEARN_SEED is not numeric.
    # Side effects: global RNG state, stdout.
    """Fixture for the tests to assure globally controllable seeding of RNGs"""

    import numpy as np

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("SKLEARN_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
