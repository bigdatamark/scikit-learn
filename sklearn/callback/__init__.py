# [PACKAGE DOC]
# Public entry point of the callback framework. Re-exports the callback protocols
# (FitCallback, AutoPropagatedCallback), the CallbackContext task tree, the estimator
# mixin CallbackSupportMixin and the with_callbacks fit decorator, and the built-in
# callbacks ProgressBar, ScoringMonitor and ScoringMonitorLog (listed in __all__).
# Import-time side effects: imports the private submodules only; no global state set.
"""
The :mod:`sklearn.callback` module implements the framework and off the shelf
callbacks for scikit-learn estimators.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.callback._base import AutoPropagatedCallback, FitCallback
from sklearn.callback._callback_context import CallbackContext
from sklearn.callback._callback_support import CallbackSupportMixin, with_callbacks
from sklearn.callback._progressbar import ProgressBar
from sklearn.callback._scoring_monitor import ScoringMonitor, ScoringMonitorLog

__all__ = [
    "AutoPropagatedCallback",
    "CallbackContext",
    "CallbackSupportMixin",
    "FitCallback",
    "ProgressBar",
    "ScoringMonitor",
    "ScoringMonitorLog",
    "with_callbacks",
]
