# [PACKAGE DOC]
# Package grouping the opt-in "enable_*" modules for experimental scikit-learn features.
# Contents: enable_halving_search_cv and enable_iterative_imputer (which monkeypatch
# estimators into model_selection / impute when imported) and the deprecated no-op
# enable_hist_gradient_boosting. The package itself exposes nothing and has no import-
# time side effects; the side effects happen only when a submodule is imported.
"""Importable modules that enable the use of experimental features or estimators.

.. warning::

    The features and estimators that are experimental aren't subject to
    deprecation cycles. Use them at your own risks!
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
