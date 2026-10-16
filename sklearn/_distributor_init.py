# [MODULE DOC]
# Hook for redistributors (e.g. Windows wheels) to run custom initialisation code
# when sklearn is imported; imported by sklearn/__init__.py before OpenMP-linked
# modules. In the standard source distribution it contains no code: no contents
# and no side effects.
"""Distributor init file

Distributors: you can add custom code here to support particular distributions
of scikit-learn.

For example, this is a good place to put any checks for hardware requirements.

The scikit-learn standard source distribution will not put code in this file,
so you can safely replace this file with your own version.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
