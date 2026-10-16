# [MODULE DOC]
# Gallery script: trains SVC with a callable custom kernel on two iris features
# and plots the decision surface plus support vectors.
# Side effects: downloads/loads data, prints results and opens matplotlib figures at
# import time; this is a sphinx-gallery script, not a library.
"""
======================
SVM with custom kernel
======================

Simple usage of :ref:`svm` classifier with a custom kernel. It will
plot the decision surface and highlight the support vectors.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# Import some data to play with.
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset.
Y = iris.target


def my_kernel(X, Y):
    # [FUNCTION DOC]
    # HELPER: Gram matrix X M Y^T with M = diag(2, 1).
    # Inputs: X (n, 2), Y (m, 2). Outputs: array (n, m).
    # Exceptions: shape mismatch error if the inputs do not have 2 features.
    """
    We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)


# We create an instance of SVC with that kernel and fit it on the data.
clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)

ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    multiclass_colors="Paired",
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    alpha=0.5,
)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
# Highlight the support vectors
plt.scatter(
    X[clf.support_, 0],
    X[clf.support_, 1],
    facecolor="none",
    edgecolors="k",
)
plt.title("3-Class classification using Support Vector Machine with custom kernel")
plt.axis("tight")
plt.show()
