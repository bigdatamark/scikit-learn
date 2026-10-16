# [MODULE DOC]
# Gallery script comparing ward, average, complete and single linkage
#   AgglomerativeClustering (10 clusters) on a 2D SpectralEmbedding of the digits
#   dataset.
# Steps: load digits, seed the global numpy RNG with 0, compute the 2D embedding, fit
#   each linkage, print fit time and plot via plot_clustering.
# Side effects: sets global numpy random seed, prints timings, creates one figure per
#   linkage, plt.show.
"""
=============================================================================
Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage option for agglomerative clustering on
a 2D embedding of the digits dataset.

The goal of this example is to show intuitively how the metrics behave, and
not to find good clusters for the digits. This is why the example works on a
2D embedding.

What this example shows us is the behavior "rich getting richer" of
agglomerative clustering that tends to create uneven cluster sizes.

This behavior is pronounced for the average linkage strategy,
that ends up with a couple of clusters with few datapoints.

The case of single linkage is even more pathologic with a very
large cluster covering most digits, an intermediate size (clean)
cluster with most zero digits and all other clusters being drawn
from noise points around the fringes.

The other linkage strategies lead to more evenly distributed
clusters that are therefore likely to be less sensible to a
random resampling of the dataset.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time

import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets, manifold

digits = datasets.load_digits()
X, y = digits.data, digits.target
n_samples, n_features = X.shape

np.random.seed(0)


# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    # [FUNCTION DOC]
    # Plots a 2D embedding where each point is drawn as its true digit glyph colored by
    #   its cluster label.
    # Inputs: X_red - array (n_samples, 2) embedding; labels - int array (n_samples,) of
    #   cluster ids in 0..9; title - optional figure title string.
    # Steps: min-max scale each embedding column to [0, 1]; open a new 6x4 figure; for
    #   each digit class, scatter the points of that true class (uses the module-level y
    #   and digits) with a text marker of the digit and a nipy_spectral color from label
    #   / 10.
    # Outputs: none. Side effects: creates a matplotlib figure; reads globals digits and
    #   y, so X_red must align with them.
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for digit in digits.target_names:
        plt.scatter(
            *X_red[y == digit].T,
            marker=f"${digit}$",
            s=50,
            c=plt.cm.nipy_spectral(labels[y == digit] / 10),
            alpha=0.5,
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ("ward", "average", "complete", "single"):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)


plt.show()
