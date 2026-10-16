# [MODULE DOC]
# Gallery script implementing an inductive meta-estimator: cluster training data, then
#   train a classifier on the cluster labels so new samples can be assigned.
# Steps: 5000 samples in 3 blobs; AgglomerativeClustering (3 clusters) labels; 10 new
#   points from shifted blobs plotted in black;
#   InductiveClusterer(AgglomerativeClustering, RandomForestClassifier) fitted, predicts
#   clusters for new points, and DecisionBoundaryDisplay shows the induced regions.
# Side effects: one 3-panel figure, plt.show.
"""
====================
Inductive Clustering
====================

Clustering can be expensive, especially when our dataset contains millions
of datapoints. Many clustering algorithms are not :term:`inductive` and so
cannot be directly applied to new data samples without recomputing the
clustering, which may be intractable. Instead, we can use clustering to then
learn an inductive model with a classifier, which has several benefits:

- it allows the clusters to scale and apply to new data
- unlike re-fitting the clusters to new samples, it makes sure the labelling
  procedure is consistent over time
- it allows us to use the inferential capabilities of the classifier to
  describe or explain the clusters

This example illustrates a generic implementation of a meta-estimator which
extends clustering by inducing a classifier from the cluster labels.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

N_SAMPLES = 5000
RANDOM_STATE = 42


def _classifier_has(attr):
    # [FUNCTION DOC]
    # HELPER: Builds the predicate used by available_if to expose a method only when the
    #   classifier has it.
    # Inputs: attr - method name string.
    # Outputs: a callable taking the meta-estimator and returning True when the fitted
    #   classifier_ (if present) or else the unfitted classifier has attr.
    # Side effects: none.
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )


class InductiveClusterer(ClusterMixin, BaseEstimator):
    # [CLASS DOC]
    # Meta-estimator combining a (possibly transductive) clusterer with a classifier
    #   trained on its labels. Params: clusterer, classifier (unfitted estimators,
    #   cloned at fit). Fitted attributes: clusterer_, classifier_, labels_.
    # predict/decision_function exist only if the classifier provides them.
    def __init__(self, clusterer, classifier):
        # [FUNCTION DOC]
        # Stores clusterer and classifier unchanged (scikit-learn convention, no
        #   validation).
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Clones clusterer and classifier, runs fit_predict of the clusterer on X to get
        #   labels, fits the classifier on X with those labels.
        # Inputs: X - array (n_samples, n_features); y - ignored.
        # Outputs: self. Side effects: sets clusterer_, classifier_, labels_.
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        self.labels_ = y
        return self

    @available_if(_classifier_has("predict"))
    def predict(self, X):
        # [FUNCTION DOC]
        # Predicts cluster labels for new samples using the fitted classifier_.
        # Inputs: X - array (n_samples, n_features). Outputs: label array (n_samples,).
        # Exceptions: NotFittedError if fit was not called.
        check_is_fitted(self)
        return self.classifier_.predict(X)

    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        # [FUNCTION DOC]
        # Returns classifier_.decision_function(X) (shape depends on classifier).
        # Exceptions: NotFittedError if not fitted.
        # KNOWN ISSUE: with the RandomForestClassifier used in this script this method
        #   is never available.
        check_is_fitted(self)
        return self.classifier_.decision_function(X)


def plot_scatter(X, color, alpha=0.5):
    # [FUNCTION DOC]
    # HELPER: Scatter-plots the first two columns of X with given color(s), alpha and
    #   black edges.
    # Inputs: X - array (n, >=2); color - color or per-point values; alpha - float,
    #   default 0.5.
    # Outputs: the matplotlib PathCollection. Side effects: draws on current axes.
    return plt.scatter(X[:, 0], X[:, 1], c=color, alpha=alpha, edgecolor="k")


# Generate some training data from clustering
X, y = make_blobs(
    n_samples=N_SAMPLES,
    cluster_std=[1.0, 1.0, 0.5],
    centers=[(-5, -5), (0, 0), (5, 5)],
    random_state=RANDOM_STATE,
)


# Train a clustering algorithm on the training data and get the cluster labels
clusterer = AgglomerativeClustering(n_clusters=3)
cluster_labels = clusterer.fit_predict(X)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plot_scatter(X, cluster_labels)
plt.title("Ward Linkage")


# Generate new samples and plot them along with the original dataset
X_new, y_new = make_blobs(
    n_samples=10, centers=[(-7, -1), (-2, 4), (3, 6)], random_state=RANDOM_STATE
)

plt.subplot(132)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, "black", 1)
plt.title("Unknown instances")


# Declare the inductive learning model that it will be used to
# predict cluster membership for unknown instances
classifier = RandomForestClassifier(random_state=RANDOM_STATE)
inductive_learner = InductiveClusterer(clusterer, classifier).fit(X)

probable_clusters = inductive_learner.predict(X_new)


ax = plt.subplot(133)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, probable_clusters)

# Plotting decision regions
DecisionBoundaryDisplay.from_estimator(
    inductive_learner,
    X,
    response_method="predict",
    multiclass_colors="viridis",
    alpha=0.4,
    ax=ax,
)
plt.title("Classify unknown instances")

plt.show()
