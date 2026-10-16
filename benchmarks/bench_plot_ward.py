# [MODULE DOC]
# Top-level benchmark script (no functions) comparing AgglomerativeClustering
# (ward, 3 clusters) fit time against scipy.cluster.hierarchy.ward on Gaussian
# data over a log-spaced grid of n_samples (about 3..1000) and n_features (10..3162).
# Shows log of the time ratio as an image with the ratio=1 contour.
# Side effects: unseeded randomness, matplotlib window.
# KNOWN ISSUE: grid sizes are floats from np.logspace and are passed directly as
# the size of np.random.normal, which raises TypeError in modern NumPy.
"""
Benchmark scikit-learn's Ward implement compared to SciPy's
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

from sklearn.cluster import AgglomerativeClustering

ward = AgglomerativeClustering(n_clusters=3, linkage="ward")

n_samples = np.logspace(0.5, 3, 9)
n_features = np.logspace(1, 3.5, 7)
N_samples, N_features = np.meshgrid(n_samples, n_features)
scikits_time = np.zeros(N_samples.shape)
scipy_time = np.zeros(N_samples.shape)

for i, n in enumerate(n_samples):
    for j, p in enumerate(n_features):
        X = np.random.normal(size=(n, p))
        t0 = time.time()
        ward.fit(X)
        scikits_time[j, i] = time.time() - t0
        t0 = time.time()
        hierarchy.ward(X)
        scipy_time[j, i] = time.time() - t0

ratio = scikits_time / scipy_time

plt.figure("scikit-learn Ward's method benchmark results")
plt.imshow(np.log(ratio), aspect="auto", origin="lower")
plt.colorbar()
plt.contour(
    ratio,
    levels=[
        1,
    ],
    colors="k",
)
plt.yticks(range(len(n_features)), n_features.astype(int))
plt.ylabel("N features")
plt.xticks(range(len(n_samples)), n_samples.astype(int))
plt.xlabel("N samples")
plt.title("Scikit's time, in units of scipy time (log)")
plt.show()
