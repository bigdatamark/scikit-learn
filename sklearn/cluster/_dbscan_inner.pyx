# [MODULE DOC]
# Compiled inner loop of DBSCAN. Given precomputed core-sample flags and
# radius neighborhoods, assigns cluster labels by expanding connected components
# of core samples. Used by sklearn.cluster.DBSCAN (_dbscan.py).
# Fast inner loop for DBSCAN.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libcpp.vector cimport vector

from sklearn.utils._typedefs cimport uint8_t, intp_t


def dbscan_inner(const uint8_t[::1] is_core,
                 object[:] neighborhoods,
                 intp_t[::1] labels):
    # [FUNCTION DOC]
    # Behavior: iterates samples in index order; each unlabeled core sample seeds a new
    # cluster id (0, 1, 2, ...). An iterative depth-first search with an explicit stack
    # labels every reachable sample: a popped unlabeled sample gets the current id, and
    # only if it is a core sample are its unlabeled neighbors pushed. Non-core (border)
    # samples are labeled but never expanded, so a border point reachable from several
    # clusters keeps the first id it receives. Samples never reached remain -1 (noise).
    # Inputs: is_core (uint8, n_samples) nonzero for core samples; neighborhoods
    # (object array, n_samples) each entry an intp array of neighbor indices;
    # labels (intp, n_samples) must be pre-filled with -1.
    # Outputs: none returned; labels is written in place with cluster ids.
    # Exceptions: errors only if a neighborhood entry is not convertible to an intp
    # memoryview or indices are out of range (bounds checking per build directives).
    # Side effects: mutates labels; allocates a temporary C++ vector stack; holds the GIL.
    cdef intp_t i, label_num = 0, v
    cdef intp_t[:] neighb
    cdef vector[intp_t] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for j in range(neighb.shape[0]):
                        v = neighb[j]
                        if labels[v] == -1:
                            stack.push_back(v)

            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()

        label_num += 1
