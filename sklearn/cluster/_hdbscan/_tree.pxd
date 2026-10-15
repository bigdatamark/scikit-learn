# [MODULE DOC]
# Shared declarations for the HDBSCAN tree code: the packed record types for a
# scipy-style single linkage hierarchy row (HIERARCHY_t) and a condensed tree edge
# (CONDENSED_t), cimported by _linkage.pyx and _tree.pyx, plus a declaration of
# the NumPy C-API shape accessor.
# Copyright (c) 2015, Leland McInnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from sklearn.utils._typedefs cimport intp_t, float64_t, uint8_t
cimport numpy as cnp

# This corresponds to the scipy.cluster.hierarchy format
ctypedef packed struct HIERARCHY_t:
    # [CLASS DOC]
    # One merge of a single linkage dendrogram (scipy.cluster.hierarchy layout):
    # left_node and right_node are the merged node ids (< n_samples are samples, larger
    # ids are earlier merges, id n_samples + row index), value the merge distance,
    # cluster_size the number of samples in the merged cluster. Matches HIERARCHY_dtype.
    intp_t left_node
    intp_t right_node
    float64_t value
    intp_t cluster_size

# Effectively an edgelist encoding a parent/child pair, along with a value and
# the corresponding cluster_size in each row providing a tree structure.
ctypedef packed struct CONDENSED_t:
    # [CLASS DOC]
    # One edge of the HDBSCAN condensed tree: parent cluster id, child id (a sample if
    # < n_samples, otherwise a cluster), value the lambda (1/distance) at which the
    # child separates from the parent, cluster_size the child's size (1 for samples).
    # Matches CONDENSED_dtype.
    intp_t parent
    intp_t child
    float64_t value
    intp_t cluster_size

cdef extern from "numpy/arrayobject.h":
    intp_t * PyArray_SHAPE(cnp.PyArrayObject *)
    # [FUNCTION DOC]
    # External NumPy C-API declaration: returns a pointer to the shape array of an
    # ndarray (used for fast length lookups without Python attribute access).
    # Cannot raise; caller must pass a valid array object.
