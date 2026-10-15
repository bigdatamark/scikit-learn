# [PACKAGE DOC]
# Package: sklearn.feature_extraction
# Cython component: _hashing_fast.transform, the hashing loop behind
# FeatureHasher (MurmurHash3 of feature names into n_features columns, optional
# alternating sign), returning CSR indices/indptr/data arrays. HashingVectorizer
# in feature_extraction.text uses it via FeatureHasher.
"""Feature extraction from raw data."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.feature_extraction import image, text
from sklearn.feature_extraction._dict_vectorizer import DictVectorizer
from sklearn.feature_extraction._hash import FeatureHasher
from sklearn.feature_extraction.image import grid_to_graph, img_to_graph

__all__ = [
    "DictVectorizer",
    "FeatureHasher",
    "grid_to_graph",
    "image",
    "img_to_graph",
    "text",
]
