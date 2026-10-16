# [MODULE DOC]
# Image feature extraction utilities: building pixel/voxel adjacency graphs
# (img_to_graph weighted by intensity gradients, grid_to_graph unweighted
# connectivity) and extracting/reconstructing 2-D patches (extract_patches_2d,
# reconstruct_from_patches_2d, PatchExtractor). Voxels are numbered in C order
# over an (n_x, n_y, n_z) grid; graphs connect 6-neighbours (2-D: 4-neighbours).
"""Utilities to extract features from images."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import product
from numbers import Integral, Number, Real

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import _align_api_if_sparse, check_array, check_random_state
from sklearn.utils._param_validation import (
    Hidden,
    Interval,
    RealNotInt,
    validate_params,
)

__all__ = [
    "PatchExtractor",
    "extract_patches_2d",
    "grid_to_graph",
    "img_to_graph",
    "reconstruct_from_patches_2d",
]

from sklearn.utils.validation import validate_data

###############################################################################
# From an image to a graph


def _make_edges_3d(n_x, n_y, n_z=1):
    # [FUNCTION DOC]
    # HELPER: Builds the edge list of a 3-D grid lattice. Numbers voxels 0..N-1 in C
    # order on shape (n_x, n_y, n_z) and concatenates edges between each voxel and its
    # next neighbour along z, then y, then x.
    # Inputs: n_x, n_y, n_z - positive ints (n_z default 1).
    # Outputs: int array of shape (2, n_edges); row 0 source, row 1 target (source
    # index always smaller). Side effects: none.
    """Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x : int
        The size of the grid in the x direction.
    n_y : int
        The size of the grid in the y direction.
    n_z : integer, default=1
        The size of the grid in the z direction, defaults to 1
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_gradient_3d(edges, img):
    # [FUNCTION DOC]
    # HELPER: For each edge, decodes both endpoint voxel indices back to (x, y, z)
    # coordinates of img and returns the absolute difference of their intensities.
    # Inputs: edges - (2, n_edges) int array; img - 3-D array.
    # Outputs: 1-D array of length n_edges with img's dtype (unsigned dtypes may wrap
    # before abs). Side effects: none.
    _, n_y, n_z = img.shape
    gradient = np.abs(
        img[
            edges[0] // (n_y * n_z),
            (edges[0] % (n_y * n_z)) // n_z,
            (edges[0] % (n_y * n_z)) % n_z,
        ]
        - img[
            edges[1] // (n_y * n_z),
            (edges[1] % (n_y * n_z)) // n_z,
            (edges[1] % (n_y * n_z)) % n_z,
        ]
    )
    return gradient


# XXX: Why mask the image after computing the weights?


def _mask_edges_weights(mask, edges, weights=None):
    # [FUNCTION DOC]
    # HELPER: Keeps only edges whose two endpoints are both inside the boolean mask,
    # then renumbers vertices to consecutive indices among masked voxels (via
    # searchsorted on the flat nonzero positions of the mask).
    # Inputs: mask - boolean array over the grid; edges - (2, n_edges); weights -
    # optional per-edge array filtered alongside.
    # Outputs: renumbered edges, or (edges, weights) when weights is given.
    # Side effects: none. Assumes mask is boolean (indexes with mask.ravel()).
    """Apply a mask to edges (weighted or not)"""
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.isin(edges[0], inds), np.isin(edges[1], inds))
    edges = edges[:, ind_mask]
    if weights is not None:
        weights = weights[ind_mask]
    if len(edges.ravel()):
        maxval = edges.max()
    else:
        maxval = 0
    order = np.searchsorted(np.flatnonzero(mask), np.arange(maxval + 1))
    edges = order[edges]
    if weights is None:
        return edges
    else:
        return edges, weights


def _to_graph(
    n_x, n_y, n_z, mask=None, img=None, return_as=sparse.coo_array, dtype=None
):
    # [FUNCTION DOC]
    # HELPER: Common graph builder. Creates grid edges; dtype defaults to int (no
    # image) or img's dtype. With an image: weights are intensity gradients, the
    # diagonal holds voxel intensities (restricted to the mask if given). Without an
    # image: mask is cast to bool, edges are masked, weights and diagonal are ones.
    # Assembles a symmetric COO array of shape (n_voxels, n_voxels) containing each
    # edge in both directions plus the diagonal, then returns a dense ndarray if
    # return_as is np.ndarray, otherwise return_as applied to the COO array
    # (API-aligned).
    # Inputs: n_x, n_y, n_z - grid sizes; mask - optional; img - optional array;
    # return_as - class; dtype - optional.
    # Outputs: adjacency matrix (dense or sparse).
    # KNOWN ISSUE: in the image branch the mask is not cast to bool (unlike the
    # no-image branch) and the diagonal is taken with img.squeeze() indexed by mask,
    # so a non-boolean mask or a mask whose shape differs from the squeezed image
    # behaves incorrectly.
    """Auxiliary function for img_to_graph and grid_to_graph"""
    edges = _make_edges_3d(n_x, n_y, n_z)

    if dtype is None:  # To not overwrite input dtype
        if img is None:
            dtype = int
        else:
            dtype = img.dtype

    if img is not None:
        img = np.atleast_3d(img)
        weights = _compute_gradient_3d(edges, img)
        if mask is not None:
            edges, weights = _mask_edges_weights(mask, edges, weights)
            diag = img.squeeze()[mask]
        else:
            diag = img.ravel()
        n_voxels = diag.size
    else:
        if mask is not None:
            mask = mask.astype(dtype=bool, copy=False)
            edges = _mask_edges_weights(mask, edges)
            n_voxels = np.sum(mask)
        else:
            n_voxels = n_x * n_y * n_z
        weights = np.ones(edges.shape[1], dtype=dtype)
        diag = np.ones(n_voxels, dtype=dtype)

    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_array(
        (
            np.hstack((weights, weights, diag)),
            (np.hstack((i_idx, diag_idx)), np.hstack((j_idx, diag_idx))),
        ),
        (n_voxels, n_voxels),
        dtype=dtype,
    )
    if return_as is np.ndarray:
        return graph.toarray()
    return _align_api_if_sparse(return_as(graph))


@validate_params(
    {
        "img": ["array-like"],
        "mask": [None, np.ndarray],
        "return_as": [type],
        "dtype": "no_validation",  # validation delegated to numpy
    },
    prefer_skip_nested_validation=True,
)
def img_to_graph(img, *, mask=None, return_as=sparse.coo_array, dtype=None):
    # [FUNCTION DOC]
    # Gradient-weighted pixel graph. Promotes img to 3-D, takes its shape as the
    # grid and delegates to _to_graph with the image so edge weights are absolute
    # intensity differences and the diagonal holds intensities.
    # Inputs/Outputs: see docstring. Exceptions: InvalidParameterError from
    # validate_params for wrong argument types.
    """Graph of the pixel-to-pixel gradient connections.

    Edges are weighted with the gradient values.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    img : array-like of shape (height, width) or (height, width, channel)
        2D or 3D image.
    mask : ndarray of shape (height, width) or \
            (height, width, channel), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_array
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=None
        The data of the returned sparse matrix. By default it is the
        dtype of img.

    Returns
    -------
    graph : ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import img_to_graph
    >>> img = np.array([[0, 0], [0, 1]])
    >>> img_to_graph(img, return_as=np.ndarray)
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 1, 1, 1]])
    """
    img = np.atleast_3d(img)
    n_x, n_y, n_z = img.shape
    return _to_graph(n_x, n_y, n_z, mask, img, return_as, dtype)


@validate_params(
    {
        "n_x": [Interval(Integral, left=1, right=None, closed="left")],
        "n_y": [Interval(Integral, left=1, right=None, closed="left")],
        "n_z": [Interval(Integral, left=1, right=None, closed="left")],
        "mask": [None, np.ndarray],
        "return_as": [type],
        "dtype": "no_validation",  # validation delegated to numpy
    },
    prefer_skip_nested_validation=True,
)
def grid_to_graph(n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_array, dtype=int):
    # [FUNCTION DOC]
    # Unweighted grid connectivity graph: delegates to _to_graph without an image so
    # edges and diagonal entries are 1 (in dtype). Inputs/Outputs: see docstring.
    # Exceptions: InvalidParameterError for non-positive sizes or wrong types.
    """Graph of the pixel-to-pixel connections.

    Edges exist if 2 voxels are connected.

    Read more in the :ref:`User Guide <connectivity_graph_image>`.

    Parameters
    ----------
    n_x : int
        Dimension in x axis.
    n_y : int
        Dimension in y axis.
    n_z : int, default=1
        Dimension in z axis.
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_array
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int.

    Returns
    -------
    graph : np.ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import grid_to_graph
    >>> shape_img = (4, 4, 1)
    >>> mask = np.zeros(shape=shape_img, dtype=bool)
    >>> mask[[1, 2], [1, 2], :] = True
    >>> graph = grid_to_graph(*shape_img, mask=mask)
    >>> print(graph)
    <COOrdinate sparse matrix of dtype 'int64'
      with 2 stored elements and shape (2, 2)>
      Coords	Values
      (0, 0)    1
      (1, 1)    1
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as, dtype=dtype)


###############################################################################
# From an image to a set of small image patches


def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    # [FUNCTION DOC]
    # HELPER: Computes how many patches to extract. all_patches is
    # (i_h - p_h + 1) * (i_w - p_w + 1). If max_patches is falsy, returns all_patches;
    # an int returns min(max_patches, all_patches); a real in (0, 1) returns the
    # truncated fraction of all_patches.
    # Outputs: int. Exceptions: ValueError for any other truthy value (e.g. a
    # float >= 1 or a negative float).
    # KNOWN ISSUE: a negative integer max_patches is returned unchanged rather than
    # rejected (public callers guard against it via parameter validation).
    """Compute the number of patches that will be extracted in an image.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None, all possible patches are extracted.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if isinstance(max_patches, (Integral)) and max_patches < all_patches:
            return max_patches
        elif isinstance(max_patches, (Integral)) and max_patches >= all_patches:
            return all_patches
        elif isinstance(max_patches, (Real)) and 0 < max_patches < 1:
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _extract_patches(arr, patch_shape=8, extraction_step=1):
    # [FUNCTION DOC]
    # HELPER: Builds a zero-copy strided view of all patches of an n-D array. Scalars
    # for patch_shape/extraction_step are broadcast to every dimension. The view's
    # shape is (number of patch positions per dimension, computed as
    # (dim - patch) // step + 1) followed by patch_shape; its strides are the strides
    # of the step-sliced array followed by the original strides.
    # Outputs: 2n-D ndarray view sharing memory with arr (writes alias).
    # Side effects: none; no bounds checking (patch larger than arr yields invalid
    # shapes).
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : int or tuple of length arr.ndim.default=8
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : int or tuple of length arr.ndim, default=1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


@validate_params(
    {
        "image": [np.ndarray],
        "patch_size": [tuple, list],
        "max_patches": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def extract_patches_2d(image, patch_size, *, max_patches=None, random_state=None):
    # [FUNCTION DOC]
    # Extracts 2-D patches from one image. Validates patch fits the image, checks the
    # array (allow_nd), reshapes to (h, w, channels), builds a strided view of all
    # patches with full channel depth, computes n_patches; if max_patches is given,
    # draws n_patches random top-left positions independently with replacement from
    # the random state and gathers them, otherwise takes all positions in row-major
    # order. Copies into shape (n, p_h, p_w, channels) and drops the channel axis when
    # there is a single channel.
    # Outputs: new ndarray of patches (same dtype as image).
    # Exceptions: ValueError if patch height/width exceed the image's;
    # InvalidParameterError for invalid argument types/values.
    # Side effects: consumes random numbers from random_state when sampling.
    # KNOWN ISSUE: error messages say "should be less than" but patches equal to
    # the image size are accepted; random sampling may return duplicate patches.
    """Reshape a 2D image into a collection of patches.

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None it corresponds to the total number
        of patches that can be extracted.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_image
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the first image in this dataset:
    >>> one_image = load_sample_image("china.jpg")
    >>> print('Image shape: {}'.format(one_image.shape))
    Image shape: (427, 640, 3)
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print('Patches shape: {}'.format(patches.shape))
    Patches shape: (272214, 2, 2, 3)
    >>> # Here are just two of these patches:
    >>> print(patches[1])
    [[[174 201 231]
      [174 201 231]]
     [[173 200 230]
      [173 200 230]]]
    >>> print(patches[800])
    [[[187 214 243]
      [188 215 244]]
     [[187 214 243]
      [188 215 244]]]
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = _extract_patches(
        image, patch_shape=(p_h, p_w, n_colors), extraction_step=1
    )

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


@validate_params(
    {"patches": [np.ndarray], "image_size": [tuple, Hidden(list)]},
    prefer_skip_nested_validation=True,
)
def reconstruct_from_patches_2d(patches, image_size):
    # [FUNCTION DOC]
    # Inverse of exhaustive extract_patches_2d. Allocates a float64 zero image of
    # image_size, adds each patch at consecutive positions in row-major order over the
    # (n_h, n_w) grid of positions, then divides each pixel by the number of patches
    # that covered it, computed as the product over both axes of the minimum of
    # (index + 1, patch size, number of positions, size - index).
    # Outputs: float64 ndarray of shape image_size.
    # Exceptions: broadcasting errors if patch and image channel layouts mismatch.
    # Side effects: none. Extra patches beyond n_h * n_w are ignored by zip; fewer
    # patches leave regions under-filled but still divided.
    """Reconstruct the image from all of its patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patches : ndarray of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of int (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The size of the image that will be reconstructed.

    Returns
    -------
    image : ndarray of shape image_size
        The reconstructed image.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_image
    >>> from sklearn.feature_extraction import image
    >>> one_image = load_sample_image("china.jpg")
    >>> print('Image shape: {}'.format(one_image.shape))
    Image shape: (427, 640, 3)
    >>> image_patches = image.extract_patches_2d(image=one_image, patch_size=(10, 10))
    >>> print('Patches shape: {}'.format(image_patches.shape))
    Patches shape: (263758, 10, 10, 3)
    >>> image_reconstructed = image.reconstruct_from_patches_2d(
    ...     patches=image_patches,
    ...     image_size=one_image.shape
    ... )
    >>> print(f"Reconstructed shape: {image_reconstructed.shape}")
    Reconstructed shape: (427, 640, 3)
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i : i + p_h, j : j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(
                min(i + 1, p_h, n_h, i_h - i) * min(j + 1, p_w, n_w, i_w - j)
            )
    return img


class PatchExtractor(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Stateless transformer applying extract_patches_2d to every image of a batch;
    # validation happens only in fit (via _fit_context) and data validation in
    # transform with reset=False.
    """Extracts patches from a collection of images.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    patch_size : tuple of int (patch_height, patch_width), default=None
        The dimensions of one patch. If set to None, the patch size will be
        automatically set to `(img_height // 10, img_width // 10)`, where
        `img_height` and `img_width` are the dimensions of the input images.

    max_patches : int or float, default=None
        The maximum number of patches per image to extract. If `max_patches` is
        a float in (0, 1), it is taken to mean a proportion of the total number
        of patches. If set to None, extract all possible patches.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches is not None`. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    See Also
    --------
    reconstruct_from_patches_2d : Reconstruct image from all of its patches.

    Notes
    -----
    This estimator is stateless and does not need to be fitted. However, we
    recommend to call :meth:`fit_transform` instead of :meth:`transform`, as
    parameter validation is only performed in :meth:`fit`.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_images
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the second image in this dataset:
    >>> X = load_sample_images().images[1]
    >>> X = X[None, ...]
    >>> print(f"Image shape: {X.shape}")
    Image shape: (1, 427, 640, 3)
    >>> pe = image.PatchExtractor(patch_size=(10, 10))
    >>> pe_trans = pe.transform(X)
    >>> print(f"Patches shape: {pe_trans.shape}")
    Patches shape: (263758, 10, 10, 3)
    >>> X_reconstructed = image.reconstruct_from_patches_2d(pe_trans, X.shape[1:])
    >>> print(f"Reconstructed shape: {X_reconstructed.shape}")
    Reconstructed shape: (427, 640, 3)
    """

    _parameter_constraints: dict = {
        "patch_size": [tuple, None],
        "max_patches": [
            None,
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "random_state": ["random_state"],
    }

    def __init__(self, *, patch_size=None, max_patches=None, random_state=None):
        # [FUNCTION DOC]
        # Stores patch_size, max_patches and random_state unchanged.
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # No-op apart from parameter validation by the _fit_context decorator.
        # Inputs: X, y - ignored. Outputs: self.
        """Only validate the parameters of the estimator.

        This method allows to: (i) validate the parameters of the estimator  and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : ndarray of shape (n_samples, image_height, image_width) or \
                (n_samples, image_height, image_width, n_channels)
            Array of images from which to extract patches. For color images,
            the last dimension specifies the channel: a RGB image would have
            `n_channels=3`.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # Validates X (n-D allowed, at least one sample/feature, no reset of
        # n_features_in_), resolves the random state once, picks patch_size (default
        # (height // 10, width // 10), else must have length 2), reshapes images to
        # (n, h, w, channels), and fills a preallocated float64 array of shape
        # (n_imgs * n_patches, p_h, p_w[, channels]) with each image's patches from
        # extract_patches_2d (sharing the same RandomState so each image gets different
        # random positions).
        # Outputs: float64 ndarray of patches (channel axis only when channels > 1).
        # Exceptions: ValueError if patch_size length is not 2, or from
        # extract_patches_2d/validation.
        # Side effects: none on self.
        # KNOWN ISSUE: the default patch size can be 0 for images smaller than 10 pixels
        # in a dimension; n_imgs/img_height/img_width are computed twice redundantly; the
        # output is always float64 regardless of input dtype.
        """Transform the image samples in `X` into a matrix of patch data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, image_height, image_width) or \
                (n_samples, image_height, image_width, n_channels)
            Array of images from which to extract patches. For color images,
            the last dimension specifies the channel: a RGB image would have
            `n_channels=3`.

        Returns
        -------
        patches : array of shape (n_patches, patch_height, patch_width) or \
                (n_patches, patch_height, patch_width, n_channels)
            The collection of patches extracted from the images, where
            `n_patches` is either `n_samples * max_patches` or the total
            number of patches that can be extracted.
        """
        X = validate_data(
            self,
            X=X,
            ensure_2d=False,
            allow_nd=True,
            ensure_min_samples=1,
            ensure_min_features=1,
            reset=False,
        )
        random_state = check_random_state(self.random_state)
        n_imgs, img_height, img_width = X.shape[:3]
        if self.patch_size is None:
            patch_size = img_height // 10, img_width // 10
        else:
            if len(self.patch_size) != 2:
                raise ValueError(
                    "patch_size must be a tuple of two integers. Got"
                    f" {self.patch_size} instead."
                )
            patch_size = self.patch_size

        n_imgs, img_height, img_width = X.shape[:3]
        X = np.reshape(X, (n_imgs, img_height, img_width, -1))
        n_channels = X.shape[-1]

        # compute the dimensions of the patches array
        patch_height, patch_width = patch_size
        n_patches = _compute_n_patches(
            img_height, img_width, patch_height, patch_width, self.max_patches
        )
        patches_shape = (n_imgs * n_patches,) + patch_size
        if n_channels > 1:
            patches_shape += (n_channels,)

        # extract the patches
        patches = np.empty(patches_shape)
        for ii, image in enumerate(X):
            patches[ii * n_patches : (ii + 1) * n_patches] = extract_patches_2d(
                image,
                patch_size,
                max_patches=self.max_patches,
                random_state=random_state,
            )
        return patches

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Tags: accepts 3-D arrays, not 2-D arrays; requires_fit is False.
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        tags.requires_fit = False
        return tags
