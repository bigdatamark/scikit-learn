# [MODULE DOC]
# Tests for sklearn.feature_extraction.image: pixel graphs (img_to_graph,
# grid_to_graph) including masks and connectivity, 2-D patch extraction and
# reconstruction, PatchExtractor, and the strided _extract_patches helper. Uses the
# raccoon_face_fxt fixture from the shared conftest.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

from sklearn.feature_extraction.image import (
    PatchExtractor,
    _extract_patches,
    extract_patches_2d,
    grid_to_graph,
    img_to_graph,
    reconstruct_from_patches_2d,
)
from sklearn.utils._testing import assert_allclose


def test_img_to_graph():
    # [FUNCTION DOC]
    # Builds x and y coordinate grids (shifted negative) and checks their gradient
    # graphs have equal nnz and identical positive (gradient) entries, since unit
    # steps give gradient 1 along each axis; negative entries are the diagonal.
    x, y = np.mgrid[:4, :4] - 10
    grad_x = img_to_graph(x)
    grad_y = img_to_graph(y)
    assert grad_x.nnz == grad_y.nnz
    # Negative elements are the diagonal: the elements of the original
    # image. Positive elements are the values of the gradient, they
    # should all be equal on grad_x and grad_y
    np.testing.assert_array_equal(
        grad_x.data[grad_x.data > 0], grad_y.data[grad_y.data > 0]
    )


def test_img_to_graph_sparse():
    # [FUNCTION DOC]
    # With a mask containing an isolated pixel and a 2-pixel column, checks the
    # dense gradient graph equals the expected 3x3 matrix (diagonal = intensities,
    # off-diagonal = absolute difference), verifying masked vertex renumbering.
    # Check that the edges are in the right position
    #  when using a sparse image with a singleton component
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = 1
    mask[:, 2] = 1
    x = np.zeros((2, 3))
    x[0, 0] = 1
    x[0, 2] = -1
    x[1, 2] = -2
    grad_x = img_to_graph(x, mask=mask).todense()
    desired = np.array([[1, 0, 0], [0, -1, 1], [0, 1, -2]])
    np.testing.assert_array_equal(grad_x, desired)


def test_grid_to_graph():
    # [FUNCTION DOC]
    # Checks grid_to_graph: a mask with two isolated vertices (no edges) yields two
    # connected components; vertex ordering after masking matches an expected matrix;
    # an int16 mask works (one component); and the dtype argument (bool, int,
    # float64) controls the output dtype.
    # KNOWN ISSUE: the last two blocks pass a 2x2 mask for a 2x2x2 grid; it only works
    # because the mask is flattened and edges to vertices beyond its length are
    # dropped, so the test does not really exercise a full 3-D mask.
    # Checking that the function works with graphs containing no edges
    size = 2
    roi_size = 1
    # Generating two convex parts with one vertex
    # Thus, edges will be empty in _to_graph
    mask = np.zeros((size, size), dtype=bool)
    mask[0:roi_size, 0:roi_size] = True
    mask[-roi_size:, -roi_size:] = True
    mask = mask.reshape(size**2)
    A = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)
    assert connected_components(A)[0] == 2

    # check ordering
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = 1
    mask[:, 2] = 1
    graph = grid_to_graph(2, 3, 1, mask=mask.ravel()).todense()
    desired = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    np.testing.assert_array_equal(graph, desired)

    # Checking that the function works whatever the type of mask is
    mask = np.ones((size, size), dtype=np.int16)
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask)
    assert connected_components(A)[0] == 1

    # Checking dtype of the graph
    mask = np.ones((size, size))
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=bool)
    assert A.dtype == bool
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=int)
    assert A.dtype == int
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=np.float64)
    assert A.dtype == np.float64


def test_connect_regions(raccoon_face_fxt):
    # [FUNCTION DOC]
    # For two thresholds on a 4x subsampled face, checks the number of connected
    # components of img_to_graph with the threshold mask equals ndimage.label's count.
    face = raccoon_face_fxt
    # subsample by 4 to reduce run time
    face = face[::4, ::4]
    for thr in (50, 150):
        mask = face > thr
        graph = img_to_graph(face, mask=mask)
        assert ndimage.label(mask)[1] == connected_components(graph)[0]


def test_connect_regions_with_grid(raccoon_face_fxt):
    # [FUNCTION DOC]
    # Same component-count check using grid_to_graph on the face shape, once with the
    # default dtype and once with dtype=None.
    face = raccoon_face_fxt

    # subsample by 4 to reduce run time
    face = face[::4, ::4]

    mask = face > 50
    graph = grid_to_graph(*face.shape, mask=mask)
    assert ndimage.label(mask)[1] == connected_components(graph)[0]

    mask = face > 150
    graph = grid_to_graph(*face.shape, mask=mask, dtype=None)
    assert ndimage.label(mask)[1] == connected_components(graph)[0]


@pytest.fixture
def downsampled_face(raccoon_face_fxt):
    # [FUNCTION DOC]
    # HELPER: Fixture returning the raccoon face downsampled 4x per axis by summing
    # 2x2 blocks twice, converted to float32 and divided by 16 (block mean).
    face = raccoon_face_fxt
    face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
    face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
    face = face.astype(np.float32)
    face /= 16.0
    return face


@pytest.fixture
def orange_face(downsampled_face):
    # [FUNCTION DOC]
    # HELPER: Fixture building a 3-channel float64 image from the downsampled face,
    # each channel being 256 minus the face scaled by 1, 1/2 and 1/4.
    face = downsampled_face
    face_color = np.zeros(face.shape + (3,))
    face_color[:, :, 0] = 256 - face
    face_color[:, :, 1] = 256 - face / 2
    face_color[:, :, 2] = 256 - face / 4
    return face_color


def _make_images(face):
    # [FUNCTION DOC]
    # HELPER: Stacks three copies of face offset by 0, 1 and 2 into a float64 array
    # of shape (3,) + face.shape. Side effects: none.
    # make a collection of faces
    images = np.zeros((3,) + face.shape)
    images[0] = face
    images[1] = face + 1
    images[2] = face + 2
    return images


@pytest.fixture
def downsampled_face_collection(downsampled_face):
    # [FUNCTION DOC]
    # HELPER: Fixture returning _make_images applied to the downsampled face.
    return _make_images(downsampled_face)


def test_extract_patches_all(downsampled_face):
    # [FUNCTION DOC]
    # Checks exhaustive 16x16 patch extraction on a grayscale image returns all
    # (i_h - 15) * (i_w - 15) patches with the channel axis dropped.
    face = downsampled_face
    i_h, i_w = face.shape
    p_h, p_w = 16, 16
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)
    patches = extract_patches_2d(face, (p_h, p_w))
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_extract_patches_all_color(orange_face):
    # [FUNCTION DOC]
    # Same as test_extract_patches_all on the 3-channel image; the channel axis (3)
    # must be kept.
    face = orange_face
    i_h, i_w = face.shape[:2]
    p_h, p_w = 16, 16
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)
    patches = extract_patches_2d(face, (p_h, p_w))
    assert patches.shape == (expected_n_patches, p_h, p_w, 3)


def test_extract_patches_all_rect(downsampled_face):
    # [FUNCTION DOC]
    # Checks exhaustive extraction with a rectangular 16x12 patch on a cropped image
    # gives the expected count and shape.
    face = downsampled_face
    face = face[:, 32:97]
    i_h, i_w = face.shape
    p_h, p_w = 16, 12
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)

    patches = extract_patches_2d(face, (p_h, p_w))
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_extract_patches_max_patches(downsampled_face):
    # [FUNCTION DOC]
    # Checks max_patches=100 returns 100 patches, max_patches=0.5 returns the
    # truncated half of all patches, and 2.0 or -1.0 raise ValueError (parameter
    # validation).
    face = downsampled_face
    i_h, i_w = face.shape
    p_h, p_w = 16, 16

    patches = extract_patches_2d(face, (p_h, p_w), max_patches=100)
    assert patches.shape == (100, p_h, p_w)

    expected_n_patches = int(0.5 * (i_h - p_h + 1) * (i_w - p_w + 1))
    patches = extract_patches_2d(face, (p_h, p_w), max_patches=0.5)
    assert patches.shape == (expected_n_patches, p_h, p_w)

    with pytest.raises(ValueError):
        extract_patches_2d(face, (p_h, p_w), max_patches=2.0)
    with pytest.raises(ValueError):
        extract_patches_2d(face, (p_h, p_w), max_patches=-1.0)


def test_extract_patch_same_size_image(downsampled_face):
    # [FUNCTION DOC]
    # Checks a patch equal to the whole image with max_patches=2 yields exactly one
    # patch (max_patches clipped to the available count).
    face = downsampled_face
    # Request patches of the same size as image
    # Should return just the single patch a.k.a. the image
    patches = extract_patches_2d(face, face.shape, max_patches=2)
    assert patches.shape[0] == 1


def test_extract_patches_less_than_max_patches(downsampled_face):
    # [FUNCTION DOC]
    # Checks an integer max_patches above the available count returns all patches.
    face = downsampled_face
    i_h, i_w = face.shape
    p_h, p_w = 3 * i_h // 4, 3 * i_w // 4
    # this is 3185
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)

    patches = extract_patches_2d(face, (p_h, p_w), max_patches=4000)
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_reconstruct_patches_perfect(downsampled_face):
    # [FUNCTION DOC]
    # Checks exhaustive extraction followed by reconstruct_from_patches_2d recovers
    # the grayscale image to almost-equal precision.
    face = downsampled_face
    p_h, p_w = 16, 16

    patches = extract_patches_2d(face, (p_h, p_w))
    face_reconstructed = reconstruct_from_patches_2d(patches, face.shape)
    np.testing.assert_array_almost_equal(face, face_reconstructed)


def test_reconstruct_patches_perfect_color(orange_face):
    # [FUNCTION DOC]
    # Same perfect-reconstruction check for the 3-channel image.
    face = orange_face
    p_h, p_w = 16, 16

    patches = extract_patches_2d(face, (p_h, p_w))
    face_reconstructed = reconstruct_from_patches_2d(patches, face.shape)
    np.testing.assert_array_almost_equal(face, face_reconstructed)


@pytest.mark.parametrize(
    "image_size, patch_size",
    [
        ((128, 256), (128, 128)),  # patch_h == image_h
        ((256, 128), (128, 128)),  # patch_w == image_w
        ((128, 128), (128, 128)),  # patch == image
        ((128, 256, 3), (128, 128)),  # patch_h == image_h, with channels
    ],
)
def test_reconstruct_patches_edge_patch_size(image_size, patch_size):
    # [FUNCTION DOC]
    # Non-regression (#10910): random images where a patch dimension equals the image
    # dimension (with and without channels) are reconstructed exactly (assert_allclose).
    """Check that reconstruct_from_patches_2d works when a patch dimension
    equals the corresponding image dimension.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/10910
    """
    rng = np.random.RandomState(0)
    image = rng.rand(*image_size)
    patches = extract_patches_2d(image, patch_size)
    reconstructed = reconstruct_from_patches_2d(patches, image_size)
    assert_allclose(image, reconstructed)


def test_patch_extractor_fit(downsampled_face_collection, global_random_seed):
    # [FUNCTION DOC]
    # Checks PatchExtractor.fit returns the estimator itself.
    faces = downsampled_face_collection
    extr = PatchExtractor(
        patch_size=(8, 8), max_patches=100, random_state=global_random_seed
    )
    assert extr == extr.fit(faces)


def test_patch_extractor_max_patches(downsampled_face_collection, global_random_seed):
    # [FUNCTION DOC]
    # Checks PatchExtractor output count is n_images * max_patches for an int, and
    # n_images * truncated fraction of all patches for a float.
    faces = downsampled_face_collection
    i_h, i_w = faces.shape[1:3]
    p_h, p_w = 8, 8

    max_patches = 100
    expected_n_patches = len(faces) * max_patches
    extr = PatchExtractor(
        patch_size=(p_h, p_w), max_patches=max_patches, random_state=global_random_seed
    )
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w)

    max_patches = 0.5
    expected_n_patches = len(faces) * int(
        (i_h - p_h + 1) * (i_w - p_w + 1) * max_patches
    )
    extr = PatchExtractor(
        patch_size=(p_h, p_w), max_patches=max_patches, random_state=global_random_seed
    )
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_patch_extractor_max_patches_default(
    downsampled_face_collection, global_random_seed
):
    # [FUNCTION DOC]
    # Checks the default patch size (height // 10, width // 10) gives (19, 25) patches
    # for the downsampled face collection, with 100 patches per image.
    faces = downsampled_face_collection
    extr = PatchExtractor(max_patches=100, random_state=global_random_seed)
    patches = extr.transform(faces)
    assert patches.shape == (len(faces) * 100, 19, 25)


def test_patch_extractor_all_patches(downsampled_face_collection, global_random_seed):
    # [FUNCTION DOC]
    # Checks exhaustive PatchExtractor extraction count and shape on grayscale images.
    faces = downsampled_face_collection
    i_h, i_w = faces.shape[1:3]
    p_h, p_w = 8, 8
    expected_n_patches = len(faces) * (i_h - p_h + 1) * (i_w - p_w + 1)
    extr = PatchExtractor(patch_size=(p_h, p_w), random_state=global_random_seed)
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_patch_extractor_color(orange_face, global_random_seed):
    # [FUNCTION DOC]
    # Checks exhaustive PatchExtractor extraction on 3-channel images keeps the
    # channel axis.
    faces = _make_images(orange_face)
    i_h, i_w = faces.shape[1:3]
    p_h, p_w = 8, 8
    expected_n_patches = len(faces) * (i_h - p_h + 1) * (i_w - p_w + 1)
    extr = PatchExtractor(patch_size=(p_h, p_w), random_state=global_random_seed)
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w, 3)


def test_extract_patches_strided():
    # [FUNCTION DOC]
    # Table-driven test of _extract_patches on 1-D, 2-D and 3-D arange arrays with
    # various patch sizes and steps: the leading (position) dimensions of the view
    # must match the expected counts and the last patch must equal the slice of the
    # image starting at the expected corner.
    image_shapes_1D = [(10,), (10,), (11,), (10,)]
    patch_sizes_1D = [(1,), (2,), (3,), (8,)]
    patch_steps_1D = [(1,), (1,), (4,), (2,)]

    expected_views_1D = [(10,), (9,), (3,), (2,)]
    last_patch_1D = [(10,), (8,), (8,), (2,)]

    image_shapes_2D = [(10, 20), (10, 20), (10, 20), (11, 20)]
    patch_sizes_2D = [(2, 2), (10, 10), (10, 11), (6, 6)]
    patch_steps_2D = [(5, 5), (3, 10), (3, 4), (4, 2)]

    expected_views_2D = [(2, 4), (1, 2), (1, 3), (2, 8)]
    last_patch_2D = [(5, 15), (0, 10), (0, 8), (4, 14)]

    image_shapes_3D = [(5, 4, 3), (3, 3, 3), (7, 8, 9), (7, 8, 9)]
    patch_sizes_3D = [(2, 2, 3), (2, 2, 2), (1, 7, 3), (1, 3, 3)]
    patch_steps_3D = [(1, 2, 10), (1, 1, 1), (2, 1, 3), (3, 3, 4)]

    expected_views_3D = [(4, 2, 1), (2, 2, 2), (4, 2, 3), (3, 2, 2)]
    last_patch_3D = [(3, 2, 0), (1, 1, 1), (6, 1, 6), (6, 3, 4)]

    image_shapes = image_shapes_1D + image_shapes_2D + image_shapes_3D
    patch_sizes = patch_sizes_1D + patch_sizes_2D + patch_sizes_3D
    patch_steps = patch_steps_1D + patch_steps_2D + patch_steps_3D
    expected_views = expected_views_1D + expected_views_2D + expected_views_3D
    last_patches = last_patch_1D + last_patch_2D + last_patch_3D

    for image_shape, patch_size, patch_step, expected_view, last_patch in zip(
        image_shapes, patch_sizes, patch_steps, expected_views, last_patches
    ):
        image = np.arange(np.prod(image_shape)).reshape(image_shape)
        patches = _extract_patches(
            image, patch_shape=patch_size, extraction_step=patch_step
        )

        ndim = len(image_shape)

        assert patches.shape[:ndim] == expected_view
        last_patch_slices = tuple(
            slice(i, i + j, None) for i, j in zip(last_patch, patch_size)
        )
        assert (
            patches[(-1, None, None) * ndim] == image[last_patch_slices].squeeze()
        ).all()


def test_extract_patches_square(downsampled_face):
    # [FUNCTION DOC]
    # Checks an integer patch_shape is broadcast to all dimensions, giving a 4-D
    # view of shape (i_h - p + 1, i_w - p + 1, p, p).
    # test same patch size for all dimensions
    face = downsampled_face
    i_h, i_w = face.shape
    p = 8
    expected_n_patches = ((i_h - p + 1), (i_w - p + 1))
    patches = _extract_patches(face, patch_shape=p)
    assert patches.shape == (expected_n_patches[0], expected_n_patches[1], p, p)


def test_width_patch():
    # [FUNCTION DOC]
    # Checks extract_patches_2d raises ValueError when the patch is taller or wider
    # than the image.
    # width and height of the patch should be less than the image
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        extract_patches_2d(x, (4, 1))
    with pytest.raises(ValueError):
        extract_patches_2d(x, (1, 4))


def test_patch_extractor_wrong_input(orange_face):
    # [FUNCTION DOC]
    # Checks PatchExtractor.transform raises ValueError with an informative message
    # when patch_size has three elements.
    """Check that an informative error is raised if the patch_size is not valid."""
    faces = _make_images(orange_face)
    err_msg = "patch_size must be a tuple of two integers"
    extractor = PatchExtractor(patch_size=(8, 8, 8))
    with pytest.raises(ValueError, match=err_msg):
        extractor.transform(faces)
