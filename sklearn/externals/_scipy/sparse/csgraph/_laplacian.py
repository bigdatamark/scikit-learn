# [MODULE DOC]
# Vendored copy of scipy.sparse.csgraph._laplacian (SciPy 1.12) so scikit-learn can
#   build graph Laplacians of sparse arrays with older SciPy versions. Provides the
#   public laplacian function plus private builders for dense/sparse inputs, either as
#   explicit matrices or as matrix-free callables / LinearOperators.
"""
This file is a copy of the scipy.sparse.csgraph._laplacian module from SciPy 1.12

scipy.sparse.csgraph.laplacian supports sparse arrays only starting from Scipy 1.12,
see https://github.com/scipy/scipy/pull/19156. This vendored file can be removed as
soon as Scipy 1.12 becomes the minimum supported version.

Laplacian of a compressed-sparse graph
"""

# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator


###############################################################################
# Graph laplacian
def laplacian(
    csgraph,
    normed=False,
    return_diag=False,
    use_out_degree=False,
    *,
    copy=True,
    form="array",
    dtype=None,
    symmetrized=False,
):
    # [FUNCTION DOC]
    # Dispatcher computing L = D - A (or the symmetrically normalized I - D^-1/2 A
    #   D^-1/2) of a square adjacency matrix.
    # Steps: validate that csgraph is 2-D and square; if normed and the dtype is a
    #   signed integer or numpy uint, cast to float64; choose a builder: for form
    #   "array" the sparse or dense explicit builder (by issparse), otherwise the
    #   matrix-free "flo" builder; degree axis is 1 for use_out_degree else 0 (column
    #   sums = in-degree); call the builder with all options.
    # Inputs: csgraph - (N, N) ndarray or scipy sparse matrix/array; normed,
    #   return_diag, use_out_degree - bools (default False); copy (default True), form
    #   ("array" | "function" | "lo"), dtype (None keeps input dtype), symmetrized
    #   (default False) are keyword-only.
    # Outputs: lap (same kind as input for "array", a callable for "function", a
    #   LinearOperator for "lo"); additionally the length-N array d when return_diag
    #   (degrees, or sqrt-degrees with 0 replaced by 1 when normed).
    # Exceptions: ValueError if not square/2-D, or (from builders) for an unknown form.
    # Side effects: with copy=False the input may be overwritten in place.
    # KNOWN ISSUE: np.issubdtype(dtype, np.uint) only matches the platform's unsigned
    #   long, so other unsigned integer dtypes (e.g. uint8) are not cast to float before
    #   normalization.
    """
    Return the Laplacian of a directed graph.

    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        Compressed-sparse graph, with shape (N, N).
    normed : bool, optional
        If True, then compute symmetrically normalized Laplacian.
        Default: False.
    return_diag : bool, optional
        If True, then also return an array related to vertex degrees.
        Default: False.
    use_out_degree : bool, optional
        If True, then use out-degree instead of in-degree.
        This distinction matters only if the graph is asymmetric.
        Default: False.
    copy : bool, optional
        If False, then change `csgraph` in place if possible,
        avoiding doubling the memory use.
        Default: True, for backward compatibility.
    form : 'array', or 'function', or 'lo'
        Determines the format of the output Laplacian:

        * 'array' is a numpy array;
        * 'function' is a pointer to evaluating the Laplacian-vector
          or Laplacian-matrix product;
        * 'lo' results in the format of the `LinearOperator`.

        Choosing 'function' or 'lo' always avoids doubling
        the memory use, ignoring `copy` value.
        Default: 'array', for backward compatibility.
    dtype : None or one of numeric numpy dtypes, optional
        The dtype of the output. If ``dtype=None``, the dtype of the
        output matches the dtype of the input csgraph, except for
        the case ``normed=True`` and integer-like csgraph, where
        the output dtype is 'float' allowing accurate normalization,
        but dramatically increasing the memory use.
        Default: None, for backward compatibility.
    symmetrized : bool, optional
        If True, then the output Laplacian is symmetric/Hermitian.
        The symmetrization is done by ``csgraph + csgraph.T.conj``
        without dividing by 2 to preserve integer dtypes if possible
        prior to the construction of the Laplacian.
        The symmetrization will increase the memory footprint of
        sparse matrices unless the sparsity pattern is symmetric or
        `form` is 'function' or 'lo'.
        Default: False, for backward compatibility.

    Returns
    -------
    lap : ndarray, or sparse matrix, or `LinearOperator`
        The N x N Laplacian of csgraph. It will be a NumPy array (dense)
        if the input was dense, or a sparse matrix otherwise, or
        the format of a function or `LinearOperator` if
        `form` equals 'function' or 'lo', respectively.
    diag : ndarray, optional
        The length-N main diagonal of the Laplacian matrix.
        For the normalized Laplacian, this is the array of square roots
        of vertex degrees or 1 if the degree is zero.

    Notes
    -----
    The Laplacian matrix of a graph is sometimes referred to as the
    "Kirchhoff matrix" or just the "Laplacian", and is useful in many
    parts of spectral graph theory.
    In particular, the eigen-decomposition of the Laplacian can give
    insight into many properties of the graph, e.g.,
    is commonly used for spectral data embedding and clustering.

    The constructed Laplacian doubles the memory use if ``copy=True`` and
    ``form="array"`` which is the default.
    Choosing ``copy=False`` has no effect unless ``form="array"``
    or the matrix is sparse in the ``coo`` format, or dense array, except
    for the integer input with ``normed=True`` that forces the float output.

    Sparse input is reformatted into ``coo`` if ``form="array"``,
    which is the default.

    If the input adjacency matrix is not symmetric, the Laplacian is
    also non-symmetric unless ``symmetrized=True`` is used.

    Diagonal entries of the input adjacency matrix are ignored and
    replaced with zeros for the purpose of normalization where ``normed=True``.
    The normalization uses the inverse square roots of row-sums of the input
    adjacency matrix, and thus may fail if the row-sums contain
    negative or complex with a non-zero imaginary part values.

    The normalization is symmetric, making the normalized Laplacian also
    symmetric if the input csgraph was symmetric.

    References
    ----------
    .. [1] Laplacian matrix. https://en.wikipedia.org/wiki/Laplacian_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csgraph

    Our first illustration is the symmetric graph

    >>> G = np.arange(4) * np.arange(4)[:, np.newaxis]
    >>> G
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]])

    and its symmetric Laplacian matrix

    >>> csgraph.laplacian(G)
    array([[ 0,  0,  0,  0],
           [ 0,  5, -2, -3],
           [ 0, -2,  8, -6],
           [ 0, -3, -6,  9]])

    The non-symmetric graph

    >>> G = np.arange(9).reshape(3, 3)
    >>> G
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    has different row- and column sums, resulting in two varieties
    of the Laplacian matrix, using an in-degree, which is the default

    >>> L_in_degree = csgraph.laplacian(G)
    >>> L_in_degree
    array([[ 9, -1, -2],
           [-3,  8, -5],
           [-6, -7,  7]])

    or alternatively an out-degree

    >>> L_out_degree = csgraph.laplacian(G, use_out_degree=True)
    >>> L_out_degree
    array([[ 3, -1, -2],
           [-3,  8, -5],
           [-6, -7, 13]])

    Constructing a symmetric Laplacian matrix, one can add the two as

    >>> L_in_degree + L_out_degree.T
    array([[ 12,  -4,  -8],
            [ -4,  16, -12],
            [ -8, -12,  20]])

    or use the ``symmetrized=True`` option

    >>> csgraph.laplacian(G, symmetrized=True)
    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    that is equivalent to symmetrizing the original graph

    >>> csgraph.laplacian(G + G.T)
    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    The goal of normalization is to make the non-zero diagonal entries
    of the Laplacian matrix to be all unit, also scaling off-diagonal
    entries correspondingly. The normalization can be done manually, e.g.,

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True)
    >>> L
    array([[ 2, -1, -1],
           [-1,  2, -1],
           [-1, -1,  2]])
    >>> d
    array([2, 2, 2])
    >>> scaling = np.sqrt(d)
    >>> scaling
    array([1.41421356, 1.41421356, 1.41421356])
    >>> (1/scaling)*L*(1/scaling)
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    Or using ``normed=True`` option

    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    >>> L
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    which now instead of the diagonal returns the scaling coefficients

    >>> d
    array([1.41421356, 1.41421356, 1.41421356])

    Zero scaling coefficients are substituted with 1s, where scaling
    has thus no effect, e.g.,

    >>> G = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    >>> G
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    >>> L
    array([[ 0., -0., -0.],
           [-0.,  1., -1.],
           [-0., -1.,  1.]])
    >>> d
    array([1., 1., 1.])

    Only the symmetric normalization is implemented, resulting
    in a symmetric Laplacian matrix if and only if its graph is symmetric
    and has all non-negative degrees, like in the examples above.

    The output Laplacian matrix is by default a dense array or a sparse matrix
    inferring its shape, format, and dtype from the input graph matrix:

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype(np.float32)
    >>> G
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]], dtype=float32)
    >>> csgraph.laplacian(G)
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]], dtype=float32)

    but can alternatively be generated matrix-free as a LinearOperator:

    >>> L = csgraph.laplacian(G, form="lo")
    >>> L
    <3x3 _CustomLinearOperator with dtype=float32>
    >>> L(np.eye(3))
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]])

    or as a lambda-function:

    >>> L = csgraph.laplacian(G, form="function")
    >>> L
    <function _laplace.<locals>.<lambda> at 0x0000012AE6F5A598>
    >>> L(np.eye(3))
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]])

    The Laplacian matrix is used for
    spectral data clustering and embedding
    as well as for spectral graph partitioning.
    Our final example illustrates the latter
    for a noisy directed linear graph.

    >>> from scipy.sparse import diags_array, random_array
    >>> from scipy.sparse.linalg import lobpcg

    Create a directed linear graph with ``N=35`` vertices
    using a sparse adjacency matrix ``G``:

    >>> N = 35
    >>> G = diags_array((np.ones(N-1), 1), format="csr")

    Fix a random seed ``rng`` and add a random sparse noise to the graph ``G``:

    >>> rng = np.random.default_rng()
    >>> G += 1e-2 * random_array((N, N), density=0.1, random_state=rng)

    Set initial approximations for eigenvectors:

    >>> X = rng.random((N, 2))

    The constant vector of ones is always a trivial eigenvector
    of the non-normalized Laplacian to be filtered out:

    >>> Y = np.ones((N, 1))

    Alternating (1) the sign of the graph weights allows determining
    labels for spectral max- and min- cuts in a single loop.
    Since the graph is undirected, the option ``symmetrized=True``
    must be used in the construction of the Laplacian.
    The option ``normed=True`` cannot be used in (2) for the negative weights
    here as the symmetric normalization evaluates square roots.
    The option ``form="lo"`` in (2) is matrix-free, i.e., guarantees
    a fixed memory footprint and read-only access to the graph.
    Calling the eigenvalue solver ``lobpcg`` (3) computes the Fiedler vector
    that determines the labels as the signs of its components in (5).
    Since the sign in an eigenvector is not deterministic and can flip,
    we fix the sign of the first component to be always +1 in (4).

    >>> for cut in ["max", "min"]:
    ...     G = -G  # 1.
    ...     L = csgraph.laplacian(G, symmetrized=True, form="lo")  # 2.
    ...     _, eves = lobpcg(L, X, Y=Y, largest=False, tol=1e-3)  # 3.
    ...     eves *= np.sign(eves[0, 0])  # 4.
    ...     print(cut + "-cut labels:\\n", 1 * (eves[:, 0]>0))  # 5.
    max-cut labels:
    [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]
    min-cut labels:
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    As anticipated for a (slightly noisy) linear graph,
    the max-cut strips all the edges of the graph coloring all
    odd vertices into one color and all even vertices into another one,
    while the balanced min-cut partitions the graph
    in the middle by deleting a single edge.
    Both determined partitions are optimal.
    """
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError("csgraph must be a square matrix or array")

    if normed and (
        np.issubdtype(csgraph.dtype, np.signedinteger)
        or np.issubdtype(csgraph.dtype, np.uint)
    ):
        csgraph = csgraph.astype(np.float64)

    if form == "array":
        create_lap = _laplacian_sparse if issparse(csgraph) else _laplacian_dense
    else:
        create_lap = (
            _laplacian_sparse_flo if issparse(csgraph) else _laplacian_dense_flo
        )

    degree_axis = 1 if use_out_degree else 0

    lap, d = create_lap(
        csgraph,
        normed=normed,
        axis=degree_axis,
        copy=copy,
        form=form,
        dtype=dtype,
        symmetrized=symmetrized,
    )
    if return_diag:
        return lap, d
    return lap


def _setdiag_dense(m, d):
    # [FUNCTION DOC]
    # HELPER: Writes values d onto the main diagonal of the square dense array m in
    #   place by assigning every (len(d)+1)-th element of its flattened view. Returns
    #   None.
    step = len(d) + 1
    m.flat[::step] = d


def _laplace(m, d):
    # [FUNCTION DOC]
    # HELPER: Builds the matrix-free Laplacian action. Inputs: m - adjacency (dense or
    #   sparse, N x N); d - length-N vector of row/column sums. Output: a function
    #   mapping v (N or N x k) to d times v row-wise minus m times v.
    return lambda v: v * d[:, np.newaxis] - m @ v


def _laplace_normed(m, d, nd):
    # [FUNCTION DOC]
    # HELPER: Normalized matrix-free Laplacian: returns a function that scales v row-
    #   wise by nd, applies the _laplace(m, d) action, and scales the result row-wise by
    #   nd again. nd is the inverse sqrt-degree vector.
    laplace = _laplace(m, d)
    return lambda v: nd[:, np.newaxis] * laplace(v * nd[:, np.newaxis])


def _laplace_sym(m, d):
    # [FUNCTION DOC]
    # HELPER: Matrix-free Laplacian of the symmetrized graph m + m^H without forming it:
    #   returns a function computing d times v row-wise minus m times v minus (v^H times
    #   m)^H, where d must already contain the symmetrized sums.
    return (
        lambda v: v * d[:, np.newaxis]
        - m @ v
        - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m))
    )


def _laplace_normed_sym(m, d, nd):
    # [FUNCTION DOC]
    # HELPER: Same as _laplace_normed but wrapping the symmetrized action of
    #   _laplace_sym.
    laplace_sym = _laplace_sym(m, d)
    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])


def _linearoperator(mv, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: Wraps a matvec/matmat callable mv into a scipy LinearOperator of the given
    #   shape and dtype (the same function serves both products).
    return LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)


def _laplacian_sparse_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    # [FUNCTION DOC]
    # HELPER: Matrix-free Laplacian builder for sparse inputs (form "function" or "lo").
    # Behavior: dtype defaults to the graph's dtype; graph_sum = sums along axis; diag =
    #   graph_sum minus the graph diagonal (self loops excluded); if symmetrized add the
    #   sums along the other axis to graph_sum and subtract the diagonal twice for diag.
    #   If normed: w = sqrt(diag), with isolated (zero-degree) nodes set to 1; the
    #   action is the normed (or normed symmetric) closure with 1/w. Otherwise the plain
    #   (or symmetric) closure with graph_sum. Wrap in a LinearOperator for "lo".
    # Inputs: graph - sparse (N, N); normed, axis (0/1), copy (ignored), form, dtype,
    #   symmetrized.
    # Outputs: (callable or LinearOperator, w or diag cast to dtype).
    # Exceptions: ValueError for any form other than "function"/"lo".
    # Side effects: none; the closures keep a reference to graph (read-only).
    # The keyword argument `copy` is unused and has no effect here.
    del copy

    if dtype is None:
        dtype = graph.dtype

    graph_sum = np.asarray(graph.sum(axis=axis)).ravel()
    graph_diagonal = graph.diagonal()
    diag = graph_sum - graph_diagonal
    if symmetrized:
        graph_sum += np.asarray(graph.sum(axis=1 - axis)).ravel()
        diag = graph_sum - graph_diagonal - graph_diagonal

    if normed:
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(graph, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(graph, graph_sum, 1.0 / w)
        if form == "function":
            return md, w.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, w.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")
    else:
        if symmetrized:
            md = _laplace_sym(graph, graph_sum)
        else:
            md = _laplace(graph, graph_sum)
        if form == "function":
            return md, diag.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, diag.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")


def _laplacian_sparse(graph, normed, axis, copy, form, dtype, symmetrized):
    # [FUNCTION DOC]
    # HELPER: Explicit sparse Laplacian builder (form "array").
    # Behavior: dtype defaults to graph dtype; lil/dok inputs are converted to COO (a
    #   copy), other formats are used directly and marked for copying when copy is True;
    #   if symmetrized add the conjugate transpose; w = sums along axis minus diagonal.
    #   If normed: convert to COO (copying if needed), replace zero w by 1 else take
    #   sqrt, divide each stored value by w of its row and of its column, negate, set
    #   the diagonal to 1 for non-isolated nodes and 0 for isolated ones. Else: DIA
    #   matrices are copied, others converted to COO (copy if needed); negate data and
    #   set the diagonal to w.
    # Inputs: as _laplacian_sparse_flo; form is ignored.
    # Outputs: (sparse Laplacian cast to dtype, w cast to dtype).
    # Side effects: with copy=False and COO input, the input's data is modified in
    #   place.
    # KNOWN ISSUE: for normed inputs the returned w are sqrt-degrees, while for non-
    #   normed they are degrees; consistent with upstream docs but easy to misread.
    #   Also, w.astype(dtype) always copies whereas the dense builder uses copy=False.
    # The keyword argument `form` is unused and has no effect here.
    del form

    if dtype is None:
        dtype = graph.dtype

    needs_copy = False
    if graph.format in ("lil", "dok"):
        m = graph.tocoo()
    else:
        m = graph
        if copy:
            needs_copy = True

    if symmetrized:
        m += m.T.conj()

    w = np.asarray(m.sum(axis=axis)).ravel() - m.diagonal()
    if normed:
        m = m.tocoo(copy=needs_copy)
        isolated_node_mask = w == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        m.setdiag(1 - isolated_node_mask)
    else:
        if m.format == "dia":
            m = m.copy()
        else:
            m = m.tocoo(copy=needs_copy)
        m.data *= -1
        m.setdiag(w)

    return m.astype(dtype, copy=False), w.astype(dtype)


def _laplacian_dense_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    # [FUNCTION DOC]
    # HELPER: Matrix-free Laplacian builder for dense inputs; identical algorithm to
    #   _laplacian_sparse_flo but first takes np.array (copy) or np.asarray of the graph
    #   depending on copy, and dtype defaults to that array's dtype.
    # Outputs: (callable or LinearOperator, w or diag cast to dtype). Exceptions:
    #   ValueError for invalid form.
    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    if dtype is None:
        dtype = m.dtype

    graph_sum = m.sum(axis=axis)
    graph_diagonal = m.diagonal()
    diag = graph_sum - graph_diagonal
    if symmetrized:
        graph_sum += m.sum(axis=1 - axis)
        diag = graph_sum - graph_diagonal - graph_diagonal

    if normed:
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(m, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(m, graph_sum, 1.0 / w)
        if form == "function":
            return md, w.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, w.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")
    else:
        if symmetrized:
            md = _laplace_sym(m, graph_sum)
        else:
            md = _laplace(m, graph_sum)
        if form == "function":
            return md, diag.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, diag.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")


def _laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized):
    # [FUNCTION DOC]
    # HELPER: Explicit dense Laplacian builder.
    # Behavior: require form "array"; dtype defaults to graph dtype; copy or view the
    #   graph; if symmetrized add its conjugate transpose in place; zero the diagonal; w
    #   = sums along axis. If normed: zero w replaced by 1 and others by sqrt, divide
    #   columns then rows by w, negate, and set the diagonal to 1 for non-isolated and 0
    #   for isolated nodes. Else negate and set the diagonal to w.
    # Outputs: (dense Laplacian cast to dtype, w cast to dtype).
    # Exceptions: ValueError if form is not "array".
    # Side effects: with copy=False the input array is overwritten (diagonal zeroed,
    #   values negated/scaled).
    # KNOWN ISSUE: the second "if dtype is None" check is dead code because dtype was
    #   already set from graph.dtype.
    if form != "array":
        raise ValueError(f'{form!r} must be "array"')

    if dtype is None:
        dtype = graph.dtype

    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    if dtype is None:
        dtype = m.dtype

    if symmetrized:
        m += m.T.conj()
    np.fill_diagonal(m, 0)
    w = m.sum(axis=axis)
    if normed:
        isolated_node_mask = w == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m /= w
        m /= w[:, np.newaxis]
        m *= -1
        _setdiag_dense(m, 1 - isolated_node_mask)
    else:
        m *= -1
        _setdiag_dense(m, w)

    return m.astype(dtype, copy=False), w.astype(dtype, copy=False)
