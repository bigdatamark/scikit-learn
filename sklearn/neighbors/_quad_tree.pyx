# [MODULE DOC]
# Implementation of _QuadTree, an array-backed quadtree (2D) / octree (3D)
# used by the Barnes-Hut approximation of t-SNE (sklearn.manifold). Cells are
# stored contiguously in a growable C array of Cell structs (declared in
# _quad_tree.pxd); children are referenced by integer cell ids, so the whole
# tree can be exposed as a NumPy structured array (CELL_DTYPE) for pickling.
# Module-level state: CELL_DTYPE (NumPy dtype mirroring Cell, built from a
# dummy Cell) and EPSILON (1e-6, tolerance for duplicate point detection).
# The fixed-size 3-element point buffers are used for both 2D and 3D; only the
# first n_dimensions entries are meaningful.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.math cimport fabsf
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdio cimport printf
from libc.stdint cimport SIZE_MAX

from sklearn.tree._utils cimport safe_realloc

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)

# Build the corresponding numpy dtype for Cell.
# This works by casting `dummy` to an array of Cell of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Cell dummy
CELL_DTYPE = np.asarray(<Cell[:1]>(&dummy)).dtype

assert CELL_DTYPE.itemsize == sizeof(Cell)

cdef const float EPSILON = 1e-6


cdef class _QuadTree:
    # [CLASS DOC]
    # Quadtree/octree over 2D or 3D float32 points. Each cell covers an
    # axis-aligned box [min_bounds, max_bounds) split at its center into
    # 2**n_dimensions children. Leaves hold at most one distinct point (duplicates
    # within EPSILON are merged by incrementing cumulative_size). Every cell keeps
    # the barycenter (center of mass) and count of the points in its subtree, which
    # summarize() uses for Barnes-Hut force approximation. Owns the malloc'ed
    # cells array (freed in __dealloc__).
    """Array-based representation of a QuadTree.

    This class is currently working for indexing 2D data (regular QuadTree) and
    for indexing 3D data (OcTree). It is planned to split the 2 implementations
    using `Cython.Tempita` to save some memory for QuadTree.

    Note that this code is currently internally used only by the Barnes-Hut
    method in `sklearn.manifold.TSNE`. It is planned to be refactored and
    generalized in the future to be compatible with nearest neighbors API of
    `sklearn.neighbors` with 2D and 3D data.
    """
    def __cinit__(self, int n_dimensions, int verbose):
        # [FUNCTION DOC]
        # Behavior: initializes parameters and an empty tree; n_cells_per_cell is
        # set to 2**n_dimensions. No cell memory is allocated yet (cells is NULL).
        # Inputs: n_dimensions (int, expected 2 or 3), verbose (int verbosity level).
        # Outputs: none. Exceptions: none beyond argument conversion errors.
        # Side effects: sets instance fields.
        """Constructor."""
        # Parameters of the tree
        self.n_dimensions = n_dimensions
        self.verbose = verbose
        self.n_cells_per_cell = <int> (2 ** self.n_dimensions)

        # Inner structures
        self.max_depth = 0
        self.cell_count = 0
        self.capacity = 0
        self.n_points = 0
        self.cells = NULL

    def __dealloc__(self):
        # [FUNCTION DOC]
        # Behavior: frees the cells array (free of NULL is safe).
        # Inputs/Outputs: none. Exceptions: none. Side effects: releases memory.
        """Destructor."""
        # Free all inner structures
        free(self.cells)

    @property
    def cumulative_size(self):
        # [FUNCTION DOC]
        # Behavior: read-only property returning the cumulative_size field (points
        # per subtree) of the first cell_count cells, as a NumPy view.
        # Inputs: none. Outputs: 1D intp array view backed by the tree memory.
        # Exceptions: ValueError if the array view cannot be created.
        # Side effects: the returned view holds a reference to the tree.
        cdef Cell[:] cell_mem_view = self._get_cell_ndarray()
        return cell_mem_view.base['cumulative_size'][:self.cell_count]

    @property
    def leafs(self):
        # [FUNCTION DOC]
        # Behavior: read-only property returning the is_leaf flag of the first
        # cell_count cells as a NumPy view.
        # Inputs: none. Outputs: 1D array view backed by the tree memory.
        # Exceptions: ValueError if the view cannot be created.
        # Side effects: the returned view holds a reference to the tree.
        cdef Cell[:] cell_mem_view = self._get_cell_ndarray()
        return cell_mem_view.base['is_leaf'][:self.cell_count]

    def build_tree(self, X):
        # [FUNCTION DOC]
        # Behavior: builds the tree from scratch. Allocates room for 100 cells,
        # computes per-axis min and max of X, inflates the max (relative 1e-3 and
        # absolute 1e-3, whichever is larger) so every point lies strictly inside the
        # half-open root box, initializes the root, inserts each row of X in order
        # (row index used as point_index), then shrinks the cells array to cell_count.
        # Inputs: X, 2D array-like of shape (n_samples, n_dimensions); no validation
        # is done (the check_array call is commented out); expected float32-compatible.
        # Outputs: none (tree is filled in place).
        # Exceptions: MemoryError on allocation failure; ValueError propagated from
        # insertion only when DEBUGFLAG checks are enabled.
        # Side effects: (re)allocates and mutates self.cells and counters; prints
        # bounding boxes when verbose > 10.
        # KNOWN ISSUE: assumes an empty tree; calling it twice reuses cell_count and
        # n_points from the previous build instead of resetting them.
        """Build a tree from an array of points X."""
        cdef:
            int i
            float32_t[3] pt
            float32_t[3] min_bounds, max_bounds

        # validate X and prepare for query
        # X = check_array(X, dtype=float32_t, order='C')
        n_samples = X.shape[0]

        capacity = 100
        self._resize(capacity)
        m = np.min(X, axis=0)
        M = np.max(X, axis=0)
        # Scale the maximum to get all points strictly in the tree bounding box
        # The 3 bounds are for positive, negative and small values
        M = np.maximum(M * (1. + 1e-3 * np.sign(M)), M + 1e-3)
        for i in range(self.n_dimensions):
            min_bounds[i] = m[i]
            max_bounds[i] = M[i]

            if self.verbose > 10:
                printf("[QuadTree] bounding box axis %i : [%f, %f]\n",
                       i, min_bounds[i], max_bounds[i])

        # Create the initial node with boundaries from the dataset
        self._init_root(min_bounds, max_bounds)

        for i in range(n_samples):
            for j in range(self.n_dimensions):
                pt[j] = X[i, j]
            self.insert_point(pt, i)

        # Shrink the cells array to reduce memory usage
        self._resize(capacity=self.cell_count)

    cdef int insert_point(self, float32_t[3] point, intp_t point_index,
                          intp_t cell_id=0) except -1 nogil:
        # [FUNCTION DOC]
        # Behavior: recursively inserts a point starting at cell_id. If the cell is
        # empty, the point is stored there. If it is an internal cell, its barycenter
        # is updated as a running weighted mean, its cumulative_size incremented, and
        # insertion recurses into the child selected by comparing the point to the
        # cell center per axis (creating a new child if that slot is empty). If it is
        # an occupied leaf: a duplicate (within EPSILON of the stored point) only
        # increments the count; otherwise the existing point is pushed down into a new
        # child (keeping its multiplicity) and insertion is retried on the now
        # internal cell.
        # Inputs: point (3 floats, first n_dimensions used, must lie in the cell box),
        # point_index (caller's id for the point), cell_id (start cell, default root).
        # Outputs: id of the cell that ended up holding the point (as int).
        # Exceptions: error return -1 with ValueError only from the DEBUGFLAG bounds
        # check; runs without the GIL.
        # Side effects: mutates cells, n_points, may grow the cells array; printf
        # debug output depending on verbose.
        """Insert a point in the QuadTree."""
        cdef int ax
        cdef intp_t selected_child
        cdef Cell* cell = &self.cells[cell_id]
        cdef intp_t n_point = cell.cumulative_size

        if self.verbose > 10:
            printf("[QuadTree] Inserting depth %li\n", cell.depth)

        # Assert that the point is in the right range
        if DEBUGFLAG:
            self._check_point_in_cell(point, cell)

        # If the cell is an empty leaf, insert the point in it
        if cell.cumulative_size == 0:
            cell.cumulative_size = 1
            self.n_points += 1
            for i in range(self.n_dimensions):
                cell.barycenter[i] = point[i]
            cell.point_index = point_index
            if self.verbose > 10:
                printf("[QuadTree] inserted point %li in cell %li\n",
                       point_index, cell_id)
            return cell_id

        # If the cell is not a leaf, update cell internals and
        # recurse in selected child
        if not cell.is_leaf:
            for ax in range(self.n_dimensions):
                # barycenter update using a weighted mean
                cell.barycenter[ax] = (
                    n_point * cell.barycenter[ax] + point[ax]) / (n_point + 1)

            # Increase the size of the subtree starting from this cell
            cell.cumulative_size += 1

            # Insert child in the correct subtree
            selected_child = self._select_child(point, cell)
            if self.verbose > 49:
                printf("[QuadTree] selected child %li\n", selected_child)
            if selected_child == -1:
                self.n_points += 1
                return self._insert_point_in_new_child(point, cell, point_index)
            return self.insert_point(point, point_index, selected_child)

        # Finally, if the cell is a leaf with a point already inserted,
        # split the cell in n_cells_per_cell if the point is not a duplicate.
        # If it is a duplicate, increase the size of the leaf and return.
        if self._is_duplicate(point, cell.barycenter):
            if self.verbose > 10:
                printf("[QuadTree] found a duplicate!\n")
            cell.cumulative_size += 1
            self.n_points += 1
            return cell_id

        # In a leaf, the barycenter correspond to the only point included
        # in it.
        self._insert_point_in_new_child(cell.barycenter, cell, cell.point_index,
                                        cell.cumulative_size)
        return self.insert_point(point, point_index, cell_id)

    # XXX: This operation is not Thread safe
    cdef intp_t _insert_point_in_new_child(
        self, float32_t[3] point, Cell* cell, intp_t point_index, intp_t size=1
    ) noexcept nogil:
        # [FUNCTION DOC]
        # HELPER: creates child cells for insert_point.
        # Behavior: allocates a new cell (doubling the cells array if full, after
        # saving the parent id and a copy of point since both may live in the
        # reallocated array), initializes it as a leaf at depth+1, marks the parent
        # as internal (point_index -1), computes the child's slot index (one bit per
        # axis, set when point >= parent center) and its bounds/center/squared max
        # width, stores point as its barycenter with point_index and cumulative_size
        # = size, and links it into the parent's children array.
        # Inputs: point (3 floats), cell (pointer to parent, may be invalidated by
        # reallocation and is reloaded), point_index, size (multiplicity, default 1).
        # Outputs: id of the new child cell.
        # Exceptions: declared noexcept; nogil.
        # Side effects: mutates/reallocates self.cells, increments cell_count.
        # Not thread safe.
        # KNOWN ISSUE: _resize may raise MemoryError, but this function is noexcept,
        # so the error is only printed and execution continues with an unenlarged
        # array (out-of-bounds write).
        """Create a child of cell which will contain point."""

        # Local variable definition
        cdef:
            intp_t cell_id, cell_child_id, parent_id
            float32_t[3] save_point
            float32_t width
            Cell* child
            int i

        # If the maximal capacity of the Tree have been reached, double the capacity
        # We need to save the current cell id and the current point to retrieve them
        # in case the reallocation
        if self.cell_count + 1 > self.capacity:
            parent_id = cell.cell_id
            for i in range(self.n_dimensions):
                save_point[i] = point[i]
            self._resize(SIZE_MAX)
            cell = &self.cells[parent_id]
            point = save_point

        # Get an empty cell and initialize it
        cell_id = self.cell_count
        self.cell_count += 1
        child = &self.cells[cell_id]

        self._init_cell(child, cell.cell_id, cell.depth + 1)
        child.cell_id = cell_id

        # Set the cell as an inner cell of the Tree
        cell.is_leaf = False
        cell.point_index = -1

        # Set the correct boundary for the cell, store the point in the cell
        # and compute its index in the children array.
        cell_child_id = 0
        for i in range(self.n_dimensions):
            cell_child_id *= 2
            if point[i] >= cell.center[i]:
                cell_child_id += 1
                child.min_bounds[i] = cell.center[i]
                child.max_bounds[i] = cell.max_bounds[i]
            else:
                child.min_bounds[i] = cell.min_bounds[i]
                child.max_bounds[i] = cell.center[i]
            child.center[i] = (child.min_bounds[i] + child.max_bounds[i]) / 2.
            width = child.max_bounds[i] - child.min_bounds[i]

            child.barycenter[i] = point[i]
            child.squared_max_width = max(child.squared_max_width, width*width)

        # Store the point info and the size to account for duplicated points
        child.point_index = point_index
        child.cumulative_size = size

        # Store the child cell in the correct place in children
        cell.children[cell_child_id] = child.cell_id

        if DEBUGFLAG:
            # Assert that the point is in the right range
            self._check_point_in_cell(point, child)
        if self.verbose > 10:
            printf("[QuadTree] inserted point %li in new child %li\n",
                   point_index, cell_id)

        return cell_id

    cdef bint _is_duplicate(self, float32_t[3] point1, float32_t[3] point2) noexcept nogil:
        # [FUNCTION DOC]
        # HELPER: duplicate detection for insert_point and _get_cell.
        # Behavior: true if the two points agree within EPSILON (absolute) on every
        # one of the first n_dimensions coordinates.
        # Inputs: point1, point2 (3-float buffers). Outputs: boolean.
        # Exceptions: none (noexcept, nogil). Side effects: none.
        """Check if the two given points are equals."""
        cdef int i
        cdef bint res = True
        for i in range(self.n_dimensions):
            # Use EPSILON to avoid numerical error that would overgrow the tree
            res &= fabsf(point1[i] - point2[i]) <= EPSILON
        return res

    cdef intp_t _select_child(self, float32_t[3] point, Cell* cell) noexcept nogil:
        # [FUNCTION DOC]
        # HELPER: child lookup for insert_point and _get_cell.
        # Behavior: computes the child slot index by building one bit per axis
        # (most significant bit first), set when point >= cell center on that axis,
        # and returns the id stored in that slot.
        # Inputs: point (3 floats), cell (pointer to an internal cell).
        # Outputs: child cell id, or -1 (SIZE_MAX cast) if that child does not exist.
        # Exceptions: none (noexcept, nogil). Side effects: none.
        """Select the child of cell which contains the given query point."""
        cdef:
            int i
            intp_t selected_child = 0

        for i in range(self.n_dimensions):
            # Select the correct child cell to insert the point by comparing
            # it to the borders of the cells using precomputed center.
            selected_child *= 2
            if point[i] >= cell.center[i]:
                selected_child += 1
        return cell.children[selected_child]

    cdef void _init_cell(self, Cell* cell, intp_t parent, intp_t depth) noexcept nogil:
        # [FUNCTION DOC]
        # HELPER: cell initialization for _init_root and _insert_point_in_new_child.
        # Behavior: sets parent and depth, marks the cell as an empty leaf
        # (cumulative_size 0, squared_max_width 0) and sets the first
        # n_cells_per_cell children ids to SIZE_MAX (i.e. -1, meaning absent).
        # Bounds, center, cell_id and point_index are left for the caller to set.
        # Inputs: cell pointer, parent id, depth. Outputs: none.
        # Exceptions: none. Side effects: writes into *cell.
        """Initialize a cell structure with some constants."""
        cell.parent = parent
        cell.is_leaf = True
        cell.depth = depth
        cell.squared_max_width = 0
        cell.cumulative_size = 0
        for i in range(self.n_cells_per_cell):
            cell.children[i] = SIZE_MAX

    cdef void _init_root(self, float32_t[3] min_bounds, float32_t[3] max_bounds
                         ) noexcept nogil:
        # [FUNCTION DOC]
        # HELPER: root creation for build_tree.
        # Behavior: initializes cell 0 as an empty leaf with no parent at depth 0,
        # sets its bounds, center (midpoint) and squared max width (largest squared
        # side length) from the given box, and increments cell_count.
        # Inputs: min_bounds, max_bounds (3 floats, first n_dimensions used).
        # Precondition: cells array allocated with capacity >= 1.
        # Outputs: none. Exceptions: none (noexcept). Side effects: mutates cells[0]
        # and cell_count.
        """Initialize the root node with the given space boundaries"""
        cdef:
            int i
            float32_t width
            Cell* root = &self.cells[0]

        self._init_cell(root, -1, 0)
        for i in range(self.n_dimensions):
            root.min_bounds[i] = min_bounds[i]
            root.max_bounds[i] = max_bounds[i]
            root.center[i] = (max_bounds[i] + min_bounds[i]) / 2.
            width = max_bounds[i] - min_bounds[i]
            root.squared_max_width = max(root.squared_max_width, width*width)
        root.cell_id = 0

        self.cell_count += 1

    cdef int _check_point_in_cell(self, float32_t[3] point, Cell* cell
                                  ) except -1 nogil:
        # [FUNCTION DOC]
        # HELPER: debug bounds check used by insertion and _check_coherence.
        # Behavior: optionally prints the point and cell box (verbose >= 50), then
        # checks min_bounds <= point < max_bounds on each axis.
        # Inputs: point (3 floats), cell pointer.
        # Outputs: 0 on success (implicit), -1 with exception set on failure.
        # Exceptions: ValueError (acquiring the GIL) if the point is outside the box.
        # Side effects: printf output when verbose.
        """Check that the given point is in the cell boundaries."""

        if self.verbose >= 50:
            if self.n_dimensions == 3:
                printf("[QuadTree] Checking point (%f, %f, %f) in cell %li "
                       "([%f/%f, %f/%f, %f/%f], size %li)\n",
                       point[0], point[1], point[2], cell.cell_id,
                       cell.min_bounds[0], cell.max_bounds[0], cell.min_bounds[1],
                       cell.max_bounds[1], cell.min_bounds[2], cell.max_bounds[2],
                       cell.cumulative_size)
            else:
                printf("[QuadTree] Checking point (%f, %f) in cell %li "
                       "([%f/%f, %f/%f], size %li)\n",
                       point[0], point[1], cell.cell_id, cell.min_bounds[0],
                       cell.max_bounds[0], cell.min_bounds[1],
                       cell.max_bounds[1], cell.cumulative_size)

        for i in range(self.n_dimensions):
            if (cell.min_bounds[i] > point[i] or
                    cell.max_bounds[i] <= point[i]):
                with gil:
                    msg = "[QuadTree] InsertionError: point out of cell "
                    msg += "boundary.\nAxis %li: cell [%f, %f]; point %f\n"

                    msg %= i, cell.min_bounds[i],  cell.max_bounds[i], point[i]
                    raise ValueError(msg)

    def _check_coherence(self):
        # [FUNCTION DOC]
        # Behavior: testing/validation routine. For each used cell, checks that its
        # barycenter lies in its box; for internal cells, checks that the sum of the
        # children's cumulative_size equals its own and that child cell_id fields are
        # consistent. Finally checks that n_points equals the root's cumulative_size.
        # Inputs: none. Outputs: none.
        # Exceptions: ValueError on incoherence; AssertionError on bad child ids.
        # Side effects: none (reads cells, which are copied to Python dicts).
        """Check the coherence of the cells of the tree.

        Check that the info stored in each cell is compatible with the info
        stored in descendent and sibling cells. Raise a ValueError if this
        fails.
        """
        for cell in self.cells[:self.cell_count]:
            # Check that the barycenter of inserted point is within the cell
            # boundaries
            self._check_point_in_cell(cell.barycenter, &cell)

            if not cell.is_leaf:
                # Compute the number of point in children and compare with
                # its cumulative_size.
                n_points = 0
                for idx in range(self.n_cells_per_cell):
                    child_id = cell.children[idx]
                    if child_id != -1:
                        child = self.cells[child_id]
                        n_points += child.cumulative_size
                        assert child.cell_id == child_id, (
                            "Cell id not correctly initialized.")
                if n_points != cell.cumulative_size:
                    raise ValueError(
                        "Cell {} is incoherent. Size={} but found {} points "
                        "in children. ({})"
                        .format(cell.cell_id, cell.cumulative_size,
                                n_points, cell.children))

        # Make sure that the number of point in the tree correspond to the
        # cumulative size in root cell.
        if self.n_points != self.cells[0].cumulative_size:
            raise ValueError(
                "QuadTree is incoherent. Size={} but found {} points "
                "in children."
                .format(self.n_points, self.cells[0].cumulative_size))

    cdef long summarize(self, float32_t[3] point, float32_t* results,
                        float squared_theta=.5, intp_t cell_id=0, long idx=0
                        ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: Barnes-Hut summary of the tree as seen from a query point,
        # performed by depth-first recursion from cell_id. For the current cell it
        # writes, starting at results[idx], the per-axis difference point minus cell
        # barycenter (n_dimensions values) followed by the squared euclidean distance.
        # A leaf that coincides (within EPSILON) with the query is skipped (no self
        # interaction). If the cell is a leaf or squared_max_width / squared distance
        # < squared_theta, the cell count is written after the distance and the entry
        # (n_dimensions + 2 floats) is kept; otherwise the entry is discarded and the
        # existing children are summarized recursively.
        # Inputs: point (3 floats), results (float buffer large enough for
        # n_points*(n_dimensions+2) values), squared_theta (theta squared), cell_id
        # and idx (recursion state; 0 for external calls).
        # Outputs: entries written to results; returns the new end index (number of
        # floats used).
        # Exceptions: none (noexcept, nogil). Side effects: writes results only.
        # KNOWN ISSUE: the docstring offsets are off by one: the squared distance is
        # at results[idx+n_dimensions] and the count at results[idx+n_dimensions+1].
        # KNOWN ISSUE: division by the squared distance is unguarded; an internal
        # cell whose barycenter equals the query gives inf and is then expanded.
        """Summarize the tree compared to a query point.

        Input arguments
        ---------------
        point : array (n_dimensions)
             query point to construct the summary.
        cell_id : integer, optional (default: 0)
            current cell of the tree summarized. This should be set to 0 for
            external calls.
        idx : integer, optional (default: 0)
            current index in the result array. This should be set to 0 for
            external calls
        squared_theta: float, optional (default: .5)
            threshold to decide whether the node is sufficiently far
            from the query point to be a good summary. The formula is such that
            the node is a summary if
                node_width^2 / dist_node_point^2 < squared_theta.
            Note that the argument should be passed as theta^2 to avoid
            computing square roots of the distances.

        Output arguments
        ----------------
        results : array (n_samples * (n_dimensions+2))
            result will contain a summary of the tree information compared to
            the query point:
            - results[idx:idx+n_dimensions] contains the coordinate-wise
                difference between the query point and the summary cell idx.
                This is useful in t-SNE to compute the negative forces.
            - result[idx+n_dimensions+1] contains the squared euclidean
                distance to the summary cell idx.
            - result[idx+n_dimensions+2] contains the number of point of the
                tree contained in the summary cell idx.

        Return
        ------
        idx : integer
            number of elements in the results array.
        """
        cdef:
            int i, idx_d = idx + self.n_dimensions
            bint duplicate = True
            Cell* cell = &self.cells[cell_id]

        results[idx_d] = 0.
        for i in range(self.n_dimensions):
            results[idx + i] = point[i] - cell.barycenter[i]
            results[idx_d] += results[idx + i] * results[idx + i]
            duplicate &= fabsf(results[idx + i]) <= EPSILON

        # Do not compute self interactions
        if duplicate and cell.is_leaf:
            return idx

        # Check whether we can use this node as a summary
        # It's a summary node if the angular size as measured from the point
        # is relatively small (w.r.t. theta) or if it is a leaf node.
        # If it can be summarized, we use the cell center of mass
        # Otherwise, we go a higher level of resolution and into the leaves.
        if cell.is_leaf or (
                (cell.squared_max_width / results[idx_d]) < squared_theta):
            results[idx_d + 1] = <float32_t> cell.cumulative_size
            return idx + self.n_dimensions + 2

        else:
            # Recursively compute the summary in nodes
            for c in range(self.n_cells_per_cell):
                child_id = cell.children[c]
                if child_id != -1:
                    idx = self.summarize(point, results, squared_theta,
                                         child_id, idx)

        return idx

    def get_cell(self, point):
        # [FUNCTION DOC]
        # Behavior: Python wrapper that copies the query point into a 3-float buffer
        # and returns _get_cell from the root.
        # Inputs: point, sequence of length n_dimensions.
        # Outputs: id of the leaf cell containing the point.
        # Exceptions: AssertionError on wrong length; ValueError if not in the tree.
        # Side effects: none.
        """return the id of the cell containing the query point or raise
        ValueError if the point is not in the tree
        """
        cdef float32_t[3] query_pt
        cdef int i

        assert len(point) == self.n_dimensions, (
            "Query point should be a point in dimension {}."
            .format(self.n_dimensions))

        for i in range(self.n_dimensions):
            query_pt[i] = point[i]

        return self._get_cell(query_pt, 0)

    cdef int _get_cell(self, float32_t[3] point, intp_t cell_id=0
                       ) except -1 nogil:
        # [FUNCTION DOC]
        # HELPER: implementation of get_cell.
        # Behavior: descends from cell_id: at a leaf, returns its id if its stored
        # point matches the query within EPSILON; at an internal cell, follows the
        # selected child if it exists.
        # Inputs: point (3 floats), cell_id (start cell, default 0).
        # Outputs: cell id; -1 with exception set on failure.
        # Exceptions: ValueError (with GIL acquired) if the point is not in the tree.
        # Side effects: printf output when verbose > 99.
        """guts of get_cell.

        Return the id of the cell containing the query point or raise ValueError
        if the point is not in the tree"""
        cdef:
            intp_t selected_child
            Cell* cell = &self.cells[cell_id]

        if cell.is_leaf:
            if self._is_duplicate(cell.barycenter, point):
                if self.verbose > 99:
                    printf("[QuadTree] Found point in cell: %li\n",
                           cell.cell_id)
                return cell_id
            with gil:
                raise ValueError("Query point not in the Tree.")

        selected_child = self._select_child(point, cell)
        if selected_child > 0:
            if self.verbose > 99:
                printf("[QuadTree] Selected_child: %li\n", selected_child)
            return self._get_cell(point, selected_child)
        with gil:
            raise ValueError("Query point not in the Tree.")

    # Pickling primitives

    def __reduce__(self):
        # [FUNCTION DOC]
        # Behavior: pickling support; returns the class, constructor arguments
        # (n_dimensions, verbose) and the state dictionary from __getstate__.
        # Inputs: none. Outputs: reduce tuple. Exceptions: as __getstate__.
        # Side effects: none.
        """Reduce re-implementation, for pickling."""
        return (_QuadTree, (self.n_dimensions, self.verbose), self.__getstate__())

    def __getstate__(self):
        # [FUNCTION DOC]
        # Behavior: returns a dict with max_depth, cell_count, capacity, n_points
        # and the cells as a NumPy structured array view (length cell_count).
        # Inputs: none. Outputs: dict.
        # Exceptions: ValueError if the array view cannot be created.
        # Side effects: the array view keeps a reference to the tree.
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["cell_count"] = self.cell_count
        d["capacity"] = self.capacity
        d["n_points"] = self.n_points
        d["cells"] = self._get_cell_ndarray().base
        return d

    def __setstate__(self, d):
        # [FUNCTION DOC]
        # Behavior: restores counters, validates that the cells array is 1D,
        # of CELL_DTYPE and C-contiguous, reallocates self.cells to its length and
        # copies the cell data into it.
        # Inputs: d, dict as produced by __getstate__.
        # Outputs: none.
        # Exceptions: ValueError for missing or malformed cells, MemoryError on
        # allocation failure.
        # Side effects: replaces counters and cell memory.
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.cell_count = d["cell_count"]
        self.capacity = d["capacity"]
        self.n_points = d["n_points"]

        if 'cells' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        cell_ndarray = d['cells']

        if (cell_ndarray.ndim != 1 or
                cell_ndarray.dtype != CELL_DTYPE or
                not cell_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = cell_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        cdef Cell[:] cell_mem_view = cell_ndarray
        memcpy(
            pto=self.cells,
            pfrom=&cell_mem_view[0],
            size=self.capacity * sizeof(Cell),
        )

    # Array manipulation methods, to convert it to numpy or to resize
    # self.cells array

    cdef Cell[:] _get_cell_ndarray(self):
        # [FUNCTION DOC]
        # HELPER: exposes the cells array to NumPy for properties and pickling.
        # Behavior: wraps the first cell_count cells (no copy) in a 1D NumPy array
        # of CELL_DTYPE and sets the tree as the array's base object so the memory
        # stays alive; returns a typed memoryview of it.
        # Inputs: none. Outputs: memoryview of Cell over the tree memory.
        # Exceptions: ValueError if setting the base object fails.
        # Side effects: increments references to CELL_DTYPE and to self.
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.cell_count
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Cell)
        cdef Cell[:] arr
        Py_INCREF(CELL_DTYPE)
        arr = PyArray_NewFromDescr(
            subtype=<PyTypeObject *> np.ndarray,
            descr=CELL_DTYPE,
            nd=1,
            dims=shape,
            strides=strides,
            data=<void*> self.cells,
            flags=cnp.NPY_ARRAY_DEFAULT,
            obj=None,
        )
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr.base, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array!")
        return arr

    cdef int _resize(self, intp_t capacity) except -1 nogil:
        # [FUNCTION DOC]
        # HELPER: raising wrapper around _resize_c.
        # Behavior: resizes the cells array to capacity (SIZE_MAX, equal to -1 as
        # intp, means double / initialize); acquires the GIL to raise on failure.
        # Inputs: capacity. Outputs: 0 on success, -1 with MemoryError set.
        # Exceptions: MemoryError. Side effects: reallocates self.cells.
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, intp_t capacity=SIZE_MAX) except -1 nogil:
        # [FUNCTION DOC]
        # HELPER: reallocation core used by _resize and __setstate__.
        # Behavior: no-op if capacity is unchanged and memory exists. SIZE_MAX means
        # grow: 9 cells initially, otherwise double. Reallocates cells, truncates
        # cell_count if it exceeds the new capacity, and records the capacity.
        # Inputs: capacity (default SIZE_MAX). Outputs: 0 on success.
        # Exceptions: safe_realloc raises MemoryError on overflow/failure (-1 return).
        # Side effects: reallocates self.cells; may change cell_count and capacity.
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.cells != NULL:
            return 0

        if <size_t> capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 9  # default initial value to min
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.cells, capacity)

        # if capacity smaller than cell_count, adjust the counter
        if capacity < self.cell_count:
            self.cell_count = capacity

        self.capacity = capacity
        return 0

    def _py_summarize(self, float32_t[:] query_pt, float32_t[:, :] X, float angle):
        # [FUNCTION DOC]
        # Behavior: test helper calling summarize from Python with theta = angle,
        # using a freshly allocated float32 result buffer of length 4 * n_samples.
        # Inputs: query_pt (float32 1D), X (float32 2D, only its row count is used),
        # angle (theta). Outputs: tuple (number of floats written, summary buffer).
        # Exceptions: none specific. Side effects: allocates the buffer.
        # KNOWN ISSUE: the buffer size 4 * n_samples assumes 2D (n_dimensions + 2
        # = 4); for a 3D tree summarize can write past the end of the buffer.
        # Used for testing summarize
        cdef:
            float32_t[:] summary
            int n_samples

        n_samples = X.shape[0]
        summary = np.empty(4 * n_samples, dtype=np.float32)

        idx = self.summarize(&query_pt[0], &summary[0], angle * angle)
        return idx, summary
