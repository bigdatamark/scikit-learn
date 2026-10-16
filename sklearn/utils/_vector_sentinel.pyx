# [MODULE DOC]
# Zero-copy conversion of C++ std::vector buffers to NumPy arrays. A
# StdVectorSentinel subclass per element type owns a vector (obtained by
# swapping with the caller's vector) and is set as the base object of a NumPy
# array built over the vector's data; when the array is garbage collected the
# sentinel and its vector are freed. Used by e.g. radius-neighbors pairwise
# distance reductions to return variable-length results.
from cython.operator cimport dereference as deref
from cpython.ref cimport Py_INCREF
cimport numpy as cnp

cnp.import_array()


cdef StdVectorSentinel _create_sentinel(vector_typed * vect_ptr):
    # [FUNCTION DOC]
    # HELPER: picks the sentinel subclass matching the fused vector type.
    # Behavior: call the matching create_for (Float64, Int32, Int64, or IntP as the
    # fallback). Inputs: vect_ptr, pointer to a vector whose contents are moved.
    # Outputs: a new sentinel owning the data; *vect_ptr is left empty.
    # Exceptions: MemoryError on object allocation.
    if vector_typed is vector[float64_t]:
        return StdVectorSentinelFloat64.create_for(vect_ptr)
    elif vector_typed is vector[int32_t]:
        return StdVectorSentinelInt32.create_for(vect_ptr)
    elif vector_typed is vector[int64_t]:
        return StdVectorSentinelInt64.create_for(vect_ptr)
    else:  # intp_t
        return StdVectorSentinelIntP.create_for(vect_ptr)


cdef class StdVectorSentinel:
    # [CLASS DOC]
    # Abstract base class: holds (in subclasses) a std::vector whose buffer backs a
    # NumPy array; deallocating the sentinel frees the vector's memory.
    """Wraps a reference to a vector which will be deallocated with this object.

    When created, the StdVectorSentinel swaps the reference of its internal
    vectors with the provided one (vect_ptr), thus making the StdVectorSentinel
    manage the provided one's lifetime.
    """
    cdef void* get_data(self):
        # [FUNCTION DOC]
        # Abstract: return a pointer to the owned vector's data (overridden). The base
        # implementation has no body and returns NULL.
        """Return pointer to data."""

    cdef int get_typenum(self):
        # [FUNCTION DOC]
        # Abstract: return the NumPy type number of the elements (overridden). The base
        # implementation returns 0.
        """Get typenum for PyArray_SimpleNewFromData."""


cdef class StdVectorSentinelFloat64(StdVectorSentinel):
    # [CLASS DOC]
    # Sentinel owning a vector of float64 values (NumPy typenum NPY_FLOAT64).
    cdef vector[float64_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[float64_t] * vect_ptr):
        # [FUNCTION DOC]
        # Behavior: allocate a sentinel without calling __init__ and swap its empty
        # vector with *vect_ptr, taking ownership of the buffer in O(1).
        # Inputs: vect_ptr (non-null). Outputs: the sentinel; *vect_ptr becomes empty.
        # Exceptions: MemoryError on allocation.
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelFloat64 sentinel = StdVectorSentinelFloat64.__new__(StdVectorSentinelFloat64)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel

    cdef void* get_data(self):
        # [FUNCTION DOC]
        # Return pointer to the owned vector's buffer (may be NULL if empty).
        return self.vec.data()

    cdef int get_typenum(self):
        # [FUNCTION DOC]
        # Return NPY_FLOAT64.
        return cnp.NPY_FLOAT64


cdef class StdVectorSentinelIntP(StdVectorSentinel):
    # [CLASS DOC]
    # Sentinel owning a vector of intp values (NumPy typenum NPY_INTP).
    cdef vector[intp_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[intp_t] * vect_ptr):
        # [FUNCTION DOC]
        # Same as the Float64 variant: take ownership of *vect_ptr by swapping; the
        # caller's vector becomes empty. Exceptions: MemoryError.
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelIntP sentinel = StdVectorSentinelIntP.__new__(StdVectorSentinelIntP)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel

    cdef void* get_data(self):
        # [FUNCTION DOC]
        # Return pointer to the owned vector's buffer.
        return self.vec.data()

    cdef int get_typenum(self):
        # [FUNCTION DOC]
        # Return NPY_INTP.
        return cnp.NPY_INTP


cdef class StdVectorSentinelInt32(StdVectorSentinel):
    # [CLASS DOC]
    # Sentinel owning a vector of int32 values (NumPy typenum NPY_INT32).
    cdef vector[int32_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[int32_t] * vect_ptr):
        # [FUNCTION DOC]
        # Same as the Float64 variant: take ownership of *vect_ptr by swapping; the
        # caller's vector becomes empty. Exceptions: MemoryError.
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelInt32 sentinel = StdVectorSentinelInt32.__new__(StdVectorSentinelInt32)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel

    cdef void* get_data(self):
        # [FUNCTION DOC]
        # Return pointer to the owned vector's buffer.
        return self.vec.data()

    cdef int get_typenum(self):
        # [FUNCTION DOC]
        # Return NPY_INT32.
        return cnp.NPY_INT32


cdef class StdVectorSentinelInt64(StdVectorSentinel):
    # [CLASS DOC]
    # Sentinel owning a vector of int64 values (NumPy typenum NPY_INT64).
    cdef vector[int64_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[int64_t] * vect_ptr):
        # [FUNCTION DOC]
        # Same as the Float64 variant: take ownership of *vect_ptr by swapping; the
        # caller's vector becomes empty. Exceptions: MemoryError.
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelInt64 sentinel = StdVectorSentinelInt64.__new__(StdVectorSentinelInt64)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel

    cdef void* get_data(self):
        # [FUNCTION DOC]
        # Return pointer to the owned vector's buffer.
        return self.vec.data()

    cdef int get_typenum(self):
        # [FUNCTION DOC]
        # Return NPY_INT64.
        return cnp.NPY_INT64


cdef cnp.ndarray vector_to_nd_array(vector_typed * vect_ptr):
    # [FUNCTION DOC]
    # Behavior: record the vector size, move its data into a new sentinel, create a
    # 1d NumPy array of that size and dtype wrapping the sentinel's buffer (no
    # copy), then make the sentinel the array's base object (incrementing its
    # refcount since PyArray_SetBaseObject steals a reference).
    # Inputs: vect_ptr, pointer to a vector of a supported element type.
    # Outputs: ndarray owning (through the sentinel) the former vector contents.
    # Side effects: *vect_ptr is emptied. Exceptions: MemoryError; errors from
    # PyArray_SimpleNewFromData or PyArray_SetBaseObject.
    cdef:
        cnp.npy_intp size = deref(vect_ptr).size()
        StdVectorSentinel sentinel = _create_sentinel(vect_ptr)
        cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
            1, &size, sentinel.get_typenum(), sentinel.get_data())

    # Makes the numpy array responsible of the life-cycle of its buffer.
    # A reference to the StdVectorSentinel will be stolen by the call to
    # `PyArray_SetBaseObject` below, so we increase its reference counter.
    # See: https://docs.python.org/3/c-api/intro.html#reference-count-details
    Py_INCREF(sentinel)
    cnp.PyArray_SetBaseObject(arr, sentinel)
    return arr
