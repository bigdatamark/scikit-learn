# [MODULE DOC]
# Array API inspection namespace for CuPy: defines the __array_namespace_info__ class
# returned by the wrapped namespace's __array_namespace_info__ attribute. Imports the
# dtype constructors and cuda module from cupy at import time (bool_ is imported as
# bool, shadowing the builtin within this module).
"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
from cupy import (
    dtype,
    cuda,
    bool_ as bool,
    intp,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)


class __array_namespace_info__:
    # [CLASS DOC]
    # Stateless inspection object; instances expose capabilities, default_device,
    # default_dtypes, dtypes and devices. __module__ is set to "cupy" for display.
    """
    Get the array API inspection namespace for CuPy.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for CuPy.

    Examples
    --------
    >>> info = xp.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': cupy.float64,
     'complex floating': cupy.complex128,
     'integral': cupy.int64,
     'indexing': cupy.int64}

    """

    __module__ = 'cupy'

    def capabilities(self):
        # [FUNCTION DOC]
        # Returns a new dict of fixed capabilities: boolean indexing True, data-dependent shapes
        # True, max dimensions 64.
        # KNOWN ISSUE: the docstring list of keys omits "max dimensions".
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for CuPy.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          CuPy.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True,
         'max dimensions': 64}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            "max dimensions": 64,
        }

    def default_device(self):
        # [FUNCTION DOC]
        # Returns cupy.cuda.Device(0), the static default device (ignores the current-device
        # context, as documented).
        """
        The default device used for new CuPy arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : Device
            The default device used for new CuPy arrays.

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.default_device()
        Device(0)

        Notes
        -----
        This method returns the static default device when CuPy is initialized.
        However, the *current* device used by creation functions (``empty`` etc.)
        can be changed globally or with a context manager.

        See Also
        --------
        https://github.com/data-apis/array-api/issues/835
        """
        return cuda.Device(0)

    def default_dtypes(self, *, device=None):
        # [FUNCTION DOC]
        # Returns a new dict of default dtypes: float64, complex128, and intp for both
        # "integral" and "indexing". The device argument is accepted and ignored.
        """
        The default data types used for new CuPy arrays.

        For CuPy, this always returns the following dictionary:

        - **"real floating"**: ``cupy.float64``
        - **"complex floating"**: ``cupy.complex128``
        - **"integral"**: ``cupy.intp``
        - **"indexing"**: ``cupy.intp``

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new CuPy
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': cupy.float64,
         'complex floating': cupy.complex128,
         'integral': cupy.int64,
         'indexing': cupy.int64}

        """
        # TODO: Does this depend on device?
        return {
            "real floating": dtype(float64),
            "complex floating": dtype(complex128),
            "integral": dtype(intp),
            "indexing": dtype(intp),
        }

    def dtypes(self, *, device=None, kind=None):
        # [FUNCTION DOC]
        # Returns a dict mapping array-API dtype names to cupy.dtype objects for the requested
        # kind. kind None returns all 13 standard dtypes; a string selects one of bool, signed
        # integer, unsigned integer, integral, real floating, complex floating, numeric; a tuple
        # returns the union by recursing over each element. device is ignored.
        # Inputs: device - ignored; kind - None, str or tuple of str.
        # Outputs: new dict.
        # Exceptions: ValueError for an unknown kind string or non-str/tuple kind.
        # KNOWN ISSUE: for kind "bool" the value is the scalar type cupy.bool_ rather than
        # dtype(bool), inconsistent with every other branch (including kind None).
        """
        The array API data types supported by CuPy.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            CuPy data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': cupy.int8,
         'int16': cupy.int16,
         'int32': cupy.int32,
         'int64': cupy.int64}

        """
        # TODO: Does this depend on device?
        if kind is None:
            return {
                "bool": dtype(bool),
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
            }
        if kind == "unsigned integer":
            return {
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "integral":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "real floating":
            return {
                "float32": dtype(float32),
                "float64": dtype(float64),
            }
        if kind == "complex floating":
            return {
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "numeric":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self):
        # [FUNCTION DOC]
        # Returns a tuple of cupy.cuda.Device objects, one per GPU reported by
        # cuda.runtime.getDeviceCount().
        # Exceptions: CUDA runtime errors if no driver is available.
        # KNOWN ISSUE: the docstring says it returns a list, but a tuple is returned.
        """
        The devices supported by CuPy.

        Returns
        -------
        devices : list[Device]
            The devices supported by CuPy.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        """
        return tuple(cuda.Device(i) for i in range(cuda.runtime.getDeviceCount()))
