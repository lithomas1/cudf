# Copyright (c) 2023-2024, NVIDIA CORPORATION.
from cython cimport no_gc_clear
from cython.operator cimport dereference
from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from pyarrow cimport lib as pa

from rmm._lib.memory_resource cimport get_current_device_resource

from cudf._lib.pylibcudf.libcudf.interop cimport from_arrow as cpp_from_arrow
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport (
    fixed_point_scalar,
    numeric_scalar,
    scalar,
    string_scalar,
)
from cudf._lib.pylibcudf.libcudf.wrappers.decimals cimport (
    decimal32,
    decimal64,
    decimal128,
    scale_type,
)

from .types cimport DataType, type_id

ctypedef fused buf_scalar_t:
    int32_t
    int64_t
    float
    double


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the Scalar is in a reference cycle. Removing the tp_clear
# function with the no_gc_clear decoration prevents that. See
# https://github.com/rapidsai/rmm/pull/931 for details.
@no_gc_clear
cdef class Scalar:
    """A scalar value in device memory.

    This is the Cython representation of :cpp:class:`cudf::scalar`.
    """
    # Unlike for columns, libcudf does not support scalar views. All APIs that
    # accept scalar values accept references to the owning object rather than a
    # special view type. As a result, pylibcudf.Scalar has a simpler structure
    # than pylibcudf.Column because it can be a true wrapper around a libcudf
    # column

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

    def __init__(self, value, DataType dtype=None):
        # TODO: how to deal with overflow
        if isinstance(value, int):
            # Python int -> int64
            self.c_obj.reset(new numeric_scalar[int64_t](value, True))
        elif isinstance(value, float):
            # Python float (and numpy float) -> double
            self.c_obj.reset(new numeric_scalar[double](value, True))
        elif isinstance(value, str):
            self.c_obj.reset(new string_scalar(value.encode(), True))
        elif isinstance(value, pa.Scalar):
            self._init_from_arrow(value, dtype)
        # TODO: Better way to do this?
        elif hasattr(value, "__array__"):
            arr = value.__array__()
            # TODO: hack, cython doesn't recognize type of 0-D memoryviews
            self._init_from_buf_protocol(arr.reshape(1))
        else:
            # TODO: consider adding support for Python datetime/numpy
            # datetimes
            raise NotImplementedError(
                f"Don't know how to make Scalar with type {type(value)}"
            )

    def _init_from_buf_protocol(self, buf_scalar_t[:] arr):
        if buf_scalar_t is int32_t:
            self.c_obj.reset(new numeric_scalar[int32_t](arr[0], True))
        elif buf_scalar_t is int64_t:
            self.c_obj.reset(new numeric_scalar[int64_t](arr[0], True))
        elif buf_scalar_t is float:
            self.c_obj.reset(new numeric_scalar[float](arr[0], True))
        elif buf_scalar_t is double:
            self.c_obj.reset(new numeric_scalar[double](arr[0], True))

    cdef void _init_from_arrow(self, value, DataType dtype):
        cdef shared_ptr[pa.CScalar] arrow_scalar = pa.pyarrow_unwrap_scalar(value)

        cdef unique_ptr[scalar] c_result
        with nogil:
            c_result = move(cpp_from_arrow(dereference(arrow_scalar)))

        self.c_obj = move(c_result)
        self._data_type = DataType.from_libcudf(self.c_obj.get().type())

        # TODO: what is the point of this code
        # don't libcudf decimal types map 1-1 to arrow ones?
        if self._data_type.id() != type_id.DECIMAL128:
            if dtype is not None:
                raise ValueError(
                    "dtype may not be passed for non-decimal types"
                )
            return

        if dtype is None:
            raise ValueError(
                "Decimal scalars must be constructed with a dtype"
            )

        cdef type_id tid = dtype.id()

        if tid == type_id.DECIMAL32:
            self.c_obj.reset(
                new fixed_point_scalar[decimal32](
                    (
                        <fixed_point_scalar[decimal128]*> self.c_obj.get()
                    ).value(),
                    scale_type(-value.type.scale),
                    self.c_obj.get().is_valid()
                )
            )
        elif tid == type_id.DECIMAL64:
            self.c_obj.reset(
                new fixed_point_scalar[decimal64](
                    (
                        <fixed_point_scalar[decimal128]*> self.c_obj.get()
                    ).value(),
                    scale_type(-value.type.scale),
                    self.c_obj.get().is_valid()
                )
            )
        elif tid != type_id.DECIMAL128:
            raise ValueError(
                "Decimal scalars may only be cast to decimals"
            )
    cdef const scalar* get(self) noexcept nogil:
        return self.c_obj.get()

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef bool is_valid(self):
        """True if the scalar is valid, false if not"""
        return self.get().is_valid()

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=None):
        """Construct a Scalar object from a libcudf scalar.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_obj.swap(libcudf_scalar)
        s._data_type = DataType.from_libcudf(s.get().type())
        return s
