# Copyright (c) 2024, NVIDIA CORPORATION.
# Tell ruff it's OK that some imports occur after the sys.path.insert
# ruff: noqa: E402
import io
import os
import pathlib
import sys

import pyarrow as pa
import pytest

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.types import CompressionType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

from utils import (
    ALL_PA_TYPES,
    DEFAULT_PA_TYPES,
    NUMERIC_PA_TYPES,
    _get_vals_of_type,
)


# This fixture defines the standard set of types that all tests should default to
# running on. If there is a need for some tests to run on a different set of types, that
# type list fixture should also be defined below here if it is likely to be reused
# across modules. Otherwise it may be defined on a per-module basis.
@pytest.fixture(
    scope="session",
    params=DEFAULT_PA_TYPES,
)
def pa_type(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=NUMERIC_PA_TYPES,
)
def numeric_pa_type(request):
    return request.param


# TODO: Consider adding another fixture/adapting this
# fixture to consider nullability
@pytest.fixture(scope="session", params=[0, 100])
def table_data(request):
    """
    Returns (TableWithMetadata, pa_table).

    This is the default fixture you should be using for testing
    pylibcudf I/O writers.

    Contains one of each category (e.g. int, bool, list, struct)
    of dtypes.
    """
    nrows = request.param

    table_dict = {}
    # Colnames in the format expected by
    # plc.io.TableWithMetadata
    colnames = []

    seed = 42

    def _get_child_colnames(typ):
        child_colnames = []
        for i in range(typ.num_fields):
            grandchild_colnames = _get_child_colnames(typ.field(i).type)
            child_colnames.append((typ.field(i).name, grandchild_colnames))
        return child_colnames

    for typ in ALL_PA_TYPES:
        child_colnames = []
        if isinstance(typ, (pa.ListType, pa.StructType)):
            rand_arr = _get_vals_of_type(typ, nrows, seed=seed)
            child_colnames = _get_child_colnames(rand_arr.type)
        else:
            rand_arr = pa.array(
                _get_vals_of_type(typ, nrows, seed=seed), type=typ
            )

        table_dict[f"col_{typ}"] = rand_arr
        colnames.append((f"col_{typ}", child_colnames))

    pa_table = pa.Table.from_pydict(table_dict)

    return plc.io.TableWithMetadata(
        plc.interop.from_arrow(pa_table), column_names=colnames
    ), pa_table


@pytest.fixture(params=[(0, 0), ("half", 0), (-1, "half")])
def nrows_skiprows(table_data, request):
    """
    Parametrized nrows fixture that accompanies table_data
    """
    _, pa_table = table_data
    nrows, skiprows = request.param
    if nrows == "half":
        nrows = len(pa_table) // 2
    if skiprows == "half":
        skiprows = (len(pa_table) - nrows) // 2
    return nrows, skiprows


@pytest.fixture(
    params=["a.txt", pathlib.Path("a.txt"), io.BytesIO, io.StringIO],
)
def source_or_sink(request, tmp_path):
    fp_or_buf = request.param
    if isinstance(fp_or_buf, str):
        return f"{tmp_path}/{fp_or_buf}"
    elif isinstance(fp_or_buf, os.PathLike):
        return tmp_path.joinpath(fp_or_buf)
    elif issubclass(fp_or_buf, io.IOBase):
        # Must construct io.StringIO/io.BytesIO inside
        # fixture, or we'll end up re-using it
        return fp_or_buf()


@pytest.fixture(
    params=["a.txt", pathlib.Path("a.txt"), io.BytesIO],
)
def binary_source_or_sink(request, tmp_path):
    fp_or_buf = request.param
    if isinstance(fp_or_buf, str):
        return f"{tmp_path}/{fp_or_buf}"
    elif isinstance(fp_or_buf, os.PathLike):
        return tmp_path.joinpath(fp_or_buf)
    elif issubclass(fp_or_buf, io.IOBase):
        # Must construct io.StringIO/io.BytesIO inside
        # fixture, or we'll end up re-using it
        return fp_or_buf()


unsupported_types = {
    # Not supported by pandas
    # TODO: find a way to test these
    CompressionType.SNAPPY,
    CompressionType.BROTLI,
    CompressionType.LZ4,
    CompressionType.LZO,
    CompressionType.ZLIB,
}

unsupported_text_compression_types = unsupported_types.union(
    {
        # compressions not supported by libcudf
        # for csv/json
        CompressionType.XZ,
        CompressionType.ZSTD,
    }
)


@pytest.fixture(
    params=set(CompressionType).difference(unsupported_text_compression_types)
)
def text_compression_type(request):
    return request.param


@pytest.fixture(params=[opt for opt in plc.io.types.CompressionType])
def compression_type(request):
    return request.param


@pytest.fixture(
    scope="session", params=[opt for opt in plc.types.Interpolation]
)
def interp_opt(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[opt for opt in plc.types.Sorted],
)
def sorted_opt(request):
    return request.param


@pytest.fixture(
    scope="session", params=[False, True], ids=["without_nulls", "with_nulls"]
)
def has_nulls(request):
    return request.param


@pytest.fixture(
    scope="session", params=[False, True], ids=["without_nans", "with_nans"]
)
def has_nans(request):
    return request.param
