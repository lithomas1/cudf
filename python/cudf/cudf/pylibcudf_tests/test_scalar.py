# Copyright (c) 2024, NVIDIA CORPORATION.
import numpy as np
import pyarrow as pa
import pytest
from utils import _get_vals_of_type, metadata_from_arrow_type

import cudf._lib.pylibcudf as plc


@pytest.mark.parametrize(
    "value",
    [10, 3.1415, np.float64(3.1415), np.int64(10)],
)
def test_scalar_constructor_python(value):
    plc_scalar = plc.Scalar(value)
    res = plc.interop.to_arrow(plc_scalar).as_py()
    assert res == value


def test_scalar_constructor_arrow(pa_type):
    pa_scalar = pa.array(
        _get_vals_of_type(pa_type, length=1, seed=42), type=pa_type
    )[0]
    plc_scalar = plc.Scalar(pa_scalar)
    assert (
        plc.interop.to_arrow(
            plc_scalar, metadata=metadata_from_arrow_type(pa_type)
        )
        == pa_scalar
    )
