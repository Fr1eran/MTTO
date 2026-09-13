from __future__ import annotations

import numpy as np
import pytest

from utils.type_utils import as_1d_float_array, as_float


def test_as_float_numeric_types():
    assert as_float(0) == 0.0
    assert as_float(42) == 42.0
    assert as_float(3.14) == 3.14
    assert as_float(np.int32(10)) == 10.0
    assert as_float(np.int64(-5)) == -5.0
    assert as_float(np.float32(1.5)) == pytest.approx(1.5)
    assert as_float(np.float64(2.718)) == pytest.approx(2.718)


def test_as_float_non_numeric_returns_none():
    assert as_float(None) is None
    assert as_float("123") is None
    assert as_float("abc") is None
    assert as_float([1.0]) is None
    assert as_float({"a": 1}) is None
    assert as_float((1, 2)) is None


def test_as_1d_float_array_valid():
    arr = as_1d_float_array([1, 2, 3], "test_arr")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    # Numpy array input
    np_arr = np.array([4.0, 5.0])
    arr2 = as_1d_float_array(np_arr)
    assert arr2.shape == (2,)

    # Swapped parameter tolerance (name, values)
    arr3 = as_1d_float_array("my_name", [7.0, 8.0])
    assert arr3.shape == (2,)


def test_as_1d_float_array_rejects_non_1d():
    with pytest.raises(ValueError, match="test_2d must be a 1-D array"):
        as_1d_float_array([[1.0, 2.0], [3.0, 4.0]], "test_2d")

    with pytest.raises(ValueError, match="scalar must be a 1-D array"):
        as_1d_float_array(5.0, "scalar")


def test_as_1d_float_array_min_length():
    with pytest.raises(ValueError, match="empty must contain at least 1 sample"):
        as_1d_float_array([], "empty")

    with pytest.raises(ValueError, match="short must contain at least 2 samples"):
        as_1d_float_array([1.0], "short", min_length=2)

    # Passes when length matches
    arr = as_1d_float_array([1.0, 2.0], "ok", min_length=2)
    assert arr.size == 2


def test_as_1d_float_array_check_finite():
    # When check_finite=False, nan/inf are accepted
    arr_nan = as_1d_float_array([1.0, np.nan], check_finite=False)
    assert np.isnan(arr_nan[1])

    with pytest.raises(ValueError, match="nan_arr must contain only finite values"):
        as_1d_float_array([1.0, np.nan], "nan_arr")

    with pytest.raises(ValueError, match="inf_arr must contain only finite values"):
        as_1d_float_array([1.0, np.inf], "inf_arr")
