from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

ScalarNumeric = int | float | np.floating
NumericArray = NDArray[np.number] | Sequence[ScalarNumeric]


def restore_output_type[T: np.generic](values: NDArray[T]) -> T | NDArray[T]:
    if values.ndim == 0:
        return values.dtype.type(values.item())
    return values


def as_float(value: object) -> float | None:
    """Safely convert a scalar value to float, returning None if not numeric."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def as_1d_float_array(
    values: object,
    name: str = "array",
    *,
    min_length: int = 1,
    check_finite: bool = True,
) -> NDArray[np.float64]:
    """Convert input to a 1-D float64 array with dimension and value validation.

    Args:
        values: Array-like or sequence of numeric values.
        name: Parameter name for descriptive error messages.
        min_length: Minimum required number of elements (default: 0).
        check_finite: Whether to enforce all elements are finite (default: False).

    Returns:
        1-D numpy float64 array.

    Raises:
        ValueError: If array is not 1-D, has fewer than min_length elements,
            or contains non-finite values when check_finite is True.
    """
    if isinstance(values, str) and not isinstance(name, str):
        values, name = name, values
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if array.size < min_length:
        sample_desc = "sample" if min_length == 1 else "samples"
        raise ValueError(f"{name} must contain at least {min_length} {sample_desc}")
    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
