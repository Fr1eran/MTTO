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
