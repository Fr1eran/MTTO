from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ScalarNumeric = float | np.floating


def restore_output_type(values: NDArray[Any]) -> Any:
    if values.ndim == 0:
        return values.dtype.type(values.item())
    return values
