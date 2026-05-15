from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScalarNumeric = float | np.floating


class SigmoidVariant:
    """Numerically stable sigmoid variant used by score/reward shaping."""

    def __init__(self, x1: float, x2: float, c: float = 10.0):
        assert x2 > x1, "x2 必须大于 x1"

        self.x1 = float(x1)
        self.x2 = float(x2)
        self.c = float(c)

        self.xm = (self.x1 + self.x2) / 2.0
        self.k = self.c / (self.x2 - self.x1)

    @staticmethod
    def _restore_output_type(
        x: ScalarNumeric | ArrayLike,
        values: NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        if np.isscalar(x):
            return float(np.asarray(values).item())
        return values

    @overload
    def __call__(self, x: ScalarNumeric) -> float: ...

    @overload
    def __call__(self, x: ArrayLike) -> NDArray[np.float64]: ...

    def __call__(self, x: ScalarNumeric | ArrayLike) -> float | NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        exponent = np.clip(self.k * (x_arr - self.xm), -500.0, 500.0)
        result = np.asarray(1.0 / (1.0 + np.exp(exponent)), dtype=np.float64)
        return self._restore_output_type(x, result)

    @overload
    def gradient(self, x: ScalarNumeric) -> float: ...

    @overload
    def gradient(self, x: ArrayLike) -> NDArray[np.float64]: ...

    def gradient(self, x: ScalarNumeric | ArrayLike) -> float | NDArray[np.float64]:
        reward = np.asarray(self.__call__(x), dtype=np.float64)
        grad = np.asarray(-self.k * reward * (1.0 - reward), dtype=np.float64)
        return self._restore_output_type(x, grad)
