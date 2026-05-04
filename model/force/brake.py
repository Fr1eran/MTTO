from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScalarNumeric = float | np.floating


def _restore_output_type(
    values: NDArray[np.float64],
) -> np.floating | NDArray[np.float64]:
    if values.ndim == 0:
        return np.float64(values.item())
    return values


@overload
def sledge_frictional_brake_force(
    speed: ScalarNumeric,
    mass: ScalarNumeric,
    slope: ScalarNumeric,
    k: ScalarNumeric = 0.1,
) -> np.floating: ...


@overload
def sledge_frictional_brake_force(
    speed: ArrayLike,
    mass: ScalarNumeric,
    slope: ArrayLike,
    k: ScalarNumeric = 0.1,
) -> NDArray[np.float64]: ...


def sledge_frictional_brake_force(
    speed: ScalarNumeric | ArrayLike,
    mass: ScalarNumeric,
    slope: ScalarNumeric | ArrayLike,
    k: ScalarNumeric = 0.1,
) -> np.floating | NDArray[np.float64]:
    """
    计算滑橇摩擦制动力

    Args:
        speed: 列车速度(单位: m/s)
        mass: 列车质量(单位: T)
        slope: 坡度(百分位, 每100m上升或下降的高度)
        k: 滑橇摩擦的比例系数, 用0~1的数表示摩擦的程度(0表示无摩擦, 1表示100%摩擦)
    Returns:
        列车受到的滑橇摩擦阻力(单位: kN)
    """

    # u: 滑动摩擦系数，随速度变化而变化，这里考虑速度在0~10km/h范围内的变化情况
    # 参考文献：《高速磁浮列车精确停车控制研究》

    MIN_V_KM = 10

    speed_km = 3.6 * np.asarray(speed, dtype=np.float64)
    slope = np.asarray(slope, np.float64)

    u = -0.003 * speed_km + 0.27
    sledge_frictional_resis_force = np.where(
        speed_km <= MIN_V_KM, k * u * mass * 100 / np.sqrt(100**2 + slope**2) * 9.8, 0.0
    )

    return _restore_output_type(sledge_frictional_resis_force)


@overload
def vortex_brake_force(
    speed: ScalarNumeric,
    numoftrainsets: int,
    level: int,
) -> np.floating: ...


@overload
def vortex_brake_force(
    speed: ArrayLike,
    numoftrainsets: int,
    level: int,
) -> NDArray[np.float64]: ...


def vortex_brake_force(
    speed: ScalarNumeric | ArrayLike,
    numoftrainsets: int,
    level: int,
) -> np.floating | NDArray[np.float64]:
    """
    计算磁浮列车的涡流制动阻力

    Args:
        speed: 列车速度(单位: m/s)
        numoftrainsets: 列车编组数
        level: 制动等级0-7, 0为最大制动
    Returns:
        涡流制动力(单位: kN)
    """

    # 参考文献：《基于涡流制动技术的高速磁悬浮列车安全制动控制研究》

    COEFF = 147.8
    DENOM = 200.0
    MIN_V_KM = 10

    speed_km = 3.6 * np.asarray(speed, dtype=np.float64)
    vortex_break_force = np.where(
        speed_km > MIN_V_KM,
        (7 - level)
        / 7
        * 2
        * numoftrainsets
        * COEFF
        * np.sqrt(speed_km / DENOM)
        / (speed_km / DENOM + (1 + np.sqrt(speed_km / DENOM)) ** 2),
        0.0,
    )

    return _restore_output_type(vortex_break_force)


@overload
def wear_plate_frictional_brake_force(
    speed: ScalarNumeric,
    numoftrainsets: int,
) -> np.floating: ...


@overload
def wear_plate_frictional_brake_force(
    speed: ArrayLike,
    numoftrainsets: int,
) -> NDArray[np.float64]: ...


def wear_plate_frictional_brake_force(
    speed: ScalarNumeric | ArrayLike,
    numoftrainsets: int,
) -> np.floating | NDArray[np.float64]:
    """
    计算磁浮列车的制动磨耗板的摩擦制动力

    Args:
        velocity: 列车速度(单位: m/s)
        numoftrainsets: 列车编组数
    Returns:
        列车受到的磨耗板摩擦阻力(单位: kN)
    """

    # u: 干燥条件下的磨耗板与导向轨之间的摩擦系数，通过曲线拟合得到
    # 参考文献：《磁浮列车涡流制动系统建模及紧急制动控制策略的研究》

    A = 580.32
    B = 312384.47
    C = 3.0816
    D = 227.727
    E = 42
    MIN_V_KM = 10
    MAX_V_KM = 150

    speed_km = 3.6 * np.asarray(speed, dtype=np.float64)

    mu = np.piecewise(
        speed_km,
        [
            (speed_km >= 0) & (speed_km <= 20),
            (speed_km > 20) & (speed_km <= 30),
            (speed_km > 30) & (speed_km <= 50),
            (speed_km > 50) & (speed_km <= 100),
            (speed_km > 100) & (speed_km <= 200),
            (speed_km > 200),
        ],
        [
            lambda speed: -0.003 * speed + 0.28,
            lambda speed: -0.002 * speed + 0.26,
            lambda speed: -0.001 * speed + 0.23,
            lambda speed: -0.0008 * speed + 0.22,
            lambda speed: -0.0002 * speed + 0.16,
            lambda speed: 0.3,
        ],
    )

    wearplate_frictional_resis_force = np.zeros_like(speed_km, dtype=np.float64)
    mask = (speed_km > MIN_V_KM) & (speed_km <= MAX_V_KM)
    # 只对满足条件的点计算，防止根号内表达式求值为负抛出异常
    tmp = mu[mask] * (
        2 * numoftrainsets * (A - np.sqrt(B - C * (speed_km[mask] - D) ** 2)) - E
    )
    wearplate_frictional_resis_force[mask] = tmp

    return _restore_output_type(wearplate_frictional_resis_force)
