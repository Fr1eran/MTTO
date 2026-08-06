from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
from numba import (
    njit,
)
from numpy.typing import NDArray

from model.force import (
    air_resis_force,
    air_resis_force_numba,
    guideway_vortex_resis_force,
    guideway_vortex_resis_force_numba,
    linear_generator_resis_force,
    linear_generator_resis_force_numba,
    sledge_frictional_brake_force,
    sledge_frictional_brake_force_numba,
    slope_resis_force,
    slope_resis_force_numba,
    vortex_brake_force,
    vortex_brake_force_numba,
    wear_plate_frictional_brake_force,
    wear_plate_frictional_brake_force_numba,
)
from utils.type_utils import ScalarNumeric, restore_output_type


@dataclass
class VehicleInfo:
    mass: float  # 单位: T
    numoftrainsets: int
    length: float  # 单位: m
    max_speed: float = 500.0 / 3.6  # 单位: m/s
    max_acc: float = 1.0  # 单位: m/s^2
    max_dec: float = -1.0  # 单位: m/s^2，物理语义为负值
    max_slope_capacity: float = 4.0  # 百分位
    levi_power_per_mass: float = 1.7  # 单位 kW/T

    def __post_init__(self) -> None:
        if self.max_acc <= 0.0:
            raise ValueError("max_acc must be positive")
        if self.max_dec >= 0.0:
            raise ValueError("max_dec must be negative under physical sign semantics")

    @property
    def max_dec_abs(self) -> float:
        return -self.max_dec


# 标量化 numba 版本


@njit(cache=True)
def calc_levi_deceleration_scalar_numba(
    speed: float,
    slope: float,
    mass: float,
    numoftrainsets: float,
) -> float:
    f_total = (
        sledge_frictional_brake_force_numba(speed, mass, slope)
        + air_resis_force_numba(speed, numoftrainsets)
        + guideway_vortex_resis_force_numba(speed, numoftrainsets)
        + linear_generator_resis_force_numba(speed, numoftrainsets)
        + slope_resis_force_numba(mass, slope)
    )
    return f_total / mass


@njit(cache=True)
def calc_brake_deceleration_scalar_numba(
    speed: float,
    slope: float,
    mass: float,
    numoftrainsets: float,
    level: int,
) -> float:
    f_total = (
        vortex_brake_force_numba(speed, numoftrainsets, level)
        + wear_plate_frictional_brake_force_numba(speed, numoftrainsets)
        + sledge_frictional_brake_force_numba(speed, mass, slope)
        + air_resis_force_numba(speed, numoftrainsets)
        + guideway_vortex_resis_force_numba(speed, numoftrainsets)
        + linear_generator_resis_force_numba(speed, numoftrainsets)
        + slope_resis_force_numba(mass, slope)
    )
    return f_total / mass


@njit(cache=True)
def calc_longitudinal_force_scalar_numba(
    speed: float,
    slope: float,
    acc: float,
    mass: float,
    numoftrainsets: float,
) -> float:
    f_resis = (
        air_resis_force_numba(speed, numoftrainsets)
        + guideway_vortex_resis_force_numba(speed, numoftrainsets)
        + linear_generator_resis_force_numba(speed, numoftrainsets)
        + slope_resis_force_numba(mass, slope)
    )
    return mass * acc + f_resis


# 向量化 numpy 版本


@overload
def calc_levi_deceleration(
    *,
    mass: float,
    numoftrainsets: int,
    speed: ScalarNumeric,
    slope: ScalarNumeric,
) -> np.float64: ...


@overload
def calc_levi_deceleration(
    *,
    mass: float,
    numoftrainsets: int,
    speed: NDArray[np.floating],
    slope: ScalarNumeric | NDArray[np.floating],
) -> NDArray[np.float64]: ...


def calc_levi_deceleration(
    *,
    mass: float,
    numoftrainsets: int,
    speed: ScalarNumeric | NDArray[np.floating],
    slope: ScalarNumeric | NDArray[np.floating],
) -> np.float64 | NDArray[np.float64]:
    """
    计算列车悬浮减速度大小

    磁浮列车在无牵引情况下受到的阻力包含：
     - 滑橇摩擦制动力
     - 空气阻力
     - 导向轨的磁化阻力
     - 直线电机阻力
     - 斜坡阻力（重力分力）
     - 固定附加阻力（暂不考虑）
    Arg:
        mass: 列车质量(单位: T)
        numoftrainsets: 列车编组数
        speed: 列车运行速度(单位: m/s)
        slope: 坡度

    Returns:
        悬浮减速度
    """
    speed = np.asarray(speed, np.float64)
    slope = np.asarray(slope, np.float64)
    f_sledge = sledge_frictional_brake_force(speed, mass, slope)
    f_air_resis = air_resis_force(speed, numoftrainsets)
    f_guide_ele_resis = guideway_vortex_resis_force(speed, numoftrainsets)
    f_lineargene_resis = linear_generator_resis_force(speed, numoftrainsets)
    f_grad = slope_resis_force(mass, slope)

    f_total = f_air_resis + f_guide_ele_resis + f_lineargene_resis + f_grad + f_sledge

    return restore_output_type(f_total / mass)


@overload
def calc_brake_deceleration(
    *,
    mass: float,
    numoftrainsets: int,
    speed: ScalarNumeric,
    slope: ScalarNumeric,
    level: int,
) -> np.float64: ...


@overload
def calc_brake_deceleration(
    *,
    mass: float,
    numoftrainsets: int,
    speed: NDArray[np.floating],
    slope: ScalarNumeric | NDArray[np.floating],
    level: int,
) -> NDArray[np.float64]: ...


def calc_brake_deceleration(
    *,
    mass: float,
    numoftrainsets: int,
    speed: ScalarNumeric | NDArray[np.floating],
    slope: ScalarNumeric | NDArray[np.floating],
    level: int,
) -> np.float64 | NDArray[np.float64]:
    """
    计算列车安全制动减速度大小

    磁浮列车在安全制动情形下受到的力包含：
     - 涡流制动力
     - 制动磨耗板的摩擦制动力
     - 滑橇摩擦制动力
     - 空气阻力
     - 导向轨的磁化阻力
     - 直线电机阻力
     - 斜坡阻力（重力分力）
     - 固定附加阻力（暂不考虑）

    Arg:
        mass: 列车质量(单位: T)
        numoftrainsets: 列车编组数
        speed: 列车运行速度(单位: m/s)
        slope: 坡度
        level: 涡流制动等级

    Returns:
        安全制动减速度
    """
    speed = np.asarray(speed, np.float64)
    slope = np.asarray(slope, np.float64)
    f_vortex_brake = vortex_brake_force(speed, numoftrainsets, level)
    f_wearplate_brake = wear_plate_frictional_brake_force(speed, numoftrainsets)
    f_sledge_brake = sledge_frictional_brake_force(speed, mass, slope)
    f_air_resis = air_resis_force(speed, numoftrainsets)
    f_guide_ele_resis = guideway_vortex_resis_force(speed, numoftrainsets)
    f_lineargene_resis = linear_generator_resis_force(speed, numoftrainsets)
    f_grad = slope_resis_force(mass, slope)

    f_total = (
        f_vortex_brake
        + f_wearplate_brake
        + f_sledge_brake
        + f_air_resis
        + f_guide_ele_resis
        + f_lineargene_resis
        + f_grad
    )

    return restore_output_type(f_total / mass)


@overload
def calc_longitudinal_force(
    *,
    mass: float,
    numoftrainsets: int,
    acc: ScalarNumeric,
    speed: ScalarNumeric,
    slope: ScalarNumeric,
) -> np.float64: ...


@overload
def calc_longitudinal_force(
    *,
    mass: float,
    numoftrainsets: int,
    acc: NDArray[np.floating],
    speed: ScalarNumeric | NDArray[np.floating],
    slope: ScalarNumeric | NDArray[np.floating],
) -> NDArray[np.float64]: ...


def calc_longitudinal_force(
    *,
    mass: float,
    numoftrainsets: int,
    acc: ScalarNumeric | NDArray[np.floating],
    speed: ScalarNumeric | NDArray[np.floating],
    slope: ScalarNumeric | NDArray[np.floating],
) -> np.float64 | NDArray[np.float64]:
    """
    计算列车受到牵引系统施加的纵向力大小

    磁浮列车在正常运行时受到的阻力包含：
    - 空气阻力
    - 导向轨的磁化阻力
    - 直线电机阻力
    - 斜坡阻力（重力分力）
    - 固定附加阻力（暂不考虑）

    Args:
        mass: 列车质量(单位: T)
        numoftrainsets: 列车编组数
        acc: 列车加速度
        speed: 列车速度(单位: m/s)
        slope: 坡度
    Returns:
        纵向力
    """
    acc = np.asarray(acc, dtype=np.float64)
    f_resis = (
        air_resis_force(speed, numoftrainsets)
        + guideway_vortex_resis_force(speed, numoftrainsets)
        + linear_generator_resis_force(speed, numoftrainsets)
        + slope_resis_force(mass, slope)
    )
    f_longitudinal = mass * acc + f_resis

    return restore_output_type(f_longitudinal)
