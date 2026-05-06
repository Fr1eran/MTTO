from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.force.resis import (
    air_resis_force,
    guideway_vortex_resis_force,
    linear_generator_resis_force,
    slope_resis_force,
)
from model.force.brake import (
    sledge_frictional_brake_force,
    vortex_brake_force,
    wear_plate_frictional_brake_force,
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


class VehicleDynamic:
    @overload
    @staticmethod
    def calc_levi_deceleration(
        vehicle: VehicleInfo, speed: ScalarNumeric, slope: ScalarNumeric
    ) -> np.float64: ...

    @overload
    @staticmethod
    def calc_levi_deceleration(
        vehicle: VehicleInfo, speed: ArrayLike, slope: ArrayLike
    ) -> NDArray[np.float64]: ...

    @staticmethod
    def calc_levi_deceleration(
        vehicle: VehicleInfo,
        speed: ScalarNumeric | ArrayLike,
        slope: ScalarNumeric | ArrayLike,
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
            vehicle: 列车属性
            speed: 列车运行速度(单位: m/s)
            slope: 坡度

        Returns:
            悬浮减速度
        """
        speed = np.asarray(speed, np.float64)
        slope = np.asarray(slope, np.float64)
        f_sledge = sledge_frictional_brake_force(speed, vehicle.mass, slope)
        f_air_resis = air_resis_force(speed, vehicle.numoftrainsets)
        f_guide_ele_resis = guideway_vortex_resis_force(speed, vehicle.numoftrainsets)
        f_lineargene_resis = linear_generator_resis_force(speed, vehicle.numoftrainsets)
        f_grad = slope_resis_force(vehicle.mass, slope)

        f_total = (
            f_air_resis + f_guide_ele_resis + f_lineargene_resis + f_grad + f_sledge
        )

        return restore_output_type(f_total / vehicle.mass)

    @overload
    @staticmethod
    def calc_brake_deceleration(
        vehicle: VehicleInfo, speed: ScalarNumeric, slope: ScalarNumeric, level: int
    ) -> np.float64: ...

    @overload
    @staticmethod
    def calc_brake_deceleration(
        vehicle: VehicleInfo, speed: ArrayLike, slope: ArrayLike, level: int
    ) -> NDArray[np.float64]: ...

    @staticmethod
    def calc_brake_deceleration(
        vehicle: VehicleInfo,
        speed: ScalarNumeric | ArrayLike,
        slope: ScalarNumeric | ArrayLike,
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
            vehicle: 列车属性
            speed: 列车运行速度(单位: m/s)
            slope: 坡度
            level: 涡流制动等级

        Returns:
            安全制动减速度
        """
        speed = np.asarray(speed, np.float64)
        slope = np.asarray(slope, np.float64)
        f_vortex_brake = vortex_brake_force(speed, vehicle.numoftrainsets, level)
        f_wearplate_brake = wear_plate_frictional_brake_force(
            speed, vehicle.numoftrainsets
        )
        f_sledge_brake = sledge_frictional_brake_force(speed, vehicle.mass, slope)
        f_air_resis = air_resis_force(speed, vehicle.numoftrainsets)
        f_guide_ele_resis = guideway_vortex_resis_force(speed, vehicle.numoftrainsets)
        f_lineargene_resis = linear_generator_resis_force(speed, vehicle.numoftrainsets)
        f_grad = slope_resis_force(vehicle.mass, slope)

        f_total = (
            f_vortex_brake
            + f_wearplate_brake
            + f_sledge_brake
            + f_air_resis
            + f_guide_ele_resis
            + f_lineargene_resis
            + f_grad
        )

        return restore_output_type(f_total / vehicle.mass)

    @overload
    @staticmethod
    def calc_longitudinal_force(
        vehicle: VehicleInfo,
        acc: ScalarNumeric,
        speed: ScalarNumeric,
        slope: ScalarNumeric,
    ) -> np.float64: ...

    @overload
    @staticmethod
    def calc_longitudinal_force(
        vehicle: VehicleInfo,
        acc: ArrayLike,
        speed: ArrayLike,
        slope: ArrayLike,
    ) -> NDArray[np.float64]: ...

    @staticmethod
    def calc_longitudinal_force(
        vehicle: VehicleInfo,
        acc: ScalarNumeric | ArrayLike,
        speed: ScalarNumeric | ArrayLike,
        slope: ScalarNumeric | ArrayLike,
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
            vehicle: 列车属性
            acc: 列车加速度
            speed: 列车速度(单位: m/s)
            slope: 坡度
        Returns:
            纵向力
        """
        acc = np.asarray(acc, dtype=np.float64)
        f_resis = (
            air_resis_force(speed, vehicle.numoftrainsets)
            + guideway_vortex_resis_force(speed, vehicle.numoftrainsets)
            + linear_generator_resis_force(speed, vehicle.numoftrainsets)
            + slope_resis_force(vehicle.mass, slope)
        )
        f_longitudinal = vehicle.mass * acc + f_resis

        return restore_output_type(f_longitudinal)
