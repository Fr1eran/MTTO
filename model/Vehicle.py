from dataclasses import dataclass
from model.force.resis import AirResis, GuidewayVortexResis, LinearGeneResis, SlopeResis
from model.force.brake import (
    SledgeFrictionalBrake,
    VortexBrake,
    WearPlateFrictionalBrake,
)


@dataclass
class Vehicle:
    mass: float  # 单位: T
    numoftrainsets: int
    length: float  # 单位: m
    max_acc: float = 1.0  # 单位: m/s^2
    max_dacc: float = 1.0  # 单位: m/s^2
    levi_power_per_mass: float = 1.7  # 单位 kW/T


class VehicleDynamic:
    @staticmethod
    def CalLeviDacc(vehicle: Vehicle, speed, slope):
        """
        计算列车在给定速度、坡度下受到的悬浮减速度

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
        f_sledge = SledgeFrictionalBrake(speed, vehicle.mass, slope)
        f_air_resis = AirResis(speed, vehicle.numoftrainsets)
        f_guide_ele_resis = GuidewayVortexResis(speed, vehicle.numoftrainsets)
        f_lineargene_resis = LinearGeneResis(speed, vehicle.numoftrainsets)
        f_grad = SlopeResis(vehicle.mass, slope)

        f_total = (
            f_air_resis + f_guide_ele_resis + f_lineargene_resis + f_grad + f_sledge
        )

        return f_total / vehicle.mass

    @staticmethod
    def CalBrakeDacc(vehicle: Vehicle, speed, slope, level):
        """
        计算列车在给定速度、坡度、制动等级下的安全制动减速度

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
        f_vortex_brake = VortexBrake(speed, vehicle.numoftrainsets, level)
        f_wearplate_brake = WearPlateFrictionalBrake(speed, vehicle.numoftrainsets)
        f_sledge_brake = SledgeFrictionalBrake(speed, vehicle.mass, slope)
        f_air_resis = AirResis(speed, vehicle.numoftrainsets)
        f_guide_ele_resis = GuidewayVortexResis(speed, vehicle.numoftrainsets)
        f_lineargene_resis = LinearGeneResis(speed, vehicle.numoftrainsets)
        f_grad = SlopeResis(vehicle.mass, slope)

        f_total = (
            f_vortex_brake
            + f_wearplate_brake
            + f_sledge_brake
            + f_air_resis
            + f_guide_ele_resis
            + f_lineargene_resis
            + f_grad
        )

        return f_total / vehicle.mass

    @staticmethod
    def CalLongitudinalForce(vehicle: Vehicle, acc, speed, slope):
        """
        计算列车在给定加速度、速度、坡度下受到牵引系统施加的纵向力

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
        f_resis = (
            AirResis(speed, vehicle.numoftrainsets)
            + GuidewayVortexResis(speed, vehicle.numoftrainsets)
            + LinearGeneResis(speed, vehicle.numoftrainsets)
            + SlopeResis(vehicle.mass, slope)
        )
        f_longitudinal = vehicle.mass * acc + f_resis

        return f_longitudinal
