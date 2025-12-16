import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, TypedDict
from model.Vehicle import Vehicle
from model.Track import Track, TrackProfile
from model.Task import Task
from utils.CalcEnergy import CalcEnergy


class GeneralOperation(NamedTuple):
    acc: float
    operation_time: float


class ForwardBeginPoint(NamedTuple):
    begin_pos: float
    begin_speed: float
    begin_interval: int


class BackwardEndPoint(NamedTuple):
    end_pos: float
    end_speed: float
    end_interval: int


class AscendOperation(TypedDict):
    ascend_begin_pos: float
    ascend_begin_speed: float
    ascend_operation_time: float
    ascend_end_pos: float
    ascend_end_interval: int
    ascend_begin_interval: int


class DescendOperation(TypedDict):
    descend_end_pos: float
    descend_end_speed: float
    descend_operation_time: float
    descend_begin_pos: float
    descend_begin_interval: int
    descend_end_interval: int


class ORS:
    """Operation Reference System"""

    def __init__(
        self, *, vehicle: Vehicle, track: Track, task: Task, gamma: float
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.trackprofile = TrackProfile(track=track)
        self.end_speed: float = 0.0
        self.destination: float = task.destination
        self.schedule_time: float = task.schedule_time
        self.gamma: float = gamma

    def _getSpeedLimitsIntervalIndex(self, pos: float, *, ascend: bool = True) -> int:
        """
        获得当前位置所处区间的索引
        Args:
            pos: 当前位置
            interval_points: 区间端点（升序）
        返回:
            区间索引(np.ndarray 或 int):
              - 正向/forward: 若 pos 与某个端点相等，则视为属于将该端点作为左端点的区间（左端点包含）
              - 反向/backward: 若 pos 与某个端点相等，则视为属于将该端点作为右端点的区间（右端点包含）
        """

        if ascend:  # 正向
            side = "right"
        else:  # 反向
            side = "left"

        idx = np.searchsorted(self.track.speed_limit_intervals, pos, side=side) - 1
        return int(idx)

    def _findSpeedRiseEntryAndFallExit(
        self,
        *,
        start_idx: int,
        end_idx: int,
    ) -> tuple[list[ForwardBeginPoint], list[BackwardEndPoint]]:
        """
        返回所有限速开始上升的入口点（位置与对应限速）列表
        和所有限速下降后的出口点（位置与对应限速）列表。

        返回格式： (AscendBeginPoints, DescendEndPoints)
          - AscendBeginPoints: list of ForwardBeginPoint, 应为在上升沿边界位置、左侧的区间限速、右侧区间编号
          - DescendEndPoints: list of BackwardEndPoint, 应为在下降沿边界位置、右侧的区间限速、左侧区间编号
        """

        n = len(self.track.speed_limits)
        if n < 2:
            return [], []

        diff = np.diff(self.track.speed_limits)
        diff_indices = np.arange(n - 1)
        idx_mask = (diff_indices >= int(start_idx)) & (diff_indices < int(end_idx))

        # 上升入口点：diff > 0 且在 idx_mask 范围内 -> boundary at pts[i+1]
        inc_idxs = np.where((diff > 0) & idx_mask)[0]
        AscendBeginPoints = [
            ForwardBeginPoint(
                float(self.track.speed_limit_intervals[i + 1]),
                float(self.track.speed_limits[i] * self.gamma),
                int(i + 1),
            )
            for i in inc_idxs
        ]

        # 下降出口点：diff < 0 且在 idx_mask 范围内
        dec_idxs = np.where((diff < 0) & idx_mask)[0]
        DescendEndPoints = [
            BackwardEndPoint(
                float(self.track.speed_limit_intervals[j + 1]),
                float(self.track.speed_limits[j + 1] * self.gamma),
                int(j),
            )
            for j in dec_idxs
        ]

        return AscendBeginPoints, DescendEndPoints

    def _cal_mb_descend_operation(
        self,
        end_pos: float,
        end_speed: float,
    ) -> tuple[float, float, int]:
        """
        根据当前位置和速度反向计算在最大制动加速度下的
        部分运行过程，直至达到顶棚运行速度
        Args:
            end_speed: 终点速度
            end_pos: 终点位置
        Returns:
            最大制动模式的起始位置
            最大制动模式的运行时间
            最大制动模式的起始区间
        """
        begin_interval: int = self._getSpeedLimitsIntervalIndex(end_pos, ascend=False)
        while begin_interval >= 0:
            mark_pos = self.track.speed_limit_intervals[
                begin_interval
            ]  # 限速区间左端点
            begin_speed = self.track.speed_limits[begin_interval] * self.gamma
            operation_time = (begin_speed - end_speed) / self.vehicle.max_dec
            begin_pos = end_pos - (begin_speed**2 - end_speed**2) / (
                2 * self.vehicle.max_dec
            )
            # begin_pos = end_pos - (begin_speed + end_speed) * (
            #     begin_speed - end_speed
            # ) / (2 * self.vehicle.max_dec)

            # if begin_pos >= mark_pos:  # 达到顶棚速度的位置在当前限速区间内
            if (begin_pos > mark_pos) or np.isclose(
                begin_pos, mark_pos
            ):  # 达到顶棚速度的位置在当前限速区间内
                break
            else:
                distance = end_pos - mark_pos
                edge_speed_2 = end_speed**2 + 2 * self.vehicle.max_dec * distance
                edge_speed = np.sqrt(edge_speed_2)
                next_interval_speed_limit = (
                    self.track.speed_limits[
                        np.clip(begin_interval - 1, 0, len(self.track.speed_limits) - 1)
                    ]
                    * self.gamma
                )
                if (edge_speed < next_interval_speed_limit) or np.isclose(
                    edge_speed, next_interval_speed_limit
                ):
                    begin_interval -= 1
                else:
                    break

        return (
            float(begin_pos),
            float(operation_time),
            int(np.clip(begin_interval, 0, len(self.track.speed_limits) - 1)),
        )

    def _cal_ma_ascend_operation(
        self,
        begin_pos: float,
        begin_speed: float,
    ) -> tuple[float, float, int]:
        """
        根据当前位置和速度反向计算在最大牵引加速度下的
        部分运行过程，直至达到顶棚运行速度
        Args:
            end_speed: 终点速度
            end_pos: 终点位置
        Returns:
            最大牵引模式下的终点位置
            最大牵引模式下的运行时间
            最大牵引模式下的终止区间
        """
        end_interval: int = self._getSpeedLimitsIntervalIndex(begin_pos, ascend=True)
        while end_interval <= len(self.track.speed_limits) - 1:
            mark_pos = self.track.speed_limit_intervals[
                end_interval + 1
            ]  # 限速区间右端点
            end_speed = self.track.speed_limits[end_interval] * self.gamma
            operation_time = (end_speed - begin_speed) / self.vehicle.max_acc
            end_pos = begin_pos + (end_speed**2 - begin_speed**2) / (
                2 * self.vehicle.max_acc
            )
            # end_pos = begin_pos + (end_speed + begin_speed) * (
            #     end_speed - begin_speed
            # ) / (2 * self.vehicle.max_acc)

            if (end_pos < mark_pos) or np.isclose(
                end_pos, mark_pos
            ):  # 达到顶棚速度的位置在当前限速区间内
                break
            else:
                distance = mark_pos - begin_pos
                edge_speed_2 = begin_speed**2 + 2 * self.vehicle.max_acc * distance
                edge_speed = np.sqrt(edge_speed_2)
                next_interval_speed_limit = (
                    self.track.speed_limits[
                        np.clip(end_interval + 1, 0, len(self.track.speed_limits) - 1)
                    ]
                    * self.gamma
                )
                if (edge_speed < next_interval_speed_limit) or np.isclose(
                    edge_speed, next_interval_speed_limit
                ):
                    end_interval += 1
                else:
                    break

        return (
            float(end_pos),
            float(operation_time),
            int(np.clip(end_interval, 0, len(self.track.speed_limits) - 1)),
        )

    def _cal_withnocruise_scenario(
        self, begin_pos: float, begin_speed: float, end_pos: float, end_speed: float
    ) -> tuple[list[GeneralOperation], float | None]:
        """计算限速区间内不存在巡航阶段的操作模式序列"""
        operation = []
        min_brake_distance = (begin_speed**2 - end_speed**2) / (
            2 * self.vehicle.max_dec
        )
        sceptical_pos: float | None = None
        # min_brake_distance = (
        #     (begin_speed + end_speed)
        #     * (begin_speed - end_speed)
        #     / (2 * self.vehicle.max_dec)
        # )
        if min_brake_distance > (end_pos - begin_pos):
            # 此时即使施加最大制动也无法停靠
            # 减速到目标速度
            sceptical_pos = begin_pos - (begin_speed**2 - end_speed**2) / (
                2 * self.vehicle.max_dec
            )
            operation_time = (begin_speed - end_speed) / self.vehicle.max_dec
            operation.append(GeneralOperation(-self.vehicle.max_dec, operation_time))
        else:
            # 先以最大加速度牵引再以最大减速度制动到目标位置
            speed_peak_2 = (
                (
                    2
                    * self.vehicle.max_acc
                    * self.vehicle.max_dec
                    * (end_pos - begin_pos)
                )
                + self.vehicle.max_dec * begin_speed**2
                + self.vehicle.max_acc * end_speed**2
            ) / (self.vehicle.max_acc + self.vehicle.max_dec)  # 计算工况切换的顶棚速度
            speed_peak = np.sqrt(speed_peak_2)
            forward_operation_time = (speed_peak - begin_speed) / self.vehicle.max_acc
            backward_operation_time = (speed_peak - end_speed) / self.vehicle.max_dec
            operation.append(
                GeneralOperation(self.vehicle.max_acc, forward_operation_time)
            )
            operation.append(
                GeneralOperation(-self.vehicle.max_dec, backward_operation_time)
            )
        return operation, sceptical_pos

    def _cal_withcruise_scenario(
        self,
        cruise_begin_pos: float,
        cruise_end_pos: float,
        cruise_interval: int,
        ma_time: float,
        mb_time: float,
    ) -> list[GeneralOperation]:
        """计算限速区间内存在巡航阶段的操作模式序列"""
        operation = []
        cruise_speed = self.track.speed_limits[cruise_interval] * self.gamma
        cruise_time = (cruise_end_pos - cruise_begin_pos) / cruise_speed
        operation.append(GeneralOperation(self.vehicle.max_acc, ma_time))
        operation.append(GeneralOperation(0.0, cruise_time))
        operation.append(GeneralOperation(-self.vehicle.max_dec, mb_time))
        return operation

    def _cal_min_runtime_operation(self, current_pos: float, current_speed: float):
        """
        计算列车在给定线路条件和约束下的最短运行时间操作序列
        """
        tow_begin_speed = current_speed
        tow_begin_pos = current_pos
        brake_end_speed = 0.0
        brake_end_pos = self.destination

        (
            tow_end_pos,
            tow_operation_time,
            tow_end_interval,
        ) = self._cal_ma_ascend_operation(
            begin_pos=tow_begin_pos, begin_speed=tow_begin_speed
        )

        (
            brake_begin_pos,
            brake_operation_time,
            brake_begin_interval,
        ) = self._cal_mb_descend_operation(
            end_pos=brake_end_pos,
            end_speed=brake_end_speed,
        )
        operations: list[GeneralOperation] = []

        if tow_end_interval < brake_begin_interval:
            ascend_begin_points, descend_end_points = (
                self._findSpeedRiseEntryAndFallExit(
                    start_idx=tow_end_interval, end_idx=brake_begin_interval
                )
            )

            ascend_operations: list[AscendOperation] = [
                {
                    "ascend_begin_pos": tow_begin_pos,
                    "ascend_begin_speed": tow_begin_speed,
                    "ascend_operation_time": tow_operation_time,
                    "ascend_end_pos": tow_end_pos,
                    "ascend_end_interval": tow_end_interval,
                    "ascend_begin_interval": self._getSpeedLimitsIntervalIndex(
                        tow_begin_pos,
                        ascend=True,
                    ),
                }
            ]

            descend_operations: list[DescendOperation] = [
                {
                    "descend_end_pos": brake_end_pos,
                    "descend_end_speed": brake_end_speed,
                    "descend_operation_time": brake_operation_time,
                    "descend_begin_pos": brake_begin_pos,
                    "descend_begin_interval": brake_begin_interval,
                    "descend_end_interval": self._getSpeedLimitsIntervalIndex(
                        brake_end_pos,
                        ascend=False,
                    ),
                }
            ]
            prev_ascend_end_interval = tow_end_interval
            for (
                ascend_begin_pos,
                ascend_begin_speed,
                ascend_begin_interval,
            ) in ascend_begin_points:
                # 如果当前正向最大牵引的起始位置被上个正向牵引曲线包含，则不计算
                if ascend_begin_interval > prev_ascend_end_interval:
                    (
                        ascend_end_pos,
                        ascend_operation_time,
                        ascend_end_interval,
                    ) = self._cal_ma_ascend_operation(
                        begin_pos=ascend_begin_pos, begin_speed=ascend_begin_speed
                    )

                    ascend_operations.append(
                        {
                            "ascend_begin_pos": ascend_begin_pos,
                            "ascend_begin_speed": ascend_begin_speed,
                            "ascend_operation_time": ascend_operation_time,
                            "ascend_end_pos": ascend_end_pos,
                            "ascend_end_interval": ascend_end_interval,
                            "ascend_begin_interval": ascend_begin_interval,
                        }
                    )
                    prev_ascend_end_interval = ascend_end_interval

            prev_descend_begin_interval = brake_begin_interval
            descend_end_points.reverse()
            for (
                descend_end_pos,
                descend_end_speed,
                descend_end_interval,
            ) in descend_end_points:
                if descend_end_interval < prev_descend_begin_interval:
                    (
                        descend_begin_pos,
                        descend_operation_time,
                        descend_begin_interval,
                    ) = self._cal_mb_descend_operation(
                        end_pos=descend_end_pos, end_speed=descend_end_speed
                    )

                    descend_operations.append(
                        {
                            "descend_end_pos": descend_end_pos,
                            "descend_end_speed": descend_end_speed,
                            "descend_operation_time": descend_operation_time,
                            "descend_begin_pos": descend_begin_pos,
                            "descend_begin_interval": descend_begin_interval,
                            "descend_end_interval": descend_end_interval,
                        }
                    )
            # descend_operations不必反转
            # ToDO: 添加其他特殊情况下的子操作剔除
            # 计算操作模式
            sceptical_pos: float | None = (
                None  # 对于初始状态在最短时间运行曲线轮廓外的情况是必要的
            )
            current_interval = tow_end_interval
            while current_interval <= brake_begin_interval:
                ascend_operation_idx = next(
                    (
                        i
                        for i, fo in enumerate(ascend_operations)
                        if fo["ascend_end_interval"] >= current_interval
                    ),
                    None,
                )
                descend_operation_idx = next(
                    (
                        i
                        for i, bo in enumerate(descend_operations)
                        if bo["descend_begin_interval"] <= current_interval
                    ),
                    None,
                )
                if (ascend_operation_idx is not None) and (
                    descend_operation_idx is not None
                ):
                    # 此时必定要制动
                    ascend_op = ascend_operations[ascend_operation_idx]
                    descend_op = descend_operations[descend_operation_idx]

                    if ascend_op["ascend_end_pos"] > descend_op["descend_begin_pos"]:
                        # 此时不存在中间巡航过程
                        middle_operations, sceptical_pos = (
                            self._cal_withnocruise_scenario(
                                begin_pos=ascend_op["ascend_begin_pos"],
                                begin_speed=ascend_op["ascend_begin_speed"],
                                end_pos=descend_op["descend_end_pos"],
                                end_speed=descend_op["descend_end_speed"],
                            )
                        )
                    else:
                        # 此时必定存在中间巡航过程
                        middle_operations = self._cal_withcruise_scenario(
                            cruise_begin_pos=ascend_op["ascend_end_pos"],
                            cruise_end_pos=descend_op["descend_begin_pos"],
                            cruise_interval=current_interval,
                            ma_time=ascend_op["ascend_operation_time"],
                            mb_time=descend_op["descend_operation_time"],
                        )
                    operations += middle_operations
                    current_interval = descend_op["descend_end_interval"] + 1
                elif (ascend_operation_idx is not None) and (
                    descend_operation_idx is None
                ):
                    # 先最大牵引后巡航
                    ascend_op = ascend_operations[ascend_operation_idx]
                    operations.append(
                        GeneralOperation(
                            self.vehicle.max_acc, ascend_op["ascend_operation_time"]
                        )
                    )
                    cruise_speed = (
                        self.track.speed_limits[current_interval] * self.gamma
                    )
                    cruise_distance = (
                        self.track.speed_limit_intervals[
                            ascend_op["ascend_end_interval"] + 1
                        ]
                        - ascend_op["ascend_end_pos"]
                    )
                    cruise_time = cruise_distance / cruise_speed
                    operations.append(GeneralOperation(0.0, cruise_time))
                    current_interval = ascend_op["ascend_end_interval"] + 1
                elif (ascend_operation_idx is None) and (
                    descend_operation_idx is not None
                ):
                    descend_op = descend_operations[descend_operation_idx]
                    cruise_distance = descend_op["descend_begin_pos"] - (
                        sceptical_pos
                        if sceptical_pos is not None
                        else self.track.speed_limit_intervals[current_interval]
                    )
                    dot = descend_op["descend_operation_time"]
                    if cruise_distance < 0.0:
                        # 此时只有最大制动工况
                        operations.append(
                            GeneralOperation(
                                -self.vehicle.max_dec,
                                dot,
                            )
                        )
                        sceptical_pos += (
                            self.track.speed_limits[current_interval] * self.gamma * dot
                            - 0.5 * self.vehicle.max_dec * dot**2
                        )
                    else:
                        # 先巡航后最大制动
                        cruise_speed = (
                            self.track.speed_limits[current_interval] * self.gamma
                        )
                        cruise_time = cruise_distance / cruise_speed
                        operations.append(GeneralOperation(0.0, cruise_time))
                        operations.append(
                            GeneralOperation(
                                -self.vehicle.max_dec,
                                dot,
                            )
                        )
                        sceptical_pos = None
                    current_interval = descend_op["descend_end_interval"] + 1
                else:
                    # 全程巡航
                    cruise_speed = (
                        self.track.speed_limits[current_interval] * self.gamma
                    )
                    cruise_distance = (
                        self.track.speed_limit_intervals[current_interval + 1]
                        - self.track.speed_limit_intervals[current_interval]
                    )
                    cruise_time = cruise_distance / cruise_speed
                    operations.append(GeneralOperation(0.0, cruise_time))
                    current_interval += 1

        else:  # 没有中间最大牵引或最大制动环节
            if tow_end_pos > brake_begin_pos:
                # 此时不存在中间巡航过程
                operations, _ = self._cal_withnocruise_scenario(
                    begin_pos=tow_begin_pos,
                    begin_speed=tow_begin_speed,
                    end_pos=brake_end_pos,
                    end_speed=brake_end_speed,
                )

            else:
                # 此时必定存在中间巡航过程
                operations = self._cal_withcruise_scenario(
                    cruise_begin_pos=tow_end_pos,
                    cruise_end_pos=brake_begin_pos,
                    cruise_interval=brake_begin_interval,
                    ma_time=tow_operation_time,
                    mb_time=brake_operation_time,
                )

        return operations

    def CalMinRuntimeCurve(
        self, begin_pos: float | np.number, begin_speed: float | np.number
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """计算在给定当前位置、速度下的最小运行速度曲线及其运行时间"""
        current_pos = float(begin_pos)
        current_speed = float(begin_speed)
        operations = self._cal_min_runtime_operation(
            current_pos=current_pos, current_speed=current_speed
        )
        curve_pos_array = np.array([current_pos], dtype=np.float64)
        curve_speed_array = np.array([current_speed], dtype=np.float64)
        for acc, operation_time in operations:
            # build trajectory for this operation (constant accel for duration)
            if operation_time <= 0:
                continue

            # sampling resolution (s)
            dt = 0.1
            n_steps = max(int(np.floor(operation_time / dt)), 2)

            # sample interior points (exclude the final endpoint to avoid duplicates)
            t_samples = np.linspace(
                0.0, operation_time, n_steps, endpoint=True, dtype=np.float64
            )
            speeds = current_speed + acc * t_samples
            positions = (
                current_pos + current_speed * t_samples + 0.5 * acc * t_samples**2
            )

            curve_pos_array = np.concatenate((curve_pos_array[:-1], positions))
            curve_speed_array = np.concatenate((curve_speed_array[:-1], speeds))

            # update current state
            current_pos = curve_pos_array[-1]
            current_speed = curve_speed_array[-1]

        return (
            curve_pos_array.astype(np.float32),
            curve_speed_array.astype(np.float32),
        )

    def CalRefEnergyAndOperationTime(
        self,
        begin_pos: float | np.floating,
        begin_speed: float | np.floating,
        displacement: float | np.floating,
    ) -> tuple[float, float, float]:
        """
        计算在给定当前位置、速度、目标位移量
        和当前参考操作模式下的能耗和运行时间

        Args:
            begin_pos: 起始位置(m)
            begin_speed: 起始速度(m/s)
            displacement: 目标位移量(m)

        Returns:
            ref_mechanic_energy_consumption: 参考机械能耗(J)
            ref_leviated_energy_consumption: 参考悬浮能耗(J)
            ref_operation_time: 参考运行时间(s)
        """
        ref_operations: list[GeneralOperation] = self._cal_min_runtime_operation(
            current_pos=float(begin_pos), current_speed=float(begin_speed)
        )

        ref_mec = 0.0
        ref_lec = 0.0
        ref_operation_time = 0.0
        accumulated_displacement = 0.0

        current_pos_i = float(begin_pos)
        current_speed_i = float(begin_speed)

        # 遍历每个操作段，累计能耗和时间直到达到目标位移
        for acc, operation_time in ref_operations:
            # 计算该操作段的位移
            segment_displacement = (
                current_speed_i * operation_time + 0.5 * acc * operation_time**2
            )

            # 判断是否超过目标位移
            if accumulated_displacement + segment_displacement >= float(displacement):
                # 计算到达目标位移所需的实际时间
                remaining_displacement = float(displacement) - accumulated_displacement

                # 求解运动学方程: s = v0*t + 0.5*a*t^2
                if np.abs(acc) < 1e-9:
                    # 匀速运动
                    actual_time = remaining_displacement / np.maximum(
                        np.abs(current_speed_i), 1e-6
                    )
                else:
                    # 变速运动，求解二次方程
                    # 0.5*a*t^2 + v0*t - s = 0
                    discriminant = current_speed_i**2 + 2 * acc * remaining_displacement
                    actual_time = (-current_speed_i + np.sqrt(discriminant)) / acc

                # 计算该段的能耗
                MEC, LEC = CalcEnergy(
                    begin_pos=current_pos_i,
                    begin_velocity=current_speed_i,
                    acc=acc,
                    displacement=remaining_displacement,
                    operation_time=actual_time,
                    vehicle=self.vehicle,
                    trackprofile=self.trackprofile,
                )

                ref_mec += MEC
                ref_lec += LEC
                ref_operation_time += actual_time
                break
            else:
                # 该段未达到目标位移，计算完整段的能耗
                MEC, LEC = CalcEnergy(
                    begin_pos=current_pos_i,
                    begin_velocity=current_speed_i,
                    acc=acc,
                    displacement=segment_displacement,
                    operation_time=operation_time,
                    vehicle=self.vehicle,
                    trackprofile=self.trackprofile,
                )

                ref_mec += MEC
                ref_lec += LEC
                ref_operation_time += operation_time
                accumulated_displacement += segment_displacement

                # 更新当前状态
                current_pos_i += segment_displacement
                current_speed_i += acc * operation_time

        return float(ref_mec), float(ref_lec), float(ref_operation_time)
