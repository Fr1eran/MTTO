from dataclasses import dataclass


@dataclass
class Task:
    start_position: float  # 单位: m
    start_speed: float  # 单位: m/s
    target_position: float  # 单位: m
    schedule_time: float  # 单位: s
    max_acc_change: float  # 单位: m/s^2
    max_arr_time_error: float  # 单位: s
    max_stop_error: float  # 单位: m
