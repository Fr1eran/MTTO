from dataclasses import dataclass


@dataclass
class Task:
    starting_position: float  # 单位: m
    starting_velocity: float  # 单位: m/s
    destination: float  # 单位: m
    schedule_time: float  # 单位: s
    max_acc_change: float  # 单位: m/s^2
    max_arr_time_error: float  # 单位: s
    max_stop_error: float  # 单位: m
