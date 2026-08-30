from dataclasses import dataclass
from math import isfinite
from typing import ClassVar


@dataclass
class TrainService:
    DEFAULT_MAX_ARR_TIME_ERROR_S: ClassVar[float] = 10.0

    start_position: float  # 单位: m
    target_position: float  # 单位: m
    schedule_time: float  # 单位: s
    max_acc_change: float  # 单位: m/s^2
    max_stop_error: float  # 单位: m
    max_arr_time_error_s: float = DEFAULT_MAX_ARR_TIME_ERROR_S  # 单位: s，严格准点阈值

    def __post_init__(self) -> None:
        time_error_limit = float(self.max_arr_time_error_s)
        if not isfinite(time_error_limit) or time_error_limit <= 0.0:
            raise ValueError("max_arr_time_error_s must be a finite positive number")
