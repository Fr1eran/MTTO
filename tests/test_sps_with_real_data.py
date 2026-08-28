from typing import TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from model.ocs import SPS, SafeGuardUtility
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_excel,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
)


class _SetupData(TypedDict):
    distance: NDArray[np.float64]
    speed: NDArray[np.float64]
    time: NDArray[np.float64]
    slopes: NDArray[np.float64]
    slope_intervals: NDArray[np.float64]
    speed_limits: NDArray[np.float64]
    speed_limit_intervals: NDArray[np.float64]
    accessible_points: list[float]
    dangerous_points: list[float]
    levi_curves_list: list[NDArray[np.float64]]
    brake_curves_list: list[NDArray[np.float64]]
    min_curves_list: list[NDArray[np.float64]]
    max_curves_list: list[NDArray[np.float64]]


class TestSPSIntegration:
    @pytest.fixture
    def setup_data(self) -> _SetupData:
        raw_data = load_excel(
            "data/operation/a_longyang_to_airport.xlsx",
            sheet_name="a轨_双端两步4节_龙阳－机场",
            header=0,
            dtype=np.float32,
        )
        required_columns = ["里程(km)", "速度(km/h)", "时间(s)"]
        for col in required_columns:
            assert col in raw_data.columns, f"Column {col} missing in excel data"

        distance = (
            raw_data["里程(km)"][1:].to_numpy(dtype=np.float64) * 1000.0
        )  # km -> m
        speed_mps = (
            raw_data["速度(km/h)"][1:].to_numpy(dtype=np.float64) / 3.6
        )  # km/h -> m/s
        travel_time = raw_data["时间(s)"][1:].to_numpy(dtype=np.float64)  # s

        slopes, slope_intervals = load_slopes()
        speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
        accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
        levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
            load_safeguard_curves(
                "levi_curves_list",
                "brake_curves_list",
                "min_curves_list",
                "max_curves_list",
            )
        )

        return {
            "distance": distance,
            "speed": speed_mps,
            "time": travel_time,
            "slopes": slopes,
            "slope_intervals": slope_intervals,
            "speed_limits": speed_limits,
            "speed_limit_intervals": speed_limit_intervals,
            "accessible_points": accessible_points,
            "dangerous_points": dangerous_points,
            "levi_curves_list": levi_curves_list,
            "brake_curves_list": brake_curves_list,
            "min_curves_list": min_curves_list,
            "max_curves_list": max_curves_list,
        }

    @pytest.fixture
    def setup_system(self, setup_data: _SetupData) -> tuple[SafeGuardUtility, int]:

        # Initialize SafeGuardUtility
        safeguard_utility = SafeGuardUtility(
            speed_limits=setup_data["speed_limits"],
            speed_limit_intervals=setup_data["speed_limit_intervals"],
            levi_curves_list=setup_data["levi_curves_list"],
            brake_curves_list=setup_data["brake_curves_list"],
            min_curves_list=setup_data["min_curves_list"],
            max_curves_list=setup_data["max_curves_list"],
            factor=0.9,
        )

        return safeguard_utility, len(setup_data["accessible_points"])

    def test_sps_real_data_stops_at_the_first_missed_step_window(
        self,
        setup_data: _SetupData,
        setup_system: tuple[SafeGuardUtility, int],
    ):
        """
        使用真实运行数据测试停车点步进机制实现
        """
        safeguard_utility, num_sp = setup_system

        T_r = 2.0
        sps = SPS(
            safeguard_utility=safeguard_utility,
            accessible_positions_m=setup_data["accessible_points"],
            danger_positions_m=setup_data["dangerous_points"],
            step_delay_s=T_r,
        )

        sps_state = sps.initial_state()
        current_sp = sps_state.target_stopping_point_index
        request_seen = False
        window_missed = False

        distances = setup_data["distance"]
        speeds = setup_data["speed"]
        times = setup_data["time"]

        for i in range(len(times)):
            t = times[i]
            x = distances[i]
            v = speeds[i]

            previous_state = sps_state
            sps_state = sps.advance(
                previous_state,
                position_m=x,
                speed_mps=v,
                time_s=t,
            )
            request_seen = request_seen or (
                not previous_state.request_pending and sps_state.request_pending
            )
            old_max_speed = safeguard_utility.get_max_speed(
                current_pos=x,
                current_sp=previous_state.target_stopping_point_index,
            )
            if (
                previous_state.request_pending
                and sps_state == previous_state
                and v > old_max_speed
            ):
                window_missed = True
                break
            current_sp = sps_state.target_stopping_point_index

        assert request_seen, "No stepping request occurred in real-data replay"
        assert window_missed, "Expected the replay to expose a missed step window"
        assert current_sp < num_sp - 1
