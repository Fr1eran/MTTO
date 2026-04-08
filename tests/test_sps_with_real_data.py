import numpy as np
import pytest

from model.ocs import SafeGuardUtility,SPS
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_excel,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
)


class TestSPSIntegration:
    @pytest.fixture
    def setup_data(self):
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
        min_curves_list, max_curves_list = load_safeguard_curves(
            "min_curves_list", "max_curves_list"
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
            "min_curves_list": min_curves_list,
            "max_curves_list": max_curves_list,
        }

    @pytest.fixture
    def setup_system(self, setup_data):

        # Initialize SafeGuardUtility
        safeguard_utility = SafeGuardUtility(
            speed_limits=setup_data["speed_limits"],
            speed_limit_intervals=setup_data["speed_limit_intervals"],
            min_curves_list=setup_data["min_curves_list"],
            max_curves_list=setup_data["max_curves_list"],
            factor=0.9,
        )

        return safeguard_utility, len(setup_data["accessible_points"])

    def test_sps_stepping_with_real_data(self, setup_data, setup_system):
        """
        使用真实运行数据测试停车点步进机制实现
        """
        safeguard_utility, num_sp = setup_system

        T_r = 2.0
        sps = SPS(
            sgu=safeguard_utility,
            ASA_ap_list=setup_data["accessible_points"],
            ASA_dp_list=setup_data["dangerous_points"],
            T_s=T_r,
        )

        current_sp = -1
        step_history = []

        distances = setup_data["distance"]
        speeds = setup_data["speed"]
        times = setup_data["time"]

        for i in range(len(times)):
            t = times[i]
            x = distances[i]
            v = speeds[i]

            new_sp = sps.step_to_next_stopping_point(
                current_pos=x, current_speed=v, current_time=t, current_sp=current_sp
            )

            if new_sp != current_sp:
                print(
                    f"Step changed from {current_sp} to {new_sp} at time={t:.2f}s, pos={x:.2f}m, speed={v:.2f}m/s"
                )
                step_history.append((t, current_sp, new_sp))
                current_sp = new_sp

        # 1. 确保发起过停车点步进请求
        assert len(step_history) > 0, "No stepping occurred!"

        # 2. 检查是否到达最终停车点
        final_sp = current_sp
        print(f"Final SP: {final_sp}, Total SPs: {num_sp}")

        assert final_sp == num_sp - 1, (
            f"Train only reached SP {final_sp}, expected is {num_sp - 1}"
        )

        # 3. 检查步进是否单调
        sp_values = [h[2] for h in step_history]
        assert sp_values == sorted(sp_values), "Stepping should be monotonic"
