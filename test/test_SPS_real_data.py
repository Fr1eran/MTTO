import json
import os
import sys
import numpy as np
import pandas as pd
import pytest
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.SafeGuard import SafeGuardUtility
from model.SPS import SPS


class TestSPSIntegration:
    @pytest.fixture
    def setup_data(self):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        excel_path = os.path.join(
            base_path, "data", "operation", "a_longyang_to_airport.xlsx"
        )
        raw_data = pd.read_excel(
            excel_path,
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
        v_SI = (
            raw_data["速度(km/h)"][1:].to_numpy(dtype=np.float64) / 3.6
        )  # km/h -> m/s
        travel_time = raw_data["时间(s)"][1:].to_numpy(dtype=np.float64)  # s

        with open(
            os.path.join(base_path, "data", "rail", "raw", "slopes.json"),
            "r",
            encoding="utf-8",
        ) as f:
            slope_data = json.load(f)
            slopes = slope_data["slopes"]
            slope_intervals = slope_data["intervals"]

        with open(
            os.path.join(base_path, "data", "rail", "raw", "speed_limits.json"),
            "r",
            encoding="utf-8",
        ) as f:
            speed_limit_data = json.load(f)
            speed_limits = np.asarray(speed_limit_data["speed_limits"]) / 3.6
            speed_limit_intervals = speed_limit_data["intervals"]

        with open("data/rail/safeguard/min_curves_list.pkl", "rb") as f:
            min_curves_list = pickle.load(f)
        with open("data/rail/safeguard/max_curves_list.pkl", "rb") as f:
            max_curves_list = pickle.load(f)

        return {
            "distance": distance,
            "speed": v_SI,
            "time": travel_time,
            "slopes": slopes,
            "slope_intervals": slope_intervals,
            "speed_limits": speed_limits,
            "speed_limit_intervals": speed_limit_intervals,
            "min_curves_list": min_curves_list,
            "max_curves_list": max_curves_list,
        }

    @pytest.fixture
    def setup_system(self, setup_data):

        # Initialize SafeGuardUtility
        sgu = SafeGuardUtility(
            speed_limits=setup_data["speed_limits"],
            speed_limit_intervals=setup_data["speed_limit_intervals"],
            min_curves_list=setup_data["min_curves_list"],
            max_curves_list=setup_data["max_curves_list"],
            gamma=0.9,
        )

        return sgu, len(setup_data["min_curves_list"])

    def test_sps_stepping_with_real_data(self, setup_data, setup_system):
        """
        使用真实运行数据测试停车点步进机制实现
        """
        sgu, num_ap = setup_system

        T_r = 2.0
        sps = SPS(sgu=sgu, numofSPS=num_ap, T_r=T_r)

        current_sp = -1
        step_history = []

        distances = setup_data["distance"]
        speeds = setup_data["speed"]
        times = setup_data["time"]

        for i in range(len(times)):
            t = times[i]
            x = distances[i]
            v = speeds[i]

            new_sp = sps.StepToNextSP(
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
        print(f"Final SP: {final_sp}, Total SPs: {num_ap}")

        assert final_sp == num_ap - 1, (
            f"Train only reached SP {final_sp}, expected is {num_ap - 1}"
        )

        # 3. 检查步进是否单调
        sp_values = [h[2] for h in step_history]
        assert sp_values == sorted(sp_values), "Stepping should be monotonic"
