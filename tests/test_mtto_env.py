import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from model.common import ORS
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo, calc_levi_deceleration_scalar_numba
from rl import env_factory
from rl.mtto_env import MTTOEnv, RewardConfig
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)


def _fake_dp_reference_manifold(
    self: ORS,
    *,
    start_position: float,
    start_speed: float,
    target_position: float,
    schedule_time_s: float,
    target_speed: float = 0.0,
    **_kwargs,
):
    pos_arr = np.asarray(
        [
            start_position,
            (start_position + target_position) / 2.0,
            target_position,
        ],
        dtype=np.float64,
    )
    speed_arr = np.asarray([start_speed, 10.0, target_speed], dtype=np.float64)
    cum_time_arr = np.asarray(
        [0.0, schedule_time_s / 2.0, schedule_time_s], dtype=np.float64
    )
    ref_redundant_arr = np.asarray([20.0, 10.0, 0.0], dtype=np.float64)
    return pos_arr, speed_arr, cum_time_arr, ref_redundant_arr


@pytest.fixture(scope="module", autouse=True)
def patch_dp_reference_manifold_for_env_tests():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ORS,
        "load_or_build_ref_redundant_operation_time_from_dp",
        _fake_dp_reference_manifold,
    )
    yield
    monkeypatch.undo()


@pytest.fixture(scope="module")
def mtto_env():
    # 读取线路数据
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
    longyang_start_position, putong_end_position = load_stations_goal_positions()
    levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
        load_safeguard_curves(
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        )
    )

    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.95,
    )

    track = TrackInfo(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )

    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    train_service = TrainService(
        start_position=longyang_start_position,
        start_speed=0.0,
        target_position=putong_end_position,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=60.0,
        max_stop_error=2.0,
    )

    maglevttoenv = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=0.995,
        max_step_distance=10.0,
        reward_config=RewardConfig(),
    )
    return maglevttoenv


def _clone_train_service(train_service: TrainService) -> TrainService:
    return TrainService(
        start_position=train_service.start_position,
        start_speed=train_service.start_speed,
        target_position=train_service.target_position,
        schedule_time=train_service.schedule_time,
        max_acc_change=train_service.max_acc_change,
        max_arr_time_error_ratio=train_service.max_arr_time_error_ratio,
        max_stop_error=train_service.max_stop_error,
    )


def _build_env_like(
    source_env: MTTOEnv,
    *,
    train_service: TrainService | None = None,
    reward_config: RewardConfig | None = None,
    **kwargs,
) -> MTTOEnv:
    return MTTOEnv(
        vehicle=source_env.vehicle,
        track=source_env.track,
        safeguard_utility=source_env.safeguard_utility,
        train_service=train_service or _clone_train_service(source_env.train_service),
        gamma=source_env.gamma,
        max_step_distance=source_env.max_step_distance,
        reward_config=reward_config if reward_config is not None else RewardConfig(),
        **kwargs,
    )


def test_reset(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (12,)
    np.testing.assert_allclose(obs[0], 1.0)  # remaining_distance
    np.testing.assert_allclose(obs[1], 0.0)  # current_speed
    np.testing.assert_allclose(obs[2], 0.0)  # current_acc
    np.testing.assert_allclose(
        obs[3], mtto_env._normalize_acc_to_action(mtto_env._calc_coasting_acc())
    )  # suggested_dec_normalized
    np.testing.assert_allclose(obs[4], 1.0)  # remaining_schedule_time
    np.testing.assert_allclose(
        obs[5],
        mtto_env.current_redundant_operation_time
        / mtto_env.train_service.schedule_time,
    )  # time_redundancy
    np.testing.assert_allclose(
        obs[6],
        mtto_env.current_max_speed / mtto_env.vehicle.max_speed,
    )  # current_max_speed
    np.testing.assert_allclose(
        obs[7],
        mtto_env.current_min_speed / mtto_env.vehicle.max_speed,
    )  # current_min_speed
    np.testing.assert_allclose(obs[8], 0.0)  # current_slope
    np.testing.assert_allclose(
        obs[9],
        mtto_env._calc_lookahead_avg_slope() / mtto_env.vehicle.max_slope_capacity,
    )  # lookahead_avg_slope
    np.testing.assert_allclose(
        obs[10],
        mtto_env._calc_lookahead_avg_upper_speed() / mtto_env.vehicle.max_speed,
    )  # lookahead_avg_upper_speed
    np.testing.assert_allclose(obs[11], 0.0)  # approach_progress
    assert info == {}


def test_suggested_dec_uses_coasting_acc_outside_final_approach(mtto_env: MTTOEnv):
    mtto_env.reset()
    mtto_env.current_speed = 30.0
    mtto_env.current_slope = 1.0

    obs = mtto_env._get_obs()

    coasting_dec = calc_levi_deceleration_scalar_numba(
        mtto_env.current_speed,
        mtto_env.current_slope,
        mtto_env.vehicle.mass,
        mtto_env.vehicle.numoftrainsets,
    )
    expected_coasting_acc = float(
        np.clip(
            -coasting_dec,
            mtto_env.vehicle.max_dec,
            mtto_env.vehicle.max_acc,
        )
    )
    np.testing.assert_allclose(
        obs[3],
        mtto_env._normalize_acc_to_action(expected_coasting_acc),
    )


def test_suggested_dec_uses_required_stop_dec_in_final_approach(mtto_env: MTTOEnv):
    mtto_env.reset()
    mtto_env.current_position = mtto_env.train_service.target_position - 1000.0
    mtto_env.current_speed = 20.0
    mtto_env.current_slope = 1.0

    obs = mtto_env._get_obs()

    required_dec = -(mtto_env.current_speed**2) / (2.0 * 1000.0)
    np.testing.assert_allclose(
        obs[3],
        mtto_env._normalize_acc_to_action(required_dec),
    )
    np.testing.assert_allclose(
        obs[11],
        mtto_env._calc_approach_progress(1000.0),
    )  # approach_progress


def test_get_upper_speed_or_zero_returns_zero_outside_lut(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.reset()
    monkeypatch.setattr(mtto_env, "_upper_speed_lut_pos_min", 100.0)
    monkeypatch.setattr(mtto_env, "_upper_speed_lut_step", 10.0)
    monkeypatch.setattr(
        mtto_env,
        "_upper_speed_lut_speed_arr",
        np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
    )

    assert mtto_env._get_upper_speed_or_zero(99.0) == pytest.approx(0.0)
    assert mtto_env._get_upper_speed_or_zero(110.0) == pytest.approx(10.0)
    assert mtto_env._get_upper_speed_or_zero(121.0) == pytest.approx(0.0)


def test_lookahead_avg_upper_speed_uses_window_average(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.reset()
    mtto_env.current_position = 0.0
    mtto_env.current_speed = 5.0
    monkeypatch.setattr(mtto_env, "_upper_speed_lut_pos_min", 0.0)
    monkeypatch.setattr(mtto_env, "_upper_speed_lut_step", 10.0)
    monkeypatch.setattr(
        mtto_env,
        "_upper_speed_lut_speed_arr",
        np.asarray([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32),
    )

    avg_upper_speed = mtto_env._calc_lookahead_avg_upper_speed(
        lookahead_distance=100.0,
        num_samples=2,
    )

    np.testing.assert_allclose(avg_upper_speed, 10.0)


def test_lookahead_avg_slope_uses_window_average(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.reset()
    mtto_env.current_position = 0.0
    monkeypatch.setattr(mtto_env.track, "slopes", np.asarray([0.0, 4.0]))
    monkeypatch.setattr(mtto_env.track, "slope_intervals", np.asarray([0.0, 50.0]))

    avg_slope = mtto_env._calc_lookahead_avg_slope(
        lookahead_distance=100.0,
        num_samples=2,
    )

    np.testing.assert_allclose(avg_slope, 2.0)


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    mec1, lec1 = mtto_env.ecc.calc_energy(
        begin_pos=mtto_env.current_position,
        begin_speed=mtto_env.current_speed,
        acc=0.0,
        distance=0.0,
        direction=mtto_env.direction,
        operation_time=0.0,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = mtto_env.ecc.calc_energy(
        begin_pos=mtto_env.current_position,
        begin_speed=mtto_env.current_speed,
        acc=1.0,
        distance=100.0,
        direction=mtto_env.direction,
        operation_time=14.142,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption2 = mec2 + lec2

    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_reference_remaining_operation_time_matches_endpoints(mtto_env: MTTOEnv):
    mtto_env.reset()

    start_remaining = mtto_env._get_ref_remaining_operation_time(
        mtto_env.train_service.start_position
    )
    target_remaining = mtto_env._get_ref_remaining_operation_time(
        mtto_env.train_service.target_position
    )

    assert start_remaining == pytest.approx(
        mtto_env.ref_total_operation_time,
        rel=1e-3,
    )
    assert target_remaining == pytest.approx(0.0, abs=0.5)


def test_mtto_env_initializes_v18_dp_reference_with_task_params(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    called_kwargs = {}

    def fake_load_dp_reference(self: ORS, **kwargs):
        called_kwargs.update(kwargs)
        return _fake_dp_reference_manifold(self, **kwargs)

    monkeypatch.setattr(
        ORS,
        "load_or_build_ref_redundant_operation_time_from_dp",
        fake_load_dp_reference,
    )

    train_service = _clone_train_service(mtto_env.train_service)
    train_service.schedule_time = 440.0
    env = _build_env_like(
        mtto_env,
        train_service=train_service,
        punctuality_dp_curve_dir=tmp_path,
        punctuality_reference_match_tolerance=0.25,
    )

    assert called_kwargs["start_position"] == pytest.approx(
        train_service.start_position
    )
    assert called_kwargs["start_speed"] == pytest.approx(train_service.start_speed)
    assert called_kwargs["target_position"] == pytest.approx(
        train_service.target_position
    )
    assert called_kwargs["target_speed"] == pytest.approx(0.0)
    assert called_kwargs["schedule_time_s"] == pytest.approx(440.0)
    assert called_kwargs["dp_curve_dir"] == tmp_path
    assert called_kwargs["force_recompute"] is False
    assert called_kwargs["match_tolerance"] == pytest.approx(0.25)
    assert env.ref_redundant_operation_time_func(
        train_service.start_position
    ) == pytest.approx(20.0)


def test_potential_punctuality_v18_peaks_on_dp_reference(mtto_env: MTTOEnv) -> None:
    midpoint = (
        mtto_env.train_service.start_position + mtto_env.train_service.target_position
    ) / 2.0
    ref_value = mtto_env.ref_redundant_operation_time_func(midpoint)

    peak = mtto_env._potential_punctuality_v18(
        pos=midpoint,
        redundant_operation_time=ref_value,
    )
    off_reference = mtto_env._potential_punctuality_v18(
        pos=midpoint,
        redundant_operation_time=ref_value + 10.0,
    )

    assert peak == pytest.approx(0.0)
    assert off_reference < peak


def test_punctuality_dense_reward_uses_dp_speed_reference(
    mtto_env: MTTOEnv,
) -> None:
    mtto_env.reset()
    midpoint = (
        mtto_env.train_service.start_position + mtto_env.train_service.target_position
    ) / 2.0
    prev_pos = mtto_env.train_service.start_position

    mtto_env.current_position = midpoint
    mtto_env.current_speed = 15.0
    mtto_env.last_state["pos"] = prev_pos
    mtto_env.last_state["speed"] = mtto_env.train_service.start_speed

    expected = mtto_env._potential_punctuality_v39(
        pos=midpoint,
        speed=15.0,
    ) - mtto_env._potential_punctuality_v39(
        pos=prev_pos,
        speed=mtto_env.train_service.start_speed,
    )

    assert mtto_env._get_reward_punctuality_dense() == pytest.approx(expected)


def test_dp_speed_reference_is_loaded_and_interpolated(mtto_env: MTTOEnv) -> None:
    midpoint = (
        mtto_env.train_service.start_position + mtto_env.train_service.target_position
    ) / 2.0

    assert mtto_env.ref_dp_speed_pos_arr.size == 3
    assert mtto_env.ref_dp_speed_arr.size == 3
    assert mtto_env._get_ref_dp_speed(mtto_env.train_service.start_position) == pytest.approx(
        mtto_env.train_service.start_speed
    )
    assert mtto_env._get_ref_dp_speed(midpoint) == pytest.approx(10.0)
    assert mtto_env._get_ref_dp_speed(mtto_env.train_service.target_position) == pytest.approx(
        0.0
    )


def test_potential_punctuality_v39_peaks_on_dp_speed_curve(
    mtto_env: MTTOEnv,
) -> None:
    midpoint = (
        mtto_env.train_service.start_position + mtto_env.train_service.target_position
    ) / 2.0

    peak = mtto_env._potential_punctuality_v39(pos=midpoint, speed=10.0)
    off_reference = mtto_env._potential_punctuality_v39(pos=midpoint, speed=15.0)

    assert peak == pytest.approx(0.0)
    assert off_reference < peak


def test_punctuality_dense_reward_tracks_dp_speed_error(
    mtto_env: MTTOEnv,
) -> None:
    mtto_env.reset()
    midpoint = (
        mtto_env.train_service.start_position + mtto_env.train_service.target_position
    ) / 2.0

    mtto_env.current_position = midpoint
    mtto_env.current_speed = 15.0
    mtto_env.last_state["pos"] = mtto_env.train_service.start_position
    mtto_env.last_state["speed"] = mtto_env.train_service.start_speed

    expected = mtto_env._potential_punctuality_v39(
        pos=midpoint,
        speed=15.0,
    ) - mtto_env._potential_punctuality_v39(
        pos=mtto_env.train_service.start_position,
        speed=mtto_env.train_service.start_speed,
    )

    assert mtto_env._get_reward_punctuality_dense() == pytest.approx(expected)


def test_mtto_env_raises_when_v18_dp_reference_is_missing(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def raise_missing_reference(self: ORS, **_kwargs):
        raise FileNotFoundError("missing matching DP trajectory")

    monkeypatch.setattr(
        ORS,
        "load_or_build_ref_redundant_operation_time_from_dp",
        raise_missing_reference,
    )

    with pytest.raises(FileNotFoundError, match="missing matching DP trajectory"):
        _build_env_like(mtto_env, punctuality_dp_curve_dir=tmp_path)


def test_change_schedule_time_reloads_v18_dp_reference(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedules_seen: list[float] = []

    def fake_load_dp_reference(self: ORS, **kwargs):
        schedules_seen.append(float(kwargs["schedule_time_s"]))
        return _fake_dp_reference_manifold(self, **kwargs)

    monkeypatch.setattr(
        ORS,
        "load_or_build_ref_redundant_operation_time_from_dp",
        fake_load_dp_reference,
    )

    env = _build_env_like(mtto_env)
    env.change_schedule_time(445.0)

    assert schedules_seen == pytest.approx([440.0, 445.0])
    assert env.train_service.schedule_time == pytest.approx(445.0)


def test_punctuality_disabled_skips_v18_dp_reference_loading(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loads_dp_reference(self: ORS, **_kwargs):
        raise AssertionError("DP reference should not be loaded")

    monkeypatch.setattr(
        ORS,
        "load_or_build_ref_redundant_operation_time_from_dp",
        fail_if_loads_dp_reference,
    )

    env = _build_env_like(
        mtto_env,
        reward_config=RewardConfig(enable_potential_punctuality=False),
    )
    env.change_schedule_time(445.0)

    assert env.ref_redundant_operation_time_pos_arr.size == 0
    assert env.ref_redundant_operation_time_arr.size == 0
    assert env.train_service.schedule_time == pytest.approx(445.0)


def test_make_env_passes_punctuality_reference_options(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured_kwargs = {}

    def fake_env(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(env_factory, "MTTOEnv", fake_env)

    result = env_factory.make_env(
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
        safeguard_utility=mtto_env.safeguard_utility,
        train_service=_clone_train_service(mtto_env.train_service),
        gamma=mtto_env.gamma,
        max_step_distance=mtto_env.max_step_distance,
        punctuality_dp_curve_dir=tmp_path,
        punctuality_reference_match_tolerance=0.5,
    )

    assert result is not None
    assert captured_kwargs["punctuality_dp_curve_dir"] == tmp_path
    assert captured_kwargs["punctuality_reference_match_tolerance"] == pytest.approx(
        0.5
    )


def test_punctuality_reward_depends_on_dp_speed_error(mtto_env: MTTOEnv):
    mtto_env.reset()

    mtto_env.current_position = mtto_env.train_service.start_position + 5000.0
    mtto_env.current_operation_time = 120.0
    mtto_env.last_state["pos"] = mtto_env.current_position - 100.0
    mtto_env.last_state["operation_time"] = 118.0

    mtto_env.current_speed = 5.0
    mtto_env.last_state["speed"] = 8.0
    reward_low_speed = mtto_env._get_reward_punctuality_dense()

    mtto_env.current_speed = 25.0
    mtto_env.last_state["speed"] = 30.0
    reward_high_speed = mtto_env._get_reward_punctuality_dense()

    assert reward_low_speed != pytest.approx(reward_high_speed)


def test_whole_env(mtto_env: MTTOEnv):
    check_env(mtto_env)


def test_step_without_diagnostics_keeps_tb_dicts_empty(mtto_env: MTTOEnv):
    mtto_env.enable_diagnostics = False
    try:
        mtto_env.reset()
        action = mtto_env.action_space.sample()
        _, _, _, _, info = mtto_env.step(action)

        assert mtto_env.rewards_info == {}
        assert mtto_env.state_info == {}
        assert mtto_env.constraint_info == {}
        assert mtto_env.event_info == {}

        expected_runtime_keys = {
            "energy_consumption",
            "operation_time",
            "redundant_operation_time",
            "position",
            "stopping_point_index",
        }
        runtime_namespace = "basic" if "basic" in info else "runtime"
        assert runtime_namespace in info
        runtime = info[runtime_namespace]
        assert isinstance(runtime, dict)
        assert expected_runtime_keys.issubset(set(runtime.keys()))
        assert expected_runtime_keys.issubset(set(mtto_env.basic_info.keys()))
        assert "rewards" not in info
        assert "state" not in info
        assert "constraint" not in info
        assert "event" not in info
        assert "tb_diagnostics" not in info
    finally:
        mtto_env.enable_diagnostics = True


def test_step_with_diagnostics_puts_namespaces_at_info_top_level(
    mtto_env: MTTOEnv,
):
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1

    mtto_env.reset()
    action = mtto_env.action_space.sample()
    _, _, _, _, info = mtto_env.step(action)

    assert "tb_diagnostics" not in info
    runtime_namespace = "basic" if "basic" in info else "runtime"
    assert runtime_namespace in info
    assert "rewards" in info
    assert "state" in info
    assert "constraint" in info
    assert "event" in info

    runtime = info[runtime_namespace]
    assert isinstance(runtime, dict)
    assert "position" in runtime
    assert "stopping_position" not in runtime

    expected_runtime_keys = {
        "energy_consumption",
        "operation_time",
        "position",
        "stopping_point_index",
    }
    assert expected_runtime_keys.issubset(set(runtime.keys()))


def test_no_trajectory_tracking_data_when_disabled(mtto_env: MTTOEnv):
    assert mtto_env.render_mode is None
    assert mtto_env.enable_trajectory_tracking is False

    mtto_env.reset()
    action = mtto_env.action_space.sample()
    mtto_env.step(action)

    assert mtto_env.trajectory_pos is None
    assert mtto_env.trajectory_speed_mps is None


def test_trajectory_tracking_can_be_enabled_without_rendering(mtto_env: MTTOEnv):
    mtto_env.enable_trajectory_tracking = True
    try:
        mtto_env.reset()
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 1
        assert len(mtto_env.trajectory_speed_mps) == 1

        action = np.asarray([1.0], dtype=np.float32)
        mtto_env.step(action)

        assert mtto_env.render_mode is None
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 2
        assert len(mtto_env.trajectory_speed_mps) == 2
        assert mtto_env.trajectory_pos[-1] == pytest.approx(
            float(mtto_env.current_position)
        )
        assert mtto_env.trajectory_speed_mps[-1] == pytest.approx(
            abs(float(mtto_env.current_speed))
        )
    finally:
        mtto_env.enable_trajectory_tracking = False
        mtto_env.reset()


def _patch_step_dependencies_for_outcome_tests(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    *,
    next_speed: float,
) -> None:
    monkeypatch.setattr(mtto_env, "_update_motion", lambda: (next_speed, 0.0, 0.0))
    monkeypatch.setattr(mtto_env.ecc, "calc_energy", lambda **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(mtto_env, "_get_upper_speed", lambda _pos: 10_000.0)
    monkeypatch.setattr(
        mtto_env.safeguard_utility,
        "get_min_and_max_speed",
        lambda **_kwargs: (0.0, 10_000.0),
    )
    monkeypatch.setattr(
        mtto_env.sps,
        "step_to_next_stopping_point",
        lambda **_kwargs: -1,
    )


def test_step_failed_stop_is_truncated_with_fixed_penalty(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1
    mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)

    _, reward, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is False
    assert truncated is True
    assert reward == pytest.approx(-10.0)
    assert "constraint" in info
    assert info["constraint"]["violation_code"] == MTTOEnv.VIOLATION_CODE_FAILED_STOP


def test_step_success_is_terminated_without_truncation(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1
    mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)
    mtto_env.current_position = mtto_env.train_service.target_position
    mtto_env.current_speed = 0.0
    mtto_env.current_operation_time = mtto_env.train_service.schedule_time

    _, _, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert "constraint" in info
    assert info["constraint"]["violation_code"] == MTTOEnv.VIOLATION_CODE_ONGOING


def test_potential_punctuality_v3_is_finite_for_extreme_negative_input(
    mtto_env: MTTOEnv,
) -> None:
    val = mtto_env._potential_punctuality_v3(-100.0)
    assert np.isfinite(val)


def test_potential_punctuality_v4_peaks_at_zero_time_error(
    mtto_env: MTTOEnv,
) -> None:
    peak = mtto_env._potential_punctuality_v4(0.0)
    early = mtto_env._potential_punctuality_v4(0.1)
    late = mtto_env._potential_punctuality_v4(-0.1)

    assert peak > early
    assert peak > late
    assert late < early


def test_potential_punctuality_v4_late_side_drops_monotonically(
    mtto_env: MTTOEnv,
) -> None:
    slightly_late = mtto_env._potential_punctuality_v4(-0.05)
    clearly_late = mtto_env._potential_punctuality_v4(-0.2)

    assert clearly_late < slightly_late


def test_potential_punctuality_v4_is_finite_for_extreme_input(
    mtto_env: MTTOEnv,
) -> None:
    assert np.isfinite(mtto_env._potential_punctuality_v4(-100.0))
    assert np.isfinite(mtto_env._potential_punctuality_v4(100.0))


def test_potential_punctuality_v10_negative_redundancy_decay_grows_with_time(
    mtto_env: MTTOEnv,
) -> None:
    early = mtto_env._potential_punctuality_v10(
        redundant_operation_time=-10.0,
        operation_time=100.0,
    )
    late = mtto_env._potential_punctuality_v10(
        redundant_operation_time=-10.0,
        operation_time=300.0,
    )

    assert late < early


def test_potential_punctuality_v10_warning_band_does_not_use_time_decay(
    mtto_env: MTTOEnv,
) -> None:
    early = mtto_env._potential_punctuality_v10(
        redundant_operation_time=2.5,
        operation_time=100.0,
    )
    late = mtto_env._potential_punctuality_v10(
        redundant_operation_time=2.5,
        operation_time=300.0,
    )

    np.testing.assert_allclose(early, late)


def test_potential_punctuality_v10_positive_side_does_not_use_time_decay(
    mtto_env: MTTOEnv,
) -> None:
    early = mtto_env._potential_punctuality_v10(
        redundant_operation_time=10.0,
        operation_time=100.0,
    )
    late = mtto_env._potential_punctuality_v10(
        redundant_operation_time=10.0,
        operation_time=300.0,
    )

    np.testing.assert_allclose(early, late)


def test_punctuality_dense_reward_is_stable_for_extreme_time_redundancy(
    mtto_env: MTTOEnv,
) -> None:
    mtto_env.reset()
    mtto_env.current_redundant_operation_time = -100.0
    mtto_env.last_state["time_redundancy"] = 0.0
    val = mtto_env._get_reward_punctuality_dense()

    assert np.isfinite(val)
    assert abs(val) < 1e4
