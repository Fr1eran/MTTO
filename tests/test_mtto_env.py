from dataclasses import replace
from typing import TypedDict, Unpack, cast

import numpy as np
import pytest
from gymnasium.utils.env_checker import (
    check_env,
)
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo, get_slope_scalar_numba
from model.vehicle import VehicleInfo, calc_levi_deceleration_scalar_numba
from rl.context_pool import Context
from rl.context_sampler import ContextSampler
from rl.dspdl import DSPDLEpisodeAccumulator
from rl.evaluation import evaluate_operational_policy_once, evaluate_policy_once
from rl.mtto_env import MTTOEnv
from rl.observation_builder import ObservationBuilder
from rl.operational_state import OperationalState, OperationalTransition, ViolationCode
from rl.reward_calculator import RewardConfig
from rl.reward_diagnostics import RewardDiagnosticsAccumulator
from rl.safety_statistics import SafetyTruncationBuffer
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)


class _MTTOEnvOverrides(TypedDict, total=False):
    context_sampler: ContextSampler | None
    dspdl_accumulator: DSPDLEpisodeAccumulator | None
    enable_trajectory_tracking: bool
    safety_truncation_buffer: SafetyTruncationBuffer | None


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
        step_distance=10.0,
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
    **kwargs: Unpack[_MTTOEnvOverrides],
) -> MTTOEnv:
    return MTTOEnv(
        vehicle=source_env.vehicle,
        track=source_env.track,
        safeguard_utility=source_env.safeguard_utility,
        train_service=train_service or _clone_train_service(source_env.train_service),
        gamma=source_env.gamma,
        step_distance=source_env.step_distance,
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
        obs[3],
        mtto_env.observation_builder.normalize_acc_to_action(
            mtto_env.observation_builder.calc_coasting_acc(mtto_env.state)
        ),
    )  # suggested_dec_normalized
    np.testing.assert_allclose(obs[4], 1.0)  # remaining_schedule_time
    np.testing.assert_allclose(
        obs[5],
        mtto_env.state.redundant_operation_time_s
        / mtto_env.train_service.schedule_time,
    )  # time_redundancy
    np.testing.assert_allclose(
        obs[6],
        mtto_env.state.max_speed_mps / mtto_env.vehicle.max_speed,
    )  # current_max_speed
    np.testing.assert_allclose(
        obs[7],
        mtto_env.state.min_speed_mps / mtto_env.vehicle.max_speed,
    )  # current_min_speed
    np.testing.assert_allclose(obs[8], 0.0)  # current_slope
    np.testing.assert_allclose(
        obs[9],
        mtto_env.observation_builder.get_lookahead_avg_slope(
            mtto_env.state.step_count
        )
        / mtto_env.vehicle.max_slope_capacity,
    )  # lookahead_avg_slope
    np.testing.assert_allclose(
        obs[10],
        mtto_env.observation_builder.get_lookahead_avg_upper_speed(
            mtto_env.state.step_count
        )
        / mtto_env.vehicle.max_speed,
    )  # lookahead_avg_upper_speed
    np.testing.assert_allclose(obs[11], 0.0)  # approach_progress
    assert info == {}


class _ContextSamplerStub:
    def __init__(self, initial_state: OperationalState) -> None:
        self.initial_state = initial_state
        self.version = 0
        self.reseeded_with: int | None = None
        self.updated: tuple[object, int] | None = None

    def reseed(self, seed: int) -> None:
        self.reseeded_with = seed

    def sample(self) -> Context:
        return Context(
            context_index=2,
            remaining_distance_m=100.0,
            initial_state=self.initial_state,
        )

    def update_distribution(
        self, weights: NDArray[np.floating] | list[float], *, version: int
    ) -> None:
        self.updated = (np.asarray(weights, dtype=np.float64), version)
        self.version = version


def test_reset_can_sample_context_and_collect_dspdl_statistics(
    mtto_env: MTTOEnv,
) -> None:
    reference_state = replace(
        mtto_env.stepper.reset(),
        position_m=mtto_env.train_service.start_position + 100.0,
        operation_time_s=4.0,
        energy_consumption_kj=12.0,
        step_count=3,
    )
    sampler = _ContextSamplerStub(reference_state)
    accumulator = DSPDLEpisodeAccumulator(context_count=3, gamma=mtto_env.gamma)
    env = _build_env_like(
        mtto_env,
        context_sampler=cast(ContextSampler, cast(object, sampler)),
        dspdl_accumulator=accumulator,
        enable_trajectory_tracking=True,
    )
    env._comfort_tav = 8.0

    observation, info = env.reset(seed=123)

    assert info == {}
    assert sampler.reseeded_with == 123
    assert env.state is reference_state
    assert env._comfort_tav == 0.0
    assert env.trajectory_pos == [reference_state.position_m]
    assert observation[0] < 1.0
    _, _, _, _, info = env.step(np.asarray([0.0], dtype=np.float32))
    assert "reference_context_sample_id" not in info
    assert "reference_context_index" not in info
    assert "reference_context_distribution_version" not in info

    statistics = env.drain_dspdl_statistics(version=0)
    np.testing.assert_array_equal(statistics["context_counts"], [0, 0, 1])

    env.set_dspdl_distribution([1.0], version=1)
    assert sampler.updated is not None
    np.testing.assert_array_equal(sampler.updated[0], [1.0])
    assert sampler.updated[1] == 1
    assert accumulator.accepted_version == 1


@pytest.mark.parametrize(
    "action, expected_acceleration",
    [(-1.0, "max_dec"), (0.0, None), (1.0, "max_acc")],
)
def test_observation_builder_denormalizes_actions(
    mtto_env: MTTOEnv,
    action: float,
    expected_acceleration: str | None,
) -> None:
    actual = mtto_env.observation_builder.denormalize_action(action)
    expected = (
        (mtto_env.vehicle.max_acc + mtto_env.vehicle.max_dec) / 2.0
        if expected_acceleration is None
        else getattr(mtto_env.vehicle, expected_acceleration)
    )

    assert actual == pytest.approx(expected)
    assert mtto_env.observation_builder.normalize_acc_to_action(
        actual
    ) == pytest.approx(action)


@pytest.mark.parametrize(
    "member",
    [
        "current_position",
        "current_speed",
        "current_acc",
        "current_operation_time",
        "current_redundant_operation_time",
        "current_energy_consumption",
        "current_slope",
        "current_min_speed",
        "current_max_speed",
        "current_stopping_point_index",
        "current_steps",
        "stop_error",
        "sps",
        "ors",
        "ecc",
        "whole_distance",
        "direction",
        "max_episode_steps",
        "required_episode_steps",
        "max_energy_consumption",
        "min_operation_time",
        "max_redundant_operation_time",
        "_upper_speed_lut_pos_min",
        "_upper_speed_lut_step",
        "_upper_speed_lut_speed_arr",
        "_get_action_denormalized",
        "_update_motion",
        "_get_upper_speed",
        "_get_upper_speed_or_zero",
        "_normalize_acc_to_action",
        "_calc_coasting_acc",
        "_calc_lookahead_avg_slope",
        "_calc_lookahead_avg_upper_speed",
        "_calc_approach_progress",
        "_get_reward_goal",
        "_get_obs",
        "_last_violation_code",
    ],
)
def test_mtto_env_does_not_expose_removed_compatibility_members(
    mtto_env: MTTOEnv,
    member: str,
) -> None:
    assert not hasattr(mtto_env, member)


@pytest.mark.parametrize(
    "member",
    [
        "VIOLATION_CODE_ONGOING",
        "VIOLATION_CODE_FAILED_STOP",
        "VIOLATION_CODE_SPEED_LOW",
        "VIOLATION_CODE_SPEED_HIGH",
        "VIOLATION_CODE_STEP_LIMIT",
    ],
)
def test_mtto_env_does_not_expose_legacy_violation_constants(member: str) -> None:
    assert not hasattr(MTTOEnv, member)


def test_suggested_dec_uses_coasting_acc_outside_final_approach(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    mtto_env.state = replace(mtto_env.state, speed_mps=30.0, slope_permille=1.0)

    obs = mtto_env.observation_builder.build(mtto_env.state)

    coasting_dec = calc_levi_deceleration_scalar_numba(
        mtto_env.state.speed_mps,
        mtto_env.state.slope_permille,
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
        mtto_env.observation_builder.normalize_acc_to_action(expected_coasting_acc),
    )


def test_suggested_dec_uses_required_stop_dec_in_final_approach(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    mtto_env.state = replace(
        mtto_env.state,
        position_m=mtto_env.train_service.target_position - 1000.0,
        speed_mps=20.0,
        slope_permille=1.0,
    )

    obs = mtto_env.observation_builder.build(mtto_env.state)

    required_dec = -(mtto_env.state.speed_mps**2) / (2.0 * 1000.0)
    np.testing.assert_allclose(
        obs[3],
        mtto_env.observation_builder.normalize_acc_to_action(required_dec),
    )
    np.testing.assert_allclose(
        obs[11],
        mtto_env.observation_builder.calc_approach_progress(1000.0),
    )  # approach_progress


def test_get_upper_speed_or_zero_returns_zero_outside_lut(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    monkeypatch.setattr(mtto_env.stepper, "_upper_speed_lut_pos_min", 100.0)
    monkeypatch.setattr(mtto_env.stepper, "_upper_speed_lut_step", 10.0)
    monkeypatch.setattr(
        mtto_env.stepper,
        "_upper_speed_lut_speed_arr",
        np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
    )

    assert mtto_env.stepper.get_upper_speed_or_zero(99.0) == pytest.approx(0.0)
    assert mtto_env.stepper.get_upper_speed_or_zero(110.0) == pytest.approx(10.0)
    assert mtto_env.stepper.get_upper_speed_or_zero(121.0) == pytest.approx(0.0)


def test_lookahead_cache_matches_window_average(mtto_env: MTTOEnv) -> None:
    _ = mtto_env.reset()
    builder = mtto_env.observation_builder
    step_index = 3
    position_m = (
        mtto_env.train_service.start_position
        + mtto_env.stepper.direction * step_index * mtto_env.stepper.step_distance_m
    )
    offsets = np.linspace(
        builder.step_distance_m,
        builder.lookahead_distance_m,
        builder.lookahead_num_samples,
    )
    expected_slope = np.mean(
        [
            get_slope_scalar_numba(
                position_m + mtto_env.stepper.direction * float(offset),
                mtto_env.track.slopes,
                mtto_env.track.slope_intervals,
            )
            for offset in offsets
        ]
    )
    expected_upper_speed = np.mean(
        [
            mtto_env.stepper.get_upper_speed_or_zero(
                position_m + mtto_env.stepper.direction * float(offset)
            )
            for offset in offsets
        ]
    )

    np.testing.assert_allclose(
        builder.get_lookahead_avg_slope(step_index), expected_slope
    )
    np.testing.assert_allclose(
        builder.get_lookahead_avg_upper_speed(step_index), expected_upper_speed
    )


def test_lookahead_features_are_read_only_and_not_recomputed_during_build(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    builder = mtto_env.observation_builder

    def fail(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("lookahead feature was recomputed during build")

    monkeypatch.setattr("rl.observation_builder.get_slope_scalar_numba", fail)
    monkeypatch.setattr(builder, "_get_upper_speed_or_zero", fail)

    observation = builder.build(mtto_env.state)

    assert observation.shape == (12,)
    assert builder._lookahead_avg_slope_by_step.flags.writeable is False
    assert builder._lookahead_avg_upper_speed_by_step.flags.writeable is False


@pytest.mark.parametrize("step_index", (-1, 10**9))
def test_lookahead_cache_rejects_out_of_range_index(
    mtto_env: MTTOEnv,
    step_index: int,
) -> None:
    with pytest.raises(IndexError, match="cached lookahead range"):
        _ = mtto_env.observation_builder.get_lookahead_avg_slope(step_index)


def test_lookahead_cache_rejects_noninteger_index(mtto_env: MTTOEnv) -> None:
    with pytest.raises(TypeError, match="step_index must be an integer"):
        _ = mtto_env.observation_builder.get_lookahead_avg_upper_speed(  # type: ignore[arg-type]
            1.5
        )


def test_reverse_direction_lookahead_cache_uses_step_index_grid() -> None:
    vehicle = VehicleInfo(mass=100.0, numoftrainsets=1, length=10.0)
    track = TrackInfo(
        slopes=np.asarray([1.0, 3.0]),
        slope_intervals=np.asarray([0.0, 50.0]),
        speed_limits=np.asarray([20.0]),
        speed_limit_intervals=np.asarray([0.0, 100.0]),
    )
    service = TrainService(100.0, 0.0, 0.0, 10.0, 0.75, 0.05, 1.0)
    upper_speed_query_count = 0

    def get_upper_speed(position_m: float) -> float:
        nonlocal upper_speed_query_count
        upper_speed_query_count += 1
        return max(position_m, 0.0)

    builder = ObservationBuilder(
        vehicle=vehicle,
        track=track,
        train_service=service,
        step_distance_m=10.0,
        direction=-1,
        whole_distance_m=100.0,
        get_upper_speed_or_zero=get_upper_speed,
    )
    expected_node_count = 11
    expected_query_count = expected_node_count * builder.lookahead_num_samples

    assert builder._lookahead_avg_slope_by_step.size == expected_node_count
    assert upper_speed_query_count == expected_query_count
    assert builder.get_lookahead_avg_upper_speed(0) == pytest.approx(9.0)
    assert builder.get_lookahead_avg_slope(0) == pytest.approx(1.2)
    assert upper_speed_query_count == expected_query_count


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    mec1, lec1 = mtto_env.stepper.ecc.calc_energy(
        begin_pos=mtto_env.state.position_m,
        begin_speed=mtto_env.state.speed_mps,
        acc=0.0,
        distance=0.0,
        direction=mtto_env.stepper.direction,
        operation_time=0.0,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = mtto_env.stepper.ecc.calc_energy(
        begin_pos=mtto_env.state.position_m,
        begin_speed=mtto_env.state.speed_mps,
        acc=1.0,
        distance=100.0,
        direction=mtto_env.stepper.direction,
        operation_time=14.142,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption2 = mec2 + lec2

    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_whole_env(mtto_env: MTTOEnv):
    check_env(mtto_env)


def test_step_info_excludes_reward_diagnostics(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    action = mtto_env.action_space.sample()
    _, _, terminated, truncated, info = mtto_env.step(action)

    assert "basic" in info
    assert "outcome" in info
    assert info["outcome"] == {
        "terminated": terminated,
        "truncated": truncated,
    }
    assert "rewards" not in info


def test_compact_training_info_is_empty_and_worker_accumulator_records(
    mtto_env: MTTOEnv,
):
    mtto_env.reward_diagnostics_accumulator = RewardDiagnosticsAccumulator(
        worker_rank=0, rollout_capacity=2
    )
    mtto_env.compact_training_info = True
    try:
        _ = mtto_env.reset()
        action = mtto_env.action_space.sample()
        _, _, _, _, info = mtto_env.step(action)
        assert info == {}
        batch = mtto_env.drain_reward_diagnostics()
        np.testing.assert_array_equal(batch["transition_count"], [1])
        assert batch["reward_sum"].shape == (9,)
    finally:
        mtto_env.compact_training_info = False
        mtto_env.reward_diagnostics_accumulator = None


def test_no_trajectory_tracking_data_when_disabled(mtto_env: MTTOEnv):
    assert mtto_env.render_mode is None
    assert mtto_env.enable_trajectory_tracking is False

    _ = mtto_env.reset()
    action = mtto_env.action_space.sample()
    _ = mtto_env.step(action)

    assert mtto_env.trajectory_pos is None
    assert mtto_env.trajectory_speed_mps is None


def test_trajectory_tracking_can_be_enabled_without_rendering(mtto_env: MTTOEnv):
    mtto_env.enable_trajectory_tracking = True
    try:
        _ = mtto_env.reset()
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 1
        assert len(mtto_env.trajectory_speed_mps) == 1

        action = np.asarray([1.0], dtype=np.float32)
        _ = mtto_env.step(action)

        assert mtto_env.render_mode is None
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 2
        assert len(mtto_env.trajectory_speed_mps) == 2
        assert mtto_env.trajectory_pos[-1] == pytest.approx(
            float(mtto_env.state.position_m)
        )
        assert mtto_env.trajectory_speed_mps[-1] == pytest.approx(
            abs(float(mtto_env.state.speed_mps))
        )
    finally:
        mtto_env.enable_trajectory_tracking = False
        _ = mtto_env.reset()


def _patch_step_dependencies_for_outcome_tests(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    *,
    next_speed: float,
) -> None:
    def _advance(state: OperationalState, acceleration: float):
        next_state = replace(
            state,
            speed_mps=next_speed,
            acceleration_mps2=acceleration,
            stop_error_m=abs(mtto_env.train_service.target_position - state.position_m),
            step_count=state.step_count + 1,
        )
        success = next_speed == 0.0 and next_state.stop_error_m <= 9.0
        failed_stop = next_speed == 0.0 and not success
        return OperationalTransition(
            state,
            next_state,
            acceleration,
            0.0,
            0.0,
            0.0,
            success,
            failed_stop,
            ViolationCode.ONGOING if success else ViolationCode.FAILED_STOP,
        )

    monkeypatch.setattr(mtto_env.stepper, "advance", _advance)


def test_step_failed_stop_is_truncated_with_fixed_penalty(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)

    _, reward, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is False
    assert truncated is True
    assert reward == pytest.approx(-2.0)
    assert info["outcome"] == {"terminated": False, "truncated": True}
    assert "constraint" not in info


def test_step_buffers_speed_safety_truncation_inside_worker(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = SafetyTruncationBuffer()
    env = _build_env_like(mtto_env, safety_truncation_buffer=buffer)
    _ = env.reset()

    def _advance(state: OperationalState, acceleration: float):
        next_state = replace(
            state,
            position_m=1234.0,
            acceleration_mps2=acceleration,
            step_count=state.step_count + 1,
        )
        return OperationalTransition(
            state,
            next_state,
            acceleration,
            0.0,
            0.0,
            0.0,
            False,
            True,
            ViolationCode.SPEED_LOW,
        )

    monkeypatch.setattr(env.stepper, "advance", _advance)

    _, _, _, truncated, info = env.step(np.asarray([0.0], dtype=np.float32))
    batch = env.drain_safety_truncations()

    assert truncated is True
    assert "safety" not in info
    np.testing.assert_allclose(batch["position_m"], [1234.0])
    np.testing.assert_array_equal(batch["violation_code"], [2])
    assert env.drain_safety_truncations()["position_m"].size == 0


def test_step_success_is_terminated_without_truncation(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)
    mtto_env.state = replace(
        mtto_env.state,
        position_m=mtto_env.train_service.target_position,
        speed_mps=0.0,
        operation_time_s=mtto_env.train_service.schedule_time,
        stop_error_m=0.0,
    )

    _, _, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert info["outcome"] == {"terminated": True, "truncated": False}
    assert "constraint" not in info


def test_stepper_truncates_at_required_transition_budget(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stepper = mtto_env.stepper
    state = replace(
        stepper.reset(),
        speed_mps=10.0,
        step_count=stepper.required_episode_steps - 1,
    )

    def _build_state(**kwargs: object):
        return replace(
            state,
            position_m=kwargs["position_m"],
            speed_mps=10.0,
            acceleration_mps2=kwargs["acceleration_mps2"],
            operation_time_s=kwargs["operation_time_s"],
            energy_consumption_kj=kwargs["energy_consumption_kj"],
            step_count=kwargs["step_count"],
            sps_state=kwargs["sps_state"],
            min_speed_mps=0.0,
            max_speed_mps=100.0,
            stop_error_m=10.0,
        )

    monkeypatch.setattr(stepper, "_build_state", _build_state)
    transition = stepper.advance(state, 0.0)

    assert transition.next_state.step_count == stepper.required_episode_steps
    assert transition.terminated is False
    assert transition.truncated is True
    assert transition.violation_code is ViolationCode.STEP_LIMIT


def test_stepper_allows_success_on_required_transition_budget(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stepper = mtto_env.stepper
    state = replace(
        stepper.reset(),
        step_count=stepper.required_episode_steps - 1,
    )

    def _build_state(**kwargs: object):
        return replace(
            state,
            position_m=mtto_env.train_service.target_position,
            speed_mps=0.0,
            acceleration_mps2=kwargs["acceleration_mps2"],
            operation_time_s=kwargs["operation_time_s"],
            energy_consumption_kj=kwargs["energy_consumption_kj"],
            step_count=kwargs["step_count"],
            sps_state=kwargs["sps_state"],
            min_speed_mps=0.0,
            max_speed_mps=100.0,
            stop_error_m=0.0,
        )

    monkeypatch.setattr(stepper, "_build_state", _build_state)
    transition = stepper.advance(state, 0.0)

    assert transition.next_state.step_count == stepper.required_episode_steps
    assert transition.terminated is True
    assert transition.truncated is False
    assert transition.violation_code is ViolationCode.ONGOING


def test_stepper_custom_requested_distance_matches_default_at_maximum(
    mtto_env: MTTOEnv,
) -> None:
    stepper = mtto_env.stepper
    state = replace(stepper.reset(), speed_mps=10.0)

    default_transition = stepper.advance(state, 0.0)
    explicit_transition = stepper.advance(
        state,
        0.0,
        requested_distance_m=stepper.step_distance_m,
    )

    assert explicit_transition.distance_m == pytest.approx(
        default_transition.distance_m
    )
    assert explicit_transition.duration_s == pytest.approx(
        default_transition.duration_s
    )
    assert explicit_transition.next_state.position_m == pytest.approx(
        default_transition.next_state.position_m
    )
    assert explicit_transition.energy_delta_kj == pytest.approx(
        default_transition.energy_delta_kj
    )


def test_stepper_supports_short_reference_replay_segment(mtto_env: MTTOEnv) -> None:
    stepper = mtto_env.stepper
    state = replace(stepper.reset(), speed_mps=10.0)

    transition = stepper.advance(state, 0.0, requested_distance_m=2.5)

    assert transition.distance_m == pytest.approx(2.5)
    assert transition.duration_s == pytest.approx(0.25)
    assert transition.next_state.position_m == pytest.approx(state.position_m + 2.5)


@pytest.mark.parametrize("requested_distance_m", [0.0, -1.0, float("inf"), 11.0])
def test_stepper_rejects_invalid_requested_distance(
    mtto_env: MTTOEnv,
    requested_distance_m: float,
) -> None:
    stepper = mtto_env.stepper
    with pytest.raises(ValueError, match="requested_distance_m"):
        _ = stepper.advance(
            stepper.reset(),
            0.0,
            requested_distance_m=requested_distance_m,
        )


def test_terminal_punctuality_reward_favors_smaller_time_error(
    mtto_env: MTTOEnv,
) -> None:
    terminal_state = replace(
        mtto_env.state,
        position_m=mtto_env.train_service.target_position,
        speed_mps=0.0,
        stop_error_m=0.0,
        operation_time_s=mtto_env.train_service.schedule_time,
    )
    punctual_reward = mtto_env.reward_calculator.calculate(
        OperationalTransition(
            mtto_env.state,
            terminal_state,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ViolationCode.ONGOING,
        )
    )
    late_state = replace(
        terminal_state,
        operation_time_s=terminal_state.operation_time_s + 60.0,
    )
    late_reward = mtto_env.reward_calculator.calculate(
        OperationalTransition(
            terminal_state,
            late_state,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ViolationCode.ONGOING,
        )
    )

    assert punctual_reward.terminal_punctuality > late_reward.terminal_punctuality


def test_change_schedule_time_updates_time_context_without_dp_reference(
    mtto_env: MTTOEnv,
) -> None:
    _ = mtto_env.reset()
    observation = mtto_env.change_schedule_time(445.0)

    assert mtto_env.train_service.schedule_time == pytest.approx(445.0)
    assert observation.shape == mtto_env.observation_space.shape


def test_external_rollout_uses_same_transition_and_reward_path(
    mtto_env: MTTOEnv,
) -> None:
    action = np.asarray([-1.0], dtype=np.float32)
    _ = mtto_env.reset()
    _, env_reward, env_terminated, env_truncated, _ = mtto_env.step(action)

    def _policy(_obs: object) -> np.ndarray:
        return action

    result = evaluate_operational_policy_once(
        _policy,
        stepper=mtto_env.stepper,
        reward_calculator=mtto_env.reward_calculator,
        observation_builder=mtto_env.observation_builder,
    )

    # From standstill, maximum braking produces the same immediate failed-stop
    # truncation in both execution paths.
    assert result.episode_steps == 1
    assert result.total_reward == pytest.approx(env_reward)
    assert result.terminated is env_terminated
    assert result.truncated is env_truncated
    assert result.success is False


def test_gym_evaluation_uses_terminal_flags_without_persisting_violation_code(
    mtto_env: MTTOEnv,
) -> None:
    class BrakingPolicy:
        def predict(self, _obs: np.ndarray, deterministic: bool = True):
            del deterministic
            return np.asarray([-1.0], dtype=np.float32), None

    result = evaluate_policy_once(BrakingPolicy(), mtto_env)

    assert result.terminated is False
    assert result.truncated is True
    assert result.success is False
    assert "violation_code" not in result.to_metrics()
    assert "terminated" not in result.to_metrics()
    assert "truncated" not in result.to_metrics()
