import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import segment_energy
from mtto.domain.kinematics import accel_between
from mtto.domain.line import Line
from mtto.domain.safeguard import Safeguard
from mtto.domain.scenario import EnergyParams, Scenario, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.domain.srtsp import build_srtsp_curve
from mtto.dp.cache import (
    load_transition_graph_from_disk,
    make_cache_folder_name,
    save_transition_graph_to_disk,
    validate_transition_graph,
)
from mtto.dp.graph import (
    generate_uniform_spacing_stages,
    get_stage_speed_upper_indices,
)
from mtto.dp.solver import (
    VariableSpacingDPOptimizer,
    _DPSolution,
)
from paper.figures import load_paper_scenario
from tests.golden.cases import DP_CASES
from tests.golden.drive import build_dp_optimizer


def test_dp_optimizer_constructor_does_not_accept_max_speed() -> None:
    parameters = inspect.signature(VariableSpacingDPOptimizer).parameters

    assert "max_speed" not in parameters


def test_dp_inner_result_includes_cumulative_time_from_policy() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    cache = {
        "stages": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
        "speed_states": np.asarray([0.0, 1.0], dtype=np.float64),
        "stage_speed_upper_idx": np.asarray([0, 1, 0], dtype=int),
        "transitions": [
            [
                (
                    np.asarray([1], dtype=int),
                    np.asarray([5.0], dtype=np.float64),
                    np.asarray([2.0], dtype=np.float64),
                ),
                None,
            ],
            [
                None,
                (
                    np.asarray([0], dtype=int),
                    np.asarray([7.0], dtype=np.float64),
                    np.asarray([3.0], dtype=np.float64),
                ),
            ],
        ],
        "total_valid_edges": 2,
    }

    result = optimizer._solve_dp_inner(
        cache=cache,
        lambda_time=10.0,
        start_state_idx=0,
        target_state_idx=0,
    )

    assert result is not None
    np.testing.assert_allclose(result.cum_time_s, np.asarray([0.0, 2.0, 5.0]))
    assert result.cum_time_s[0] == pytest.approx(0.0)
    assert result.cum_time_s[-1] == pytest.approx(5.0)
    assert np.all(np.diff(result.cum_time_s) >= 0.0)


def test_stage_speed_upper_indices_include_task_upper_curve() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.speed_grid_upper_mps = 20.0
    optimizer.vehicle = cast(Vehicle, cast(object, SimpleNamespace(max_speed=20.0)))
    optimizer.safeguard = cast(
        Safeguard,
        cast(
            object,
            SimpleNamespace(
                speed_limits=np.asarray([20.0], dtype=np.float64),
                speed_limit_intervals=np.asarray([0.0, 10.0], dtype=np.float64),
                params=SimpleNamespace(factor=1.0),
            ),
        ),
    )
    optimizer.srtsp_curve = build_srtsp_curve(
        [0.0, 5.0, 10.0],
        [0.0, 4.4, 10.0],
    )

    upper = get_stage_speed_upper_indices(
        safeguard=optimizer.safeguard,
        srtsp_curve=optimizer.srtsp_curve,
        speed_grid_upper_mps=optimizer.speed_grid_upper_mps,
        vehicle_max_speed=float(optimizer.vehicle.max_speed),
        stages=np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
        speed_states=np.arange(21.0, dtype=np.float64),
    )

    np.testing.assert_array_equal(upper, np.asarray([0, 4, 10]))


def test_dp_upper_curve_uses_operational_stepper_task_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def _fake_min_operation_time_curve(**kwargs: object):
        observed.update(kwargs)
        return (
            np.asarray([135.0, 10000.0, 29276.0], dtype=np.float64),
            np.asarray([0.0, 35.0, 0.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        "mtto.dp.solver.min_operation_time_curve", _fake_min_operation_time_curve
    )
    vehicle = cast(Vehicle, cast(object, SimpleNamespace(max_speed=40.0)))
    line = cast(Line, object())
    safeguard = cast(
        Safeguard,
        cast(
            object,
            SimpleNamespace(
                params=SimpleNamespace(factor=0.99),
                speed_limits=np.asarray([30.0], dtype=np.float64),
            ),
        ),
    )
    scenario = cast(
        Scenario,
        cast(
            object,
            SimpleNamespace(
                vehicle=vehicle,
                line=line,
                safeguard=safeguard,
                energy=EnergyParams(
                    R_m=0.2796,
                    L_d=0.00292,
                    R_k=0.0736,
                    L_k=0.000142,
                    Tau=0.258,
                    Psi_fd=3.9629,
                    k_c=0.5,
                    Phi_1=0.1049,
                    Phi_2=1.006,
                ),
            ),
        ),
    )
    task = Task(
        start_position_m=135.0,
        target_position_m=29270.0,
        schedule_time_s=465.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )

    optimizer = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        cache_dir=None,
        precompute_mode="serial",
    )

    assert observed["vehicle"] is vehicle
    assert observed["track"] is line
    assert observed["factor"] == pytest.approx(0.99)
    assert observed["begin_pos"] == pytest.approx(135.0)
    assert observed["begin_speed"] == pytest.approx(0.0)
    assert observed["end_pos"] == pytest.approx(29276.0)
    assert observed["end_speed"] == pytest.approx(0.0)
    assert optimizer.speed_grid_upper_mps == pytest.approx(29.7)


def test_dp_optimize_uses_task_absolute_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=0.0,
        target_position_m=20.0,
        schedule_time_s=20.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.scenario = scenario
    optimizer.vehicle = scenario.vehicle
    optimizer.track = scenario.line
    optimizer.task = task
    optimizer.speed_grid_upper_mps = 30.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 2

    calls = 0

    def _fake_prepare(
        self: VariableSpacingDPOptimizer,
        *,
        start_position: float,
        target_position: float,
    ) -> dict[str, object]:
        del self, start_position, target_position
        return {
            "stages": np.asarray([0.0, 20.0], dtype=np.float64),
            "speed_states": np.arange(31.0, dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([30, 30], dtype=int),
            "transitions": [[]],
            "total_valid_edges": 0,
        }

    def _fake_solve_dp_inner(
        self: VariableSpacingDPOptimizer,
        *,
        cache: dict[str, object],
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> _DPSolution:
        del self, cache, lambda_time, start_state_idx, target_state_idx
        nonlocal calls
        calls += 1
        return _DPSolution(
            pos=np.asarray([0.0, 20.0], dtype=np.float64),
            speed=np.asarray([1.5, 0.0], dtype=np.float64),
            cum_time_s=np.asarray([0.0, 19.0], dtype=np.float64),
            total_energy=12.0,
        )

    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_solve_dp_inner",
        _fake_solve_dp_inner,
    )
    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_prepare_transition_graph_cache",
        _fake_prepare,
    )

    result = optimizer.optimize(
        start_pos=0.0,
        start_speed=0.0,
        target_pos=20.0,
        target_speed=0.0,
        schedule_time=20.0,
    )

    assert calls == 1
    assert result is not None
    assert result.time_s[-1] == pytest.approx(19.0)


def test_dp_optimize_expands_lambda_and_returns_closest_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=0.0,
        target_position_m=20.0,
        schedule_time_s=20.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=0.1,
    )
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.scenario = scenario
    optimizer.vehicle = scenario.vehicle
    optimizer.track = scenario.line
    optimizer.task = task
    optimizer.speed_grid_upper_mps = 1.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 2

    def _fake_prepare(
        self: VariableSpacingDPOptimizer,
        *,
        start_position: float,
        target_position: float,
    ) -> dict[str, object]:
        del self, start_position, target_position
        return {
            "stages": np.asarray([0.0, 20.0], dtype=np.float64),
            "speed_states": np.asarray([0.0, 1.0], dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([1, 1], dtype=int),
            "transitions": [[]],
            "total_valid_edges": 0,
        }

    calls: list[float] = []

    def _fake_solve(
        self: VariableSpacingDPOptimizer,
        *,
        cache: dict[str, object],
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> _DPSolution:
        del self, cache, start_state_idx, target_state_idx
        calls.append(lambda_time)
        if lambda_time < 1_000.0:
            total_time = 30.0
        elif lambda_time < 2_000.0:
            total_time = 25.0
        elif lambda_time < 4_000.0:
            total_time = 21.0
        else:
            total_time = 18.0
        return _DPSolution(
            pos=np.asarray([0.0, 20.0], dtype=np.float64),
            speed=np.asarray([0.0, 1.0], dtype=np.float64),
            cum_time_s=np.asarray([0.0, total_time], dtype=np.float64),
            total_energy=1.0,
        )

    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_prepare_transition_graph_cache",
        _fake_prepare,
    )
    monkeypatch.setattr(VariableSpacingDPOptimizer, "_solve_dp_inner", _fake_solve)

    result = optimizer.optimize(
        start_pos=0.0,
        start_speed=0.0,
        target_pos=20.0,
        target_speed=0.0,
        schedule_time=20.0,
    )

    assert result is not None
    assert result.time_s[-1] == pytest.approx(21.0)
    assert calls[:4] == pytest.approx([0.0, 1_000.0, 2_000.0, 4_000.0])
    assert len(calls) > 4


def test_dp_optimize_passes_nonzero_endpoint_states_to_inner_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=0.0,
        target_position_m=20.0,
        schedule_time_s=10.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.scenario = scenario
    optimizer.vehicle = scenario.vehicle
    optimizer.track = scenario.line
    optimizer.task = task
    optimizer.speed_grid_upper_mps = 2.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 1

    def _fake_prepare(
        self: VariableSpacingDPOptimizer,
        *,
        start_position: float,
        target_position: float,
    ) -> dict[str, object]:
        del self, start_position, target_position
        return {
            "stages": np.asarray([0.0, 20.0], dtype=np.float64),
            "speed_states": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([2, 2], dtype=int),
            "transitions": [[]],
            "total_valid_edges": 0,
        }

    observed_indices: list[tuple[int, int]] = []

    def _fake_solve(
        self: VariableSpacingDPOptimizer,
        *,
        cache: dict[str, object],
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> _DPSolution:
        del self, cache, lambda_time
        observed_indices.append((start_state_idx, target_state_idx))
        return _DPSolution(
            pos=np.asarray([0.0, 20.0], dtype=np.float64),
            speed=np.asarray([1.0, 2.0], dtype=np.float64),
            cum_time_s=np.asarray([0.0, 10.0], dtype=np.float64),
            total_energy=1.0,
        )

    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_prepare_transition_graph_cache",
        _fake_prepare,
    )
    monkeypatch.setattr(VariableSpacingDPOptimizer, "_solve_dp_inner", _fake_solve)

    result = optimizer.optimize(
        start_pos=0.0,
        start_speed=1.0,
        target_pos=20.0,
        target_speed=2.0,
        schedule_time=10.0,
    )

    assert result is not None
    assert observed_indices == [(1, 2)]


def test_dp_optimize_rejects_endpoint_speed_off_grid() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.task = Task(
        start_position_m=0.0,
        target_position_m=20.0,
        schedule_time_s=10.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    optimizer.speed_grid_upper_mps = 2.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 1

    with pytest.raises(ValueError, match="not representable"):
        _ = optimizer.optimize(
            start_pos=0.0,
            start_speed=0.5,
            target_pos=20.0,
            target_speed=0.0,
            schedule_time=10.0,
        )


def test_uniform_stage_generation_handles_reverse_direction() -> None:
    stages = generate_uniform_spacing_stages(
        uniform_step_size=30.0, start_position=100.0, target_position=0.0
    )

    assert len(stages) == 5
    assert stages[0] == pytest.approx(100.0)
    assert stages[-1] == pytest.approx(0.0)
    assert np.all(np.diff(stages) < 0.0)


def test_transition_graph_validation_rejects_malformed_payload() -> None:
    stages = np.asarray([0.0, 10.0], dtype=np.float64)
    speed_states = np.asarray([0.0, 1.0], dtype=np.float64)
    upper_idx = np.asarray([1, 1], dtype=int)
    graph = {
        "stages": stages,
        "speed_states": speed_states,
        "stage_speed_upper_idx": upper_idx,
        "transitions": [
            [
                (
                    np.asarray([1], dtype=int),
                    np.asarray([1.0], dtype=np.float64),
                    np.asarray([2.0], dtype=np.float64),
                ),
                None,
            ]
        ],
        "total_valid_edges": 1,
    }

    valid, reason = validate_transition_graph(
        graph,
        expected_stages=stages,
        expected_speed_states=speed_states,
        expected_stage_speed_upper_idx=upper_idx,
    )
    assert valid is True
    assert reason == ""

    graph["transitions"][0][0][2][0] = 0.0
    valid, reason = validate_transition_graph(
        graph,
        expected_stages=stages,
        expected_speed_states=speed_states,
        expected_stage_speed_upper_idx=upper_idx,
    )
    assert valid is False
    assert "time" in reason


def test_transition_graph_cache_requires_current_schema_and_structure(
    tmp_path: Path,
) -> None:
    folder = dict(
        stage_division="uniform",
        sub_stage_count=1,
        uniform_step_size=10.0,
        speed_grid_upper_mps=1.0,
        delta_speed=0.5,
    )

    stages = np.asarray([0.0, 10.0], dtype=np.float64)
    speed_states = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    upper_idx = np.asarray([2, 2], dtype=int)
    graph = {
        "stages": stages,
        "speed_states": speed_states,
        "stage_speed_upper_idx": upper_idx,
        "transitions": [
            [
                (
                    np.asarray([1], dtype=int),
                    np.asarray([1.0], dtype=np.float64),
                    np.asarray([2.0], dtype=np.float64),
                ),
                None,
                None,
            ]
        ],
        "total_valid_edges": 1,
    }
    content_hash = "a" * 64
    save_transition_graph_to_disk(
        cache_base_dir=tmp_path,
        graph_cache=graph,
        start_position=0.0,
        target_position=10.0,
        content_hash=content_hash,
        **folder,
    )

    loaded = load_transition_graph_from_disk(
        cache_base_dir=tmp_path,
        content_hash=content_hash,
        expected_stages=stages,
        expected_speed_states=speed_states,
        expected_stage_speed_upper_idx=upper_idx,
        start_position=0.0,
        target_position=10.0,
        **folder,
    )
    assert loaded is not None

    metadata_path = (
        tmp_path / make_cache_folder_name(content_hash=content_hash, **folder)
    ) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cache_schema_version"] = -1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert (
        load_transition_graph_from_disk(
            cache_base_dir=tmp_path,
            content_hash=content_hash,
            expected_stages=stages,
            expected_speed_states=speed_states,
            expected_stage_speed_upper_idx=upper_idx,
            start_position=0.0,
            target_position=10.0,
            **folder,
        )
        is None
    )


def test_dp_optimize_returns_valid_speed_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=135.0,
        target_position_m=335.0,
        schedule_time_s=40.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    optimizer = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        delta_speed=1.0,
        uniform_step_size=50.0,
        precompute_mode="serial",
        show_precompute_progress=False,
        cache_dir=None,
    )
    converted_solutions: list[_DPSolution] = []
    original_to_speed_profile = optimizer._to_speed_profile

    def _wrapped_to_speed_profile(solution: _DPSolution) -> SpeedProfile:
        converted_solutions.append(solution)
        return original_to_speed_profile(solution)

    monkeypatch.setattr(optimizer, "_to_speed_profile", _wrapped_to_speed_profile)

    profile = optimizer.optimize(135.0, 0.0, 335.0, 0.0, 40.0)
    assert profile is not None
    assert len(converted_solutions) == 1
    chosen_solution = converted_solutions[0]

    np.testing.assert_allclose(profile.time_s, chosen_solution.cum_time_s)
    np.testing.assert_allclose(profile.position_m, chosen_solution.pos)
    np.testing.assert_allclose(profile.speed_mps, chosen_solution.speed)
    assert profile.total_energy_kj[-1] == pytest.approx(
        chosen_solution.total_energy, rel=1e-12
    )

    pos = profile.position_m
    spd = profile.speed_mps
    cum_edge_energy = 0.0
    for k in range(pos.size - 1):
        ds = pos[k + 1] - pos[k]
        acc, dur = accel_between(spd[k], spd[k + 1], ds)
        prop, levi = segment_energy(
            scenario.energy,
            scenario.vehicle,
            scenario.line,
            begin_pos=pos[k],
            begin_speed=spd[k],
            acc=acc,
            distance=abs(ds),
            direction=1 if ds > 0 else -1,
            operation_time=dur,
        )
        cum_edge_energy += prop + levi
        delta_prop = (
            profile.propulsion_energy_kj[k + 1] - profile.propulsion_energy_kj[k]
        )
        delta_levi = (
            profile.levitation_energy_kj[k + 1] - profile.levitation_energy_kj[k]
        )
        assert delta_prop == pytest.approx(prop)
        assert delta_levi == pytest.approx(levi)

    assert profile.total_energy_kj[-1] == pytest.approx(cum_edge_energy, rel=1e-12)


def test_dp_cache_input_hash_composition_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 缓存键的输入组成包含四项：
    # 1. scenario_hash
    # 2. task.max_stop_error_m
    # 3. static_region_sample_step_m
    # 4. mtto.__version__
    # 计算配置：DP golden 的 dp_feasible 案例
    # 该基线哈希录制于 mtto.__version__ == "0.1.0"；版本号变化会改变缓存键，
    # 这是预期行为（另见下方 test_dp_cache_input_hash_sensitivity_to_new_fields
    # 的 "version" 用例），因此这里固定版本号为录制基线时的值，而不是更新
    # expected_hash，以确保本测试只守护"输入组成"（上述四项）自那以来未变，
    # 不受包版本升级本身影响。
    import mtto

    monkeypatch.setattr(mtto, "__version__", "0.1.0")
    expected_hash = "d177695b27b332f22a882cbedaefa14faa6e47337a8081a38cbdbd9ac3bce572"
    case = next(c for c in DP_CASES if c.name == "dp_feasible")
    optimizer = build_dp_optimizer(case)
    *_, actual_hash = optimizer._graph_inputs(
        optimizer.task.start_position_m, optimizer.task.target_position_m
    )
    assert actual_hash == expected_hash


@pytest.mark.parametrize(
    ("field", "change_kind"),
    [
        ("unchanged", "none"),
        ("scenario_hash", "modified"),
        ("max_stop_error_m", "modified"),
        ("version", "modified"),
    ],
)
def test_dp_cache_input_hash_sensitivity_to_new_fields(
    field: str,
    change_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(c for c in DP_CASES if c.name == "dp_feasible")
    optimizer = build_dp_optimizer(case)
    start, target = optimizer.task.start_position_m, optimizer.task.target_position_m
    *_, baseline_hash = optimizer._graph_inputs(start, target)

    if field == "scenario_hash":
        new_scenario = replace(
            optimizer.scenario, scenario_hash="modified_scenario_hash_test"
        )
        optimizer.scenario = new_scenario
    elif field == "max_stop_error_m":
        new_task = replace(
            optimizer.task, max_stop_error_m=optimizer.task.max_stop_error_m + 0.5
        )
        optimizer.task = new_task
    elif field == "version":
        import mtto

        monkeypatch.setattr(mtto, "__version__", "9.9.9")

    *_, new_hash = optimizer._graph_inputs(start, target)

    if change_kind == "none":
        assert new_hash == baseline_hash
    else:
        assert new_hash != baseline_hash
