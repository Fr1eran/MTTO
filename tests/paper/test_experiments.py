"""Small real training matrix and recovery regressions."""

from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mtto.dp.solver import VariableSpacingDPOptimizer
from mtto.evaluation.quality import assess
from mtto.io.artifacts import (
    RunKind,
    RunPayload,
    RunRecord,
    read_completed_run,
    task_to_json,
    write_run,
)
from mtto.io.scenario import load_scenario, load_tasks
from mtto.workflows.dp import DPConfig
from paper import analysis
from paper.experiments import runner, schedule_change, step_time
from paper.experiments.method_ablation import summarize, write_summary
from paper.experiments.spec import expand_matrix, load_experiment_spec
from paper.plotting import ablation
from paper.plotting.ablation import (
    method_figures,
    require_clean,
)


@pytest.fixture(scope="module")
def small_spec(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("method_ablation")
    tasks = root / "tasks.toml"
    tasks.write_text(
        "[short]\nstart_position_m = 135.0\ntarget_position_m = 1135.0\n"
        "schedule_time_s = 120.0\nmax_jerk_mps3 = 0.75\n"
        "max_stop_error_m = 0.3\nmax_arr_time_error_s = 10.0\n",
        encoding="utf-8",
    )
    spec = root / "method.toml"
    repo = Path(__file__).resolve().parents[2]
    spec.write_text(
        "[experiment]\nname = 'method'\n"
        f"scenario = '{repo / 'paper/specs/scenario.toml'}'\n"
        f"line_dir = '{repo / 'paper/data/line'}'\n"
        f"tasks = '{tasks}'\ntask = 'short'\n"
        f"output_root = '{root / 'runs'}'\nseeds = [11]\n\n"
        "[train]\nstep_time_s = 1.0\ngamma = 0.998\n"
        "budget_mode = 'environment_steps'\n"
        "training_rollouts = 1\nnum_envs = 1\nn_steps_per_env = 512\n"
        "evaluation_deterministic = true\nkeep_best = true\n"
        "safety_truncation_bin_size_m = 5000.0\ndevice = 'cpu'\n\n"
        "[[variants]]\nid = 'ppo'\nlabel = 'PPO'\nreward_preset = 'basic'\n\n"
        "[[variants]]\nid = 'ppo_safety'\nlabel = 'PPO+Safety'\n"
        "reward_preset = 'basic_safety'\n",
        encoding="utf-8",
    )
    return spec


def test_strict_spec_and_matrix(small_spec: Path) -> None:
    spec = load_experiment_spec(small_spec)
    assert spec.train["training_episodes"] is None
    assert spec.train["evaluation_interval_rollouts"] is None
    assert spec.train["evaluation_interval_episodes"] is None
    assert [run.run_label for run in expand_matrix(spec)] == [
        "method__ppo__seed0011",
        "method__ppo_safety__seed0011",
    ]
    source = small_spec.read_text(encoding="utf-8")
    for bad in (
        source.replace("gamma = 0.998\n", ""),
        source.replace("gamma = 0.998", "gamma = 0.998\nextra = 1"),
    ):
        path = small_spec.with_name("invalid.toml")
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_experiment_spec(path)


@pytest.mark.parametrize("entrypoint", ["execute_matrix", "completed_matrix"])
def test_io_error_does_not_discard_run(
    small_spec: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    spec = replace(load_experiment_spec(small_spec), output_root=tmp_path)
    run_dir = tmp_path / f"{expand_matrix(spec)[0].run_label}__01"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", True))

    def fail_read(_path: Path) -> None:
        raise OSError("transient read failure")

    monkeypatch.setattr(runner, "read_completed_run", fail_read)
    with pytest.raises(OSError, match="transient read failure"):
        getattr(runner, entrypoint)(spec)
    assert run_dir.is_dir()
    assert (run_dir / "run.json").exists()


def test_process_pool_matches_serial_training(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", True))
    spec = load_experiment_spec(small_spec)
    serial = runner.execute_matrix(replace(spec, output_root=tmp_path / "serial"), 1)
    pooled_spec = replace(spec, output_root=tmp_path / "pooled")
    pooled = runner.execute_matrix(pooled_spec, 2)
    assert [item.directory.name for item in pooled] == [
        item.directory.name for item in serial
    ]
    assert not any(item.reused for item in (*serial, *pooled))
    for left, right in zip(serial, pooled, strict=True):
        with (
            zipfile.ZipFile(left.directory / "policy.zip") as first,
            zipfile.ZipFile(right.directory / "policy.zip") as second,
        ):
            assert first.read("policy.pth") == second.read("policy.pth")
    assert all(item.reused for item in runner.execute_matrix(pooled_spec, 2))


def test_run_reuse_recovery_and_summary(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_experiment_spec(small_spec)
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", True))
    first = runner.execute_matrix(spec)
    assert len(first) == 2 and all(not item.reused for item in first)
    assert all(item.directory.name.endswith("__01") for item in first)
    for item in first:
        completed = read_completed_run(item.directory)
        paper = json.loads((item.directory / "paper.json").read_text())
        assert paper["reuse_key"] == runner.reuse_key(completed.record)

    original_train = runner.train_workflow.train
    calls = []

    def counting_train(*args: object, **kwargs: object) -> object:
        calls.append(args[3])
        return original_train(*args, **kwargs)

    monkeypatch.setattr(runner.train_workflow, "train", counting_train)
    reused = runner.execute_matrix(spec)
    assert all(item.reused for item in reused) and not calls
    assert len(list(spec.output_root.iterdir())) == 2

    (first[0].directory / "quality.json").unlink()
    recovered = runner.execute_matrix(spec)
    assert len(calls) == 1
    assert recovered[0].directory.name.endswith("__02")
    assert not first[0].directory.exists() and recovered[1].reused

    summary = summarize(spec, tuple(item.directory for item in recovered))
    write_summary(summary, tmp_path / "summary")
    written = json.loads((tmp_path / "summary/summary.json").read_text())
    payload = read_completed_run(recovered[0].directory).payload
    assert written["raw_seed_metrics"]["ppo"][0]["stop_error_m"] == abs(
        (payload.best or payload).quality.metrics.stop_error_m
    )
    assert written["raw_seed_metrics"]["ppo"][0]["time_error_s"] == (
        (payload.best or payload).quality.metrics.arrival_time_error_s
    )
    assert set(written["representative_policies"]) == {"ppo", "ppo_safety"}
    # Without periodic evaluations there is no late-stage feasibility ratio.
    assert written["performance"]["ppo"]["late_feasible_rate"] is None
    assert written["performance"]["ppo"]["first_feasible_progress"] is None
    assert written["raw_seed_metrics_final"]["ppo"][0]["time_error_s"] == (
        payload.quality.metrics.arrival_time_error_s
    )
    # Trajectory metrics average only policies of each source that reached the target.
    for method in ("ppo", "ppo_safety"):
        for source, raw_key in (
            ("best", "raw_seed_metrics"),
            ("final", "raw_seed_metrics_final"),
        ):
            arrived = [item for item in written[raw_key][method] if item["success"]]
            energy = written["performance"][method][source]["total_energy_kwh"]
            if arrived:
                assert energy["mean"] == pytest.approx(
                    np.mean([item["total_energy_kwh"] for item in arrived])
                )
            else:
                assert energy is None
    performance_table = tmp_path / "summary/method_performance_table.md"
    assert "PPO+Safety" in performance_table.read_text()

    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", False))
    clean = runner.execute_matrix(spec)
    assert len(calls) == 3 and all(not item.reused for item in clean)
    assert recovered[0].directory.exists()
    assert all(item.reused for item in runner.execute_matrix(spec))

    changed = replace(spec, train={**spec.train, "gamma": 0.997})
    changed_runs = runner.execute_matrix(changed)
    assert len(calls) == 5 and all(not item.reused for item in changed_runs)
    assert all(item.directory.exists() for item in clean)
    assert runner.completed_matrix(spec) == tuple(item.directory for item in clean)
    assert runner.completed_matrix(changed) == tuple(
        item.directory for item in changed_runs
    )

    result_path = changed_runs[0].directory / "result.json"
    data = json.loads(result_path.read_text())
    data["training"]["target_reached"] = False
    result_path.write_text(json.dumps(data))
    with pytest.raises(runner.ExperimentStopped):
        runner.execute_matrix(changed)
    assert len(calls) == 5


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "align_exact",
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0]), np.array([5.0, 7.0])),
        ),
        (
            "smooth_episode_curve",
            (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])),
        ),
        ("aggregate_matrix", (np.array([[1.0, 2.0], [3.0, np.nan]]),)),
        (
            "aggregate_indexed_series",
            (
                [
                    (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
                    (np.array([1.0]), np.array([5.0])),
                ],
            ),
        ),
        (
            "aggregate_step_binned_series",
            (
                [
                    (np.array([1.0, 4.0]), np.array([2.0, 6.0])),
                    (np.array([2.0]), np.array([4.0])),
                ],
            ),
        ),
    ],
)
def test_copied_statistics_match_legacy(name: str, args: tuple) -> None:
    kwargs = (
        {"window": 2}
        if name == "smooth_episode_curve"
        else {"bin_width": 2, "axis_max": 4}
        if name == "aggregate_step_binned_series"
        else {}
    )
    result = getattr(analysis, name)(*args, **kwargs)
    fixed = {
        "align_exact": (np.array([5.0, np.nan, 7.0]),),
        "smooth_episode_curve": (np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0])),
        "aggregate_matrix": (
            np.array([2.0, 2.0]),
            np.array([np.sqrt(2.0), 0.0]),
            np.array([2, 1]),
        ),
        "aggregate_indexed_series": (
            np.array([1.0, 2.0]),
            np.array([4.0, 4.0]),
            np.array([np.sqrt(2.0), 0.0]),
            np.array([2, 1]),
        ),
        "aggregate_step_binned_series": (
            np.array([2.0, 4.0]),
            np.array([3.0, 6.0]),
            np.array([np.sqrt(2.0), 0.0]),
            np.array([2, 1]),
        ),
    }
    arrays = result if isinstance(result, tuple) else (result,)
    for current, previous in zip(arrays, fixed[name], strict=True):
        np.testing.assert_allclose(current, previous, equal_nan=True)


@pytest.mark.parametrize("time_error", [2.0, 10.0])
def test_copied_constraint_assessment_matches_legacy(time_error: float) -> None:
    metrics = {
        "termination_reason": "STOPPED_IN_ZONE",
        "stop_error_m": 0.1,
        "time_error_s": time_error,
        "min_safety_margin_mps": 0.2,
        "safety_violation_count": 0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 10.0,
        "success": True,
        "precise_arrival": True,
        "punctual_arrival": time_error < 10.0,
        "safe": True,
        "feasible": time_error < 10.0,
    }
    assert asdict(analysis.assess_constraints(metrics)) == {
        "success": True,
        "precise_arrival": True,
        "punctual_arrival": time_error < 10.0,
        "safe": True,
        "feasible": time_error < 10.0,
        "failure_reasons": () if time_error < 10.0 else ("time_error",),
    }


def test_generic_variants_and_step_time(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = small_spec.read_text(encoding="utf-8")
    prefix = source.split("[[variants]]")[0]
    prefix = prefix.replace("name = 'method'", "name = 'step_time'")
    prefix = prefix.replace(
        "device = 'cpu'", "device = 'cpu'\nreward_preset = 'basic_safety_punctuality'"
    )
    prefix = prefix.replace(str(small_spec.parent / "runs"), str(tmp_path / "steps"))
    path = tmp_path / "step.toml"
    path.write_text(
        prefix + "[[variants]]\nid = '2p0'\nlabel = '2.0 s'\nstep_time_s = 2.0\n\n"
        "[[variants]]\nid = '1p5'\nlabel = '1.5 s'\nstep_time_s = 1.5\n",
        encoding="utf-8",
    )
    spec = load_experiment_spec(path)
    assert [planned.config.step_time_s for planned in expand_matrix(spec)] == [
        2.0,
        1.5,
    ]
    assert [planned.run_label for planned in expand_matrix(spec)] == [
        "step_time__2p0__seed0011",
        "step_time__1p5__seed0011",
    ]
    for bad in (
        path.read_text().replace("step_time_s = 1.5", "seed = 50"),
        path.read_text().replace("step_time_s = 1.5", "unknown = 50"),
    ):
        invalid = tmp_path / "invalid_step.toml"
        invalid.write_text(bad)
        with pytest.raises(ValueError):
            load_experiment_spec(invalid)
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", False))
    results = step_time.run(path)
    assert len(results) == 2
    summary = step_time.summarize(spec, tuple(item.directory for item in results))
    assert summary["matrix_id"] == "step_time"
    assert [item["step_time_s"] for item in summary["variants"].values()] == [2.0, 1.5]
    step_time.write_summary(summary, tmp_path / "step_summary")
    assert (tmp_path / "step_summary/step_time_summary.json").exists()
    assert "1.5 s" in (tmp_path / "step_summary/step_time_table.md").read_text()
    table = (tmp_path / "step_summary/step_time_table.md").read_text()
    assert "| Policy |" in table
    assert table.count("| Best |") == table.count("| Final |") == 2
    for variant, result in zip(summary["variants"].values(), results, strict=True):
        payload = read_completed_run(result.directory).payload
        final = variant["final"]["per_seed"][0]
        assert final["time_error_s"] == payload.quality.metrics.arrival_time_error_s
        assert final["feasible"] == payload.quality.feasible
        assert variant["final"]["arrived_count"] == int(payload.quality.completed)
        if not payload.quality.completed:
            assert variant["final"]["metrics"]["energy_kwh"] is None


def _step_variant(
    step: float,
    best_energy: tuple[float, float],
    final_energy_std: float,
    drift: float,
    feasible: int = 5,
    late: float = 1.0,
) -> dict[str, object]:
    def source(energy: dict[str, float]) -> dict[str, object]:
        return {
            "feasible_count": feasible,
            "per_seed": [{}] * 5,
            "metrics": {"energy_kwh": energy},
        }

    return {
        "variant_id": f"{step:g}",
        "step_time_s": step,
        "late_feasible_rate": late,
        "energy_drift_kwh": {"mean": drift, "std": 0.0},
        "best": source({"mean": best_energy[0], "std": best_energy[1]}),
        "final": source({"mean": best_energy[0] + drift, "std": final_energy_std}),
    }


@pytest.mark.parametrize(
    ("variants", "expected", "excluded"),
    [
        # The lowest best-checkpoint energy (1.5 s) does not survive at the end of
        # training; final energies within the pooled spread defer to stability,
        # and an unreliable period is gated out despite its low energy.
        (
            [
                _step_variant(0.5, (800.0, 100.0), 170.0, 40.0, feasible=4, late=0.8),
                _step_variant(1.0, (854.3, 18.4), 7.3, 13.3),
                _step_variant(1.5, (834.7, 23.0), 36.5, 61.0),
                _step_variant(2.0, (851.4, 22.2), 13.8, 28.8),
            ],
            1.0,
            {
                "0.5": "best policy not feasible in every run",
                "1.5": "final-policy energy outside margin",
            },
        ),
        # A clearly lower energy wins even with a less stable final policy.
        (
            [
                _step_variant(1.0, (900.0, 5.0), 2.0, 1.0),
                _step_variant(2.0, (850.0, 5.0), 30.0, 20.0),
            ],
            2.0,
            {"1": "final-policy energy outside margin"},
        ),
        (
            [_step_variant(1.0, (850.0, 5.0), 2.0, 1.0, late=0.5)],
            None,
            {"1": "late-stage feasible share below gate"},
        ),
    ],
)
def test_step_time_selection_rule(
    variants: list[dict[str, object]],
    expected: float | None,
    excluded: dict[str, str],
) -> None:
    selection = step_time.select_step_time(variants)
    assert selection["recommended_step_time_s"] == expected
    assert selection["excluded"] == excluded


def test_step_time_spec_expands_candidates_and_seeds() -> None:
    spec = load_experiment_spec(Path("paper/specs/step_time.toml"))
    planned = expand_matrix(spec)

    assert spec.seeds == (11, 131, 239, 359, 443)
    assert len(planned) == 4 * 5
    assert [
        (run.variant.id, run.config.step_time_s) for run in planned[:: len(spec.seeds)]
    ] == [("0p5", 0.5), ("1p0", 1.0), ("1p5", 1.5), ("2p0", 2.0)]


def test_schedule_change_evaluation_reuse_and_figures(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # DP has no feasible trajectory for the 1 km task, so use a shorter one.
    tasks = tmp_path / "tasks.toml"
    tasks.write_text(
        (small_spec.parent / "tasks.toml")
        .read_text(encoding="utf-8")
        .replace("target_position_m = 1135.0", "target_position_m = 535.0")
        .replace("schedule_time_s = 120.0", "schedule_time_s = 60.0"),
        encoding="utf-8",
    )
    source = small_spec.read_text(encoding="utf-8")
    source = (
        source.split("[[variants]]")[0]
        .replace(str(small_spec.parent / "runs"), str(tmp_path / "methods"))
        .replace(str(small_spec.parent / "tasks.toml"), str(tasks))
    )
    source = source.replace(
        "training_rollouts = 1",
        "training_rollouts = 2\nevaluation_interval_rollouts = 1",
    )
    source += (
        "[[variants]]\nid = 'ppo_pirs'\nlabel = 'PPO+PIRS'\n"
        "reward_preset = 'basic_safety_punctuality'\n"
    )
    method_path = tmp_path / "method.toml"
    method_path.write_text(source)
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", False))
    method_spec = load_experiment_spec(method_path)
    training = runner.execute_matrix(method_spec)
    assert len(training) == 1
    assert read_completed_run(training[0].directory).payload.best is not None
    from paper.experiments.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper.experiments",
            "method_ablation",
            "figures",
            "--spec",
            str(method_path),
            "--output",
            str(tmp_path / "cli_figures"),
        ],
    )
    main()
    assert (tmp_path / "cli_figures/method_training_curves.pdf").exists()
    assert (tmp_path / "cli_figures/method_representative_profiles.pdf").exists()

    scenario = load_scenario(method_spec.scenario, method_spec.line_dir)
    task = load_tasks(method_spec.tasks)[method_spec.task]
    dp_config = DPConfig(
        delta_speed=1.0,
        stage_division="uniform",
        uniform_step_size=50.0,
        sub_stage_count=30,
        max_outer_iterations=100,
        precompute_mode="serial",
        precompute_workers=None,
        precompute_chunk_size=None,
    )
    nominal = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        cache_dir=None,
        delta_speed=dp_config.delta_speed,
        uniform_step_size=dp_config.uniform_step_size,
        precompute_mode="serial",
        show_precompute_progress=False,
    ).optimize(task.start_position_m, 0.0, task.target_position_m, 0.0, 60.0)
    dp_dir = tmp_path / "dp"
    write_run(
        dp_dir,
        RunRecord(
            run_id="nominal-dp",
            kind=RunKind.DP_SOLVE,
            config=asdict(dp_config),
            scenario_hash=scenario.scenario_hash,
            task=task_to_json(task),
            policy_io_version=None,
            mtto_version="test",
            created_at="2026-10-04T00:00:00+00:00",
        ),
        RunPayload(profile=nominal, quality=assess(nominal, scenario, task)),
    )
    # The change triggers at the start position so that the barely trained
    # policy always reaches it.
    case_spec = tmp_path / "schedule.toml"
    case_spec.write_text(
        "[experiment]\n"
        f"source_spec = '{method_path}'\n"
        f"dp_run = '{dp_dir}'\ndelta_times_s = [0.0, 30.0]\n"
        "change_distance_m = 135.0\ntiming_repeats = 1\n"
        f"output_root = '{tmp_path / 'schedule'}'\n"
    )
    spec = schedule_change.load_schedule_spec(case_spec)
    evaluations, replan_dir, reused = schedule_change.run(case_spec)
    assert len(evaluations) == 2 and not reused
    assert not any(item.planned.config.use_best for item in evaluations)
    plus = read_completed_run(evaluations[1].directory)
    assert plus.record.task["schedule_change"] == {
        "trigger_position_m": 135.0,
        "new_schedule_time_s": 90.0,
    }
    replanned = read_completed_run(replan_dir / "dp__plus_30p0s")
    assert replanned.record.config["remaining_schedule_time_s"] == pytest.approx(90.0)
    assert replanned.payload.quality.metrics.arrival_time_error_s == pytest.approx(
        replanned.payload.profile.time_s[-1] - 90.0
    )
    again, again_dir, again_reused = schedule_change.run(case_spec)
    assert all(item.reused for item in again)
    assert again_reused and again_dir == replan_dir

    run_dirs, found_dir = schedule_change.completed_runs(spec)
    assert found_dir == replan_dir
    summary = schedule_change.summarize(spec, run_dirs, replan_dir)
    pairs = [(entry["case"]["token"], entry["method"]) for entry in summary["entries"]]
    assert pairs == [
        ("original", schedule_change.RL_LABEL),
        ("original", "DP"),
        ("plus_30p0s", schedule_change.RL_LABEL),
        ("plus_30p0s", "DP"),
    ]
    assert all(entry["recompute_time_s"] > 0.0 for entry in summary["entries"])
    schedule_change.write_summary(summary, tmp_path / "schedule_summary")
    assert (
        "Recomputation time"
        in (tmp_path / "schedule_summary/schedule_time_change_table.md").read_text()
    )
    require_clean(run_dirs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper.experiments",
            "schedule_change",
            "figures",
            "--spec",
            str(case_spec),
            "--output",
            str(tmp_path / "cli_schedule_figures"),
        ],
    )
    main()
    assert (
        tmp_path / "cli_schedule_figures/schedule_time_change_comparison.pdf"
    ).exists()

    timing_path = replan_dir / "timing.json"
    timing = json.loads(timing_path.read_text())
    timing["dirty"] = True
    timing_path.write_text(json.dumps(timing))
    with pytest.raises(ValueError, match=replan_dir.name):
        main()

    policy = training[0].directory / "policy.zip"
    policy.write_bytes(policy.read_bytes() + b"changed")
    changed, changed_dir, changed_reused = schedule_change.run(case_spec)
    assert all(not item.reused for item in changed)
    assert not changed_reused and changed_dir != replan_dir
    assert replan_dir.exists()


def test_schedule_change_defaults_and_case_tokens() -> None:
    defaults = schedule_change.load_schedule_spec(
        Path("paper/specs/schedule_change.toml")
    )
    assert defaults.delta_times_s == (0.0, 30.0, -30.0)
    assert defaults.change_distance_m == 8000.0
    assert [
        schedule_change.build_schedule_change_case(delta).token
        for delta in (0, 30, -30)
    ] == ["original", "plus_30p0s", "minus_30p0s"]


def test_schedule_change_trigger_position_is_absolute() -> None:
    """Regression for 7h: change_distance_m is an absolute track position.

    The legacy driver (``scripts/run_schedule_time_change.py``) treated
    ``--change-distance-m`` as the absolute track position at which the
    schedule change triggers, not a distance relative to the task's start
    position. This must not regress to ``start_position_m + change_distance_m``.
    """
    spec = schedule_change.load_schedule_spec(Path("paper/specs/schedule_change.toml"))
    source_spec = load_experiment_spec(spec.source_spec)
    task = load_tasks(source_spec.tasks)[source_spec.task]
    assert task.schedule_time_s == 465.0
    for delta in spec.delta_times_s:
        case_task = schedule_change.build_case_task(task, delta, spec.change_distance_m)
        if delta == 0.0:
            assert case_task.schedule_change is None
        else:
            assert case_task.schedule_change is not None
            assert case_task.schedule_change.trigger_position_m == 8000.0
            assert case_task.schedule_change.new_schedule_time_s == 465.0 + delta


def test_formal_figure_axes_match_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    method_spec = load_experiment_spec(Path("paper/specs/method_ablation.toml"))
    rollout_steps = 8 * 1024
    steps = [rollout * rollout_steps for rollout in range(12, 400, 12)]
    series = {"mean": [0.5] * len(steps), "std": [0.1] * len(steps)}
    method_summary = {
        "methods": [variant.id for variant in method_spec.variants],
        "method_labels": {
            variant.id: variant.label for variant in method_spec.variants
        },
        "training_curves": {
            variant.id: {
                "training_steps": steps,
                "speed_violation_rate": series,
                "arrival_ratio": series,
            }
            for variant in method_spec.variants
        },
    }
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    # The representative-profile figure reads run artifacts; it is covered by
    # the schedule-change integration test.
    monkeypatch.setattr(
        ablation, "representative_profiles_figure", lambda *_: tmp_path / "unused"
    )
    try:
        method_figures(method_summary, method_spec, tmp_path)
    finally:
        monkeypatch.setattr(plt, "close", close)
        for figure in figures:
            close(figure)
    assert len(figures) == 1
    axis_end = 400 * rollout_steps
    inset_end = 396 * rollout_steps
    (training,) = figures
    assert [axis.get_xlim() for axis in training.axes] == [(0, axis_end)] * 2
    assert training.axes[0].child_axes[0].get_xlim() == (
        200 * rollout_steps,
        inset_end,
    )
    for axis in training.axes:
        assert axis.xaxis.get_major_formatter()._powerlimits == (6, 6)
