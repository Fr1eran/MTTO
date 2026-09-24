import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt

import scripts.compare_speed_profiles as compare_module
from scripts.compare_speed_profiles import (
    DEFAULT_REAL_CURVE_PATH,
    ProfileMetrics,
    SpeedProfile,
    _build_cli_parser,
    _create_comparison_axes,
    _finalize_comparison_figure,
    _parse_baseline_spec,
    _resolve_target_schedule_time,
    _validate_common_target_position,
    compute_min_limit_margin_kmh,
    format_comparison_table,
    load_real_operation_profile,
)
from utils.trajectory import OptimizedCurveArtifact


def test_compare_speed_profiles_cli_uses_explicit_rl_model_dir() -> None:
    args = _build_cli_parser().parse_args(["--rl-model-dir", "output/model"])

    assert args.real_curve == DEFAULT_REAL_CURVE_PATH
    assert args.rl_model_dir == "output/model"
    assert args.no_safeguard is False


def test_comparison_figure_uses_one_trajectory_only_shared_legend() -> None:
    figure, axes = plt.subplots(3, 1)
    for axis in axes:
        axis.set_title("remove me")
        axis.plot([0.0, 1.0], [0.0, 1.0], label="environment")
        axis.legend()

    _finalize_comparison_figure(figure, tuple(axes))

    assert [axis.get_title() for axis in axes] == ["", "", ""]
    assert all(axis.get_legend() is None for axis in axes)
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.texts] == [
        "DP",
        "PPO-PIRS (proposed)",
        "Recorded operation",
    ]
    assert [handle.get_color() for handle in legend.legend_handles] == [
        "#181818",
        "#ED7D31",
        "#7B61A8",
    ]
    plt.close(figure)


def test_comparison_figure_layout_uses_shared_position_axis() -> None:
    figure, axes = _create_comparison_axes()

    assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
    assert axes[1].get_shared_x_axes().joined(axes[1], axes[2])

    plt.close(figure)


def test_load_real_operation_profile_reads_required_aligned_arrays(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "real_curve.npz"
    np.savez_compressed(
        curve_path,
        position_m=np.asarray([100.0, 110.0]),
        speed_mps=np.asarray([5.0, 0.0]),
        time_s=np.asarray([0.0, 4.0]),
        target_position_m=np.asarray(110.0),
    )

    profile = load_real_operation_profile(curve_path)

    assert profile.label == "Recorded operation"
    assert profile.target_position_m == pytest.approx(110.0)
    np.testing.assert_allclose(profile.position_m, [100.0, 110.0])


def test_load_real_operation_profile_rejects_missing_required_arrays(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "real_curve.npz"
    np.savez_compressed(
        curve_path,
        position_m=np.asarray([100.0, 110.0]),
        speed_mps=np.asarray([5.0, 0.0]),
    )

    with pytest.raises(ValueError, match="missing required arrays"):
        _ = load_real_operation_profile(curve_path)


def test_target_schedule_time_requires_matching_dp_and_rl_tasks() -> None:
    with pytest.raises(ValueError, match="target_time_s differ"):
        _ = _resolve_target_schedule_time(
            dp_metrics={"target_time_s": 430.0},
            rl_metrics={"target_time_s": 431.0},
        )


def test_common_target_position_rejects_unaligned_profiles() -> None:
    profiles = [
        SpeedProfile(
            "DP",
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            1.0,
        ),
        SpeedProfile(
            "Actual",
            np.asarray([0.0, 2.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            2.0,
        ),
    ]

    with pytest.raises(ValueError, match="target positions differ"):
        _ = _validate_common_target_position(profiles)


def test_format_comparison_table_contains_only_requested_metrics() -> None:
    table = format_comparison_table(
        [
            (
                "DP",
                ProfileMetrics(1.25, 0.0, 123.456, 0.123456),
            ),
            (
                "PPO-PIRS (proposed)",
                ProfileMetrics(2.5, 0.25, 120.0, 0.1),
            ),
            (
                "Recorded operation",
                ProfileMetrics(3.0, 0.5, 130.0, None),
            ),
        ]
    )

    assert "Time error Δt (s)" in table
    assert "Stop error (m)" in table
    assert "Total energy (kWh)" in table
    assert "Cumulative acceleration variation (m/s²)" in table
    assert "123.456" in table
    assert "0.123456" in table
    assert "—" in table
    assert "not directly comparable" in table
    assert "Note:" in table


def test_deduplicate_legend_is_removed() -> None:
    assert not hasattr(compare_module, "_deduplicate_legend")


def test_main_uses_comparison_axes_and_scientific_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pos = np.linspace(0.0, 6000.0, 20)
    speed = np.linspace(0.0, 20.0, 20)
    time = np.linspace(0.0, 465.0, 20)

    monkeypatch.setattr(
        compare_module,
        "_resolve_curve_artifacts",
        lambda **kw: (
            OptimizedCurveArtifact(
                npz_path="dummy_dp.npz", metrics_path="dummy_dp.json"
            ),
            OptimizedCurveArtifact(
                npz_path="dummy_rl.npz", metrics_path="dummy_rl.json"
            ),
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "load_dp_curve_artifact",
        lambda *a, **kw: (
            pos,
            speed,
            time,
            {
                "target_time_s": 465.0,
                "target_position_m": 6000.0,
                "total_energy_kj": 100.0,
            },
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "load_rl_curve_artifact",
        lambda *a, **kw: (
            pos,
            speed,
            {
                "target_time_s": 465.0,
                "target_position_m": 6000.0,
                "total_time_s": 465.0,
                "total_energy_kj": 105.0,
            },
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "load_real_operation_profile",
        lambda *a, **kw: SpeedProfile(
            label="Recorded operation",
            position_m=pos,
            speed_mps=speed,
            time_s=time,
            target_position_m=6000.0,
        ),
    )

    called_helpers: list[str] = []
    recovered_time_axes: list[np.ndarray] = []
    orig_create_axes = compare_module._create_comparison_axes
    orig_finalize = compare_module._finalize_comparison_figure
    orig_recover_time = compare_module.recover_time_axis_from_trajectory
    orig_save_sci = compare_module.save_sci_figure

    def spy_create_axes():
        called_helpers.append("_create_comparison_axes")
        return orig_create_axes()

    def spy_finalize(figure, axes, legend_entries=None):
        called_helpers.append("_finalize_comparison_figure")
        return orig_finalize(figure, axes, legend_entries)

    def spy_recover_time(pos_arr, speed_arr):
        called_helpers.append("recover_time_axis_from_trajectory")
        np.testing.assert_allclose(pos_arr, pos)
        np.testing.assert_allclose(speed_arr, speed)
        recovered_time = orig_recover_time(pos_arr, speed_arr)
        recovered_time_axes.append(recovered_time)
        return recovered_time

    def spy_save_sci(fig, output_file, **kwargs):
        called_helpers.append("save_sci_figure")
        return orig_save_sci(fig, output_file, **kwargs)

    monkeypatch.setattr(compare_module, "_create_comparison_axes", spy_create_axes)
    monkeypatch.setattr(compare_module, "_finalize_comparison_figure", spy_finalize)
    monkeypatch.setattr(
        compare_module,
        "recover_time_axis_from_trajectory",
        spy_recover_time,
    )
    monkeypatch.setattr(compare_module, "save_sci_figure", spy_save_sci)

    output_dir = tmp_path / "comparison_figure"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_speed_profiles",
            "--rl-model-dir",
            "output/model",
            "--output-dir",
            str(output_dir),
            "--no-show",
        ],
    )

    compare_module.main()

    assert called_helpers == [
        "recover_time_axis_from_trajectory",
        "_create_comparison_axes",
        "_finalize_comparison_figure",
        "save_sci_figure",
    ]
    assert len(recovered_time_axes) == 1
    assert recovered_time_axes[0][-1] > 600.0
    assert recovered_time_axes[0][-1] != pytest.approx(465.0)
    assert not np.allclose(
        np.diff(recovered_time_axes[0]),
        np.diff(recovered_time_axes[0])[0],
    )
    output_pdf = output_dir / "dp_rl_actual_comparison.pdf"
    assert output_pdf.is_file()
    assert output_pdf.stat().st_size > 0
    output_table = output_dir / "dp_rl_actual_comparison_table.md"
    assert output_table.is_file()
    assert output_table.stat().st_size > 0
    table_text = output_table.read_text(encoding="utf-8")
    assert "PPO-PIRS (proposed)" in table_text
    assert "Total energy (kWh)" in table_text
    assert "Cumulative acceleration variation (m/s²)" in table_text
    assert "—" in table_text


def test_cli_accepts_repeated_rl_baselines() -> None:
    args = _build_cli_parser().parse_args(
        [
            "--rl-model-dir",
            "output/proposed",
            "--baseline-rl",
            "PPO-BR=output/ppo_br/best",
            "--baseline-rl",
            "PPO=output/ppo/best",
        ]
    )

    assert [_parse_baseline_spec(raw) for raw in args.baseline_rl] == [
        ("PPO-BR", "output/ppo_br/best"),
        ("PPO", "output/ppo/best"),
    ]


@pytest.mark.parametrize("raw", ("output/ppo_br/best", "=output/ppo_br", "PPO-BR="))
def test_baseline_spec_requires_label_and_directory(raw: str) -> None:
    with pytest.raises(ValueError, match="LABEL=DIR"):
        _ = _parse_baseline_spec(raw)


def test_comparison_figure_legend_lists_all_supplied_trajectories() -> None:
    figure, axes = _create_comparison_axes()
    entries = [
        ("DP", "#181818", "-"),
        ("PPO-CR", "#009E73", ":"),
        ("PPO-PIRS (proposed)", "#ED7D31", "--"),
        ("Recorded operation", "#7B61A8", "-."),
    ]

    _finalize_comparison_figure(figure, axes, entries)

    legend = figure.legends[0]
    assert [text.get_text() for text in legend.texts] == [e[0] for e in entries]
    assert [h.get_color() for h in legend.legend_handles] == [e[1] for e in entries]
    plt.close(figure)


def test_min_limit_margin_uses_moving_samples_against_step_limit() -> None:
    safeguard = SimpleNamespace(
        speed_limits=np.asarray([20.0, 10.0]),
        speed_limit_intervals=np.asarray([0.0, 100.0]),
        gamma=0.5,
    )
    profile = SpeedProfile(
        "RL",
        np.asarray([0.0, 100.0, 200.0]),
        np.asarray([0.0, 6.0, 0.0]),
        np.asarray([0.0, 10.0, 20.0]),
        200.0,
    )

    margin = compute_min_limit_margin_kmh(profile, safeguard)

    # After 100 m the limit is 10 * 0.5 = 5 m/s while the speed is near 6 m/s.
    assert margin == pytest.approx((5.0 - 6.0) * 3.6, abs=0.1)


def test_table_flags_energy_obtained_outside_tolerance() -> None:
    table = format_comparison_table(
        [
            ("DP", ProfileMetrics(-8.3, 0.0, 100.0, 1.0, True, 0.0)),
            ("PPO-CR", ProfileMetrics(132.0, 0.1, 90.0, 0.5, False, 2.7)),
            ("Recorded operation", ProfileMetrics(4.6, 0.0, 200.0, None, True, 12.4)),
        ]
    )

    assert "-8.300" in table and "+132.000" in table
    assert "Within stop/time tolerance" in table
    assert "55.00^a" in table  # (200 - 90) / 200
    assert "-10.00^a" in table  # (90 - 100) / 100
    assert "50.00 " in table  # DP vs Actual, no dagger
    assert "Min. margin to line speed limit (km/h)" in table
