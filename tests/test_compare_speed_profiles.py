import sys
from pathlib import Path

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
    _resolve_target_schedule_time,
    _validate_common_target_position,
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
        "DP optimization",
        "Proposed Method",
        "Actual operation",
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

    assert profile.label == "Actual operation"
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
                "DP optimization",
                ProfileMetrics(1.25, 0.0, 123.456, 0.123456),
            ),
            (
                "Proposed Method",
                ProfileMetrics(2.5, 0.25, 120.0, 0.1),
            ),
            (
                "Actual operation",
                ProfileMetrics(3.0, 0.5, 130.0, None),
            ),
        ]
    )

    assert "Time error (s)" in table
    assert "Stop error (m)" in table
    assert "Total energy (kWh)" in table
    assert "TAV (m/s²)" in table
    assert "123.456" in table
    assert "0.123456" in table
    assert "—" in table
    assert "不可直接比较" in table


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
            label="Actual operation",
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

    def spy_finalize(figure, axes):
        called_helpers.append("_finalize_comparison_figure")
        return orig_finalize(figure, axes)

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
    assert "Proposed Method" in table_text
    assert "Total energy (kWh)" in table_text
    assert "TAV (m/s²)" in table_text
    assert "—" in table_text
