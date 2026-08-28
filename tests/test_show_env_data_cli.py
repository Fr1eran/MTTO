from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from scripts.show_env_data import (
    create_figures,
    load_track_environment_data,
    main,
    parse_args,
    save_compact_figures,
)


def test_show_env_data_cli_defaults():
    with patch("sys.argv", ["show_env_data"]):
        args = parse_args()
    assert args.view == "overview"
    assert args.output_file is None
    assert args.dpi == pytest.approx(300.0)
    assert args.pad_inches == pytest.approx(0.02)
    assert args.no_show is False


@pytest.mark.parametrize(
    "view_mode", ["overview", "full-curves", "danger-region", "all"]
)
def test_show_env_data_cli_accepts_view_mode(view_mode: str):
    with patch("sys.argv", ["show_env_data", "--view", view_mode, "--no-show"]):
        args = parse_args()
    assert args.view == view_mode
    assert args.no_show is True


def test_load_track_environment_data():
    data = load_track_environment_data()
    assert data.accessible_points.ndim == 1
    assert data.dangerous_points.ndim == 1
    assert data.stations_cor.shape == (2, 2)
    assert data.speed_limits.size > 0
    assert data.speed_limit_intervals.size > 0
    assert data.slopes.size > 0
    assert data.slope_intervals.size > 0


def test_create_figures_view_modes():
    data = load_track_environment_data()

    # overview
    figs_overview = create_figures("overview", data)
    assert set(figs_overview.keys()) == {"overview"}
    assert len(figs_overview["overview"].axes) == 2
    overview_axes = figs_overview["overview"].axes
    assert [
        text.get_text()
        for axis in overview_axes
        for text in axis.texts
        if text.get_text() in {"(a)", "(b)"}
    ] == ["(a)", "(b)"]

    # full-curves
    figs_full = create_figures("full-curves", data)
    assert set(figs_full.keys()) == {"full_curves"}
    assert len(figs_full["full_curves"].axes) == 1

    # danger-region
    figs_danger = create_figures("danger-region", data)
    assert set(figs_danger.keys()) == {"danger_region"}
    assert len(figs_danger["danger_region"].axes) == 1

    # all
    figs_all = create_figures("all", data)
    assert set(figs_all.keys()) == {"overview", "full_curves", "danger_region"}

    for fig_dict in [figs_overview, figs_full, figs_danger, figs_all]:
        for fig in fig_dict.values():
            plt.close(fig)


def test_save_compact_figures_single_and_multi(tmp_path: Path):
    data = load_track_environment_data()

    # 单图保存
    figs_single = create_figures("full-curves", data)
    out_single = tmp_path / "single" / "full_curve_plot.png"
    saved_single = save_compact_figures(
        figs_single,
        out_single,
        dpi=100.0,
        pad_inches=0.01,
    )
    assert len(saved_single) == 1
    assert saved_single[0] == out_single
    assert out_single.is_file()

    # 多图保存
    figs_multi = create_figures("all", data)
    out_multi = tmp_path / "multi" / "env_bundle.png"
    saved_multi = save_compact_figures(
        figs_multi,
        out_multi,
        dpi=100.0,
        pad_inches=0.01,
    )
    assert len(saved_multi) == 3
    assert (tmp_path / "multi" / "env_bundle_overview.png").is_file()
    assert (tmp_path / "multi" / "env_bundle_full_curves.png").is_file()
    assert (tmp_path / "multi" / "env_bundle_danger_region.png").is_file()

    for fig in figs_single.values():
        plt.close(fig)
    for fig in figs_multi.values():
        plt.close(fig)


def test_main_cli_execution_no_show(tmp_path: Path):
    output_png = tmp_path / "cli_test.png"
    with patch(
        "sys.argv",
        [
            "show_env_data",
            "--view",
            "full-curves",
            "--output-file",
            str(output_png),
            "--no-show",
        ],
    ):
        main()

    assert output_png.is_file()
