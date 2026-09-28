# 绘图迁移回归
from dataclasses import replace
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mtto.domain.speed_profile import SpeedProfile
from paper.figures import load_paper_scenario, load_paper_task
from paper.plotting import (
    profiles,
    style as plot_utils,
)


@pytest.fixture(autouse=True)
def _restore_rcparams():
    original = plt.rcParams.copy()
    yield
    plt.rcParams.update(original)


def _set_available_fonts(
    monkeypatch: pytest.MonkeyPatch, font_names: list[str]
) -> None:
    fake_fonts = [SimpleNamespace(name=name) for name in font_names]
    monkeypatch.setattr(plot_utils.fontManager, "ttflist", fake_fonts)


def test_pick_selected_font_uses_preferred_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_available_fonts(monkeypatch, ["Arial", "Liberation Sans"])

    selected = plot_utils._pick_selected_or_first_available_font(
        plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
        preferred_font="Arial",
    )

    assert selected == "Arial"


def test_pick_selected_font_falls_back_to_compatible_sans(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_available_fonts(monkeypatch, ["Liberation Sans"])

    selected = plot_utils._pick_selected_or_first_available_font(
        plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
        preferred_font="Arial",
    )

    assert selected == "Liberation Sans"


def test_pick_selected_font_raises_after_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_available_fonts(monkeypatch, [])

    with pytest.raises(ValueError, match="已尝试替代字体仍不可用"):
        _ = plot_utils._pick_selected_or_first_available_font(
            plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
            preferred_font="Arial",
        )


def test_pick_selected_font_raises_for_unknown_font(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["DejaVu Sans"])

    with pytest.raises(ValueError, match="不在候选字体中"):
        _ = plot_utils._pick_selected_or_first_available_font(
            plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
            preferred_font="MyCommercialFont",
        )


def test_set_global_plot_style_applies_fallback_font(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["Liberation Sans", "Noto Sans CJK SC"])

    style = plot_utils.set_global_plot_style(
        font_preset="sci",
        preferred_font="Arial",
    )

    assert style["font"] == "Liberation Sans"
    assert style["font_family"] == ("Liberation Sans", "Noto Sans CJK SC")
    assert plt.rcParams["font.family"] == [
        "Liberation Sans",
        "Noto Sans CJK SC",
    ]
    assert plt.rcParams["mathtext.fontset"] == "custom"
    assert plt.rcParams["mathtext.rm"] == "Liberation Sans"
    assert plt.rcParams["mathtext.it"] == "Liberation Sans:italic"
    assert plt.rcParams["mathtext.bf"] == "Liberation Sans:bold"
    assert plt.rcParams["mathtext.sf"] == "Liberation Sans"
    assert plt.rcParams["mathtext.fallback"] == "stixsans"


def test_set_chinese_font_keeps_arial_first_and_adds_cjk_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_fonts(monkeypatch, ["Arial", "Noto Sans CJK SC"])

    plot_utils.set_chinese_font()

    assert plt.rcParams["font.family"] == ["Arial", "Noto Sans CJK SC"]
    assert plt.rcParams["mathtext.fontset"] == "custom"
    assert plt.rcParams["mathtext.rm"] == "Arial"


def test_sci_figure_layout_uses_standard_column_width_and_margins() -> None:
    figure, _axis = plt.subplots()
    try:
        plot_utils.apply_sci_figure_layout(
            figure,
            columns=2,
            height_in=4.8,
            left=0.10,
            right=0.97,
            bottom=0.12,
            top=0.90,
            wspace=0.25,
            hspace=0.30,
        )

        assert figure.get_size_inches() == pytest.approx((170.0 / 25.4, 4.8))
        assert figure.subplotpars.left == pytest.approx(0.10)
        assert figure.subplotpars.right == pytest.approx(0.97)
        assert figure.subplotpars.bottom == pytest.approx(0.12)
        assert figure.subplotpars.top == pytest.approx(0.90)
        assert figure.subplotpars.wspace == pytest.approx(0.25)
        assert figure.subplotpars.hspace == pytest.approx(0.30)
    finally:
        plt.close(figure)


def test_concatenate_curves_and_draw_regions() -> None:
    first = np.array([[0.0, 1.0], [10.0, 20.0]])
    second = np.array([[2.0, 3.0], [30.0, 40.0]])
    x, y = profiles.concatenate_curves_with_NaN([first, second])
    np.testing.assert_allclose(x[[0, 1, 3, 4]], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(y[[0, 1, 3, 4]], [10.0, 20.0, 30.0, 40.0])
    assert np.isnan(x[[2, 5]]).all()
    figure, axis = plt.subplots()
    try:
        profiles.draw_regions(
            axis,
            [np.array([[0.0, 1.0], [2.0, 2.0]])],
            [np.array([[0.0, 1.0], [0.0, 0.0]])],
            "region",
            "red",
            0.5,
        )
        assert len(axis.collections) == 1
        assert axis.collections[0].get_label() == "region"
    finally:
        plt.close(figure)


def test_safeguard_render_layers() -> None:
    safeguard = load_paper_scenario().safeguard
    figure, axis = plt.subplots()
    try:
        profiles.render_safeguard(safeguard, axis, layers=profiles.DANGER_VIEW_LAYERS)
        assert axis.lines or axis.collections
        profiles.render_safeguard(
            safeguard, axis, layers=profiles.FULL_CURVE_VIEW_LAYERS
        )
        with pytest.raises(ValueError, match="Unknown render layers"):
            profiles.render_safeguard(safeguard, axis, layers=("unknown-layer",))
        for layers in (
            ("min_curve_part", "min_curve_full"),
            ("max_curve_part", "max_curve_full"),
        ):
            with pytest.raises(ValueError, match="mutually exclusive"):
                profiles.render_safeguard(safeguard, axis, layers=layers)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("render", "kwargs", "label"),
    [
        (profiles.render_rl_curve_on_axes, {}, "RL final trajectory"),
        (profiles.render_rl_curve_on_axes, {"is_best": True}, "RL best trajectory"),
        (profiles.render_dp_curve_on_axes, {}, "DP optimized speed curve"),
    ],
)
def test_profile_wrappers_keep_default_plot_behavior(render, kwargs, label) -> None:
    figure, axis = plt.subplots()
    try:
        profile = SpeedProfile.from_arrays(
            position_m=np.array([0.0, 10.0]),
            speed_mps=np.array([0.0, 1.0]),
            time_s=np.array([0.0, 1.0]),
            propulsion_energy_kj=np.array([0.0, 0.0]),
            levitation_energy_kj=np.array([0.0, 0.0]),
        )
        render(
            ax=axis,
            profile=profile,
            no_safeguard=True,
            factor=0.97,
            render_endpoints=False,
            **kwargs,
        )
        assert axis.lines[0].get_label() == label
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.0, 3.6])
        assert axis.get_xlim() == (0.0, 30000.0)
        assert axis.get_ylim() == (0.0, 500.0)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("filename", ["figure", "figure.png", "figure.pdf"])
def test_save_sci_figure_normalizes_pdf_and_uses_fixed_export_contract(
    tmp_path, filename: str
) -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object]]] = []

        def savefig(self, path, **kwargs) -> None:
            self.calls.append((path, kwargs))
            path.write_bytes(b"image")

    figure = FakeFigure()
    output = tmp_path / "paper" / filename
    expected = tmp_path / "paper" / "figure.pdf"

    saved = plot_utils.save_sci_figure(figure, output)

    assert saved == expected
    assert figure.calls == [
        (
            expected,
            {
                "dpi": 1200.0,
            },
        )
    ]


def test_apply_sci_curve_style_sets_compact_font_sizes() -> None:
    style = plot_utils.apply_sci_curve_style(
        title_font_size=9.0,
        axis_label_font_size=8.5,
        tick_font_size=7.5,
        legend_font_size=7.0,
    )
    assert style["title_font_size"] == 9.0
    assert style["axis_label_font_size"] == 8.5
    assert style["tick_font_size"] == 7.5
    assert style["legend_font_size"] == 7.0
    assert plt.rcParams["axes.titlesize"] == 9.0
    assert plt.rcParams["axes.labelsize"] == 8.5
    assert plt.rcParams["xtick.labelsize"] == 7.5
    assert plt.rcParams["ytick.labelsize"] == 7.5
    assert plt.rcParams["legend.fontsize"] == 7.0
    assert plt.rcParams["savefig.dpi"] == pytest.approx(1200.0)
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    assert plt.rcParams["lines.linewidth"] == pytest.approx(1.6)
    assert plt.rcParams["axes.axisbelow"] is True
    assert plt.rcParams["grid.color"] == "#D9D9D9"
    assert plt.rcParams["grid.linestyle"] == "--"
    assert plt.rcParams["grid.linewidth"] == pytest.approx(0.6)
    assert plt.rcParams["grid.alpha"] == pytest.approx(1.0)


def test_add_panel_label_places_text_on_axes() -> None:
    fig, ax = plt.subplots()
    try:
        # Test keyword-argument usage
        text1 = plot_utils.add_panel_label(ax=ax, label="(a)")
        assert text1.get_text() == "(a)"
        assert text1.get_position() == (0.02, 0.98)
        assert text1.get_weight() in ("bold", 700)

        # Test positional-argument usage
        text2 = plot_utils.add_panel_label(ax, "(b)", x=0.05, y=0.95, fontsize=12.0)
        assert text2.get_text() == "(b)"
        assert text2.get_position() == (0.05, 0.95)
        assert text2.get_fontsize() == 12.0
    finally:
        plt.close(fig)


def test_render_trajectory_on_axes_preserves_shared_plot_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, object]] = []

    def fake_render_safeguard(safeguard, ax, *, layers=None, **kwargs) -> None:
        calls.append((safeguard, ax, layers))

    monkeypatch.setattr(profiles, "render_safeguard", fake_render_safeguard)

    figure, axis = plt.subplots()
    safeguard = object()
    task = replace(load_paper_task(), start_position_m=10.0, target_position_m=30.0)
    try:
        profiles.render_trajectory_on_axes(
            ax=axis,
            pos_arr=[10.0, 20.0, 30.0],
            speed_arr=[1.0, 2.0, 3.0],
            task=task,
            safeguard=safeguard,
            factor=0.99,
            curve_color="purple",
            curve_label="trajectory",
            alpha=0.7,
            linewidth=2.0,
            xlim=(5.0, 35.0),
            ylim=(0.0, 15.0),
        )

        assert len(calls) == 1
        assert calls[0][1] is axis
        assert calls[0][2] == profiles.DANGER_VIEW_LAYERS
        line = axis.lines[0]
        np.testing.assert_allclose(line.get_xdata(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(line.get_ydata(), [3.6, 7.2, 10.8])
        assert line.get_color() == "purple"
        assert line.get_label() == "trajectory"
        assert line.get_alpha() == pytest.approx(0.7)
        assert line.get_linewidth() == pytest.approx(2.0)
        assert [collection.get_label() for collection in axis.collections] == [
            "start",
            "end",
        ]
        assert axis.get_xlabel() == "Position (m)"
        assert axis.get_ylabel() == "Speed (km/h)"
        assert axis.get_xlim() == pytest.approx((5.0, 35.0))
        assert axis.get_ylim() == pytest.approx((0.0, 15.0))
        assert any(gridline.get_visible() for gridline in axis.get_xgridlines())
        gridline = axis.get_xgridlines()[0]
        assert gridline.get_color() == "#D9D9D9"
        assert gridline.get_linestyle() == "--"
        assert gridline.get_linewidth() == pytest.approx(0.6)
        assert gridline.get_alpha() == pytest.approx(1.0)
        assert axis.get_axisbelow() is True
    finally:
        plt.close(figure)


def test_render_trajectory_can_disable_safeguard_endpoints_and_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load_scenario():
        pytest.fail("safeguard must not be loaded when no_safeguard=True")

    monkeypatch.setattr(
        "paper.plotting.profiles.load_scenario",
        fail_load_scenario,
    )
    figure, axis = plt.subplots()
    task = replace(load_paper_task(), start_position_m=0.0, target_position_m=1.0)
    try:
        profiles.render_trajectory_on_axes(
            ax=axis,
            pos_arr=[0.0, 1.0],
            speed_arr=[0.0, 1.0],
            task=task,
            factor=0.99,
            no_safeguard=True,
            render_endpoints=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None,
            grid=False,
        )

        assert len(axis.lines) == 1
        assert len(axis.collections) == 0
        assert axis.get_xlabel() == ""
        assert axis.get_ylabel() == ""
        assert not any(gridline.get_visible() for gridline in axis.get_xgridlines())
    finally:
        plt.close(figure)
