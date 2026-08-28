import matplotlib

matplotlib.use("Agg")
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from utils import plot_utils


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
    _set_available_fonts(monkeypatch, ["Calibri", "Carlito"])

    selected = plot_utils._pick_selected_or_first_available_font(
        plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
        preferred_font="Calibri",
    )

    assert selected == "Calibri"


def test_pick_selected_font_falls_back_to_carlito(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["Carlito"])

    selected = plot_utils._pick_selected_or_first_available_font(
        plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
        preferred_font="Calibri",
    )

    assert selected == "Carlito"


def test_pick_selected_font_raises_after_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_available_fonts(monkeypatch, [])

    with pytest.raises(ValueError, match="已尝试替代字体仍不可用"):
        _ = plot_utils._pick_selected_or_first_available_font(
            plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
            preferred_font="Calibri",
        )


def test_pick_selected_font_raises_for_unknown_font(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["DejaVu Sans"])

    with pytest.raises(ValueError, match="不在候选字体中"):
        _ = plot_utils._pick_selected_or_first_available_font(
            plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
            preferred_font="MyCommercialFont",
        )


def test_set_global_plot_style_applies_fallback_font(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["Carlito"])

    style = plot_utils.set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
    )

    assert style["font"] == "Carlito"
    assert plt.rcParams["font.family"][0] == "Carlito"


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

        assert figure.get_size_inches() == pytest.approx((180.0 / 25.4, 4.8))
        assert figure.subplotpars.left == pytest.approx(0.10)
        assert figure.subplotpars.right == pytest.approx(0.97)
        assert figure.subplotpars.bottom == pytest.approx(0.12)
        assert figure.subplotpars.top == pytest.approx(0.90)
        assert figure.subplotpars.wspace == pytest.approx(0.25)
        assert figure.subplotpars.hspace == pytest.approx(0.30)
    finally:
        plt.close(figure)


def test_save_sci_figure_uses_compact_300_dpi_export(tmp_path) -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object]]] = []

        def savefig(self, path, **kwargs) -> None:
            self.calls.append((path, kwargs))
            path.write_bytes(b"image")

    figure = FakeFigure()
    output = tmp_path / "paper" / "figure.png"

    saved = plot_utils.save_sci_figure(figure, output)

    assert saved == output
    assert figure.calls == [
        (
            output,
            {
                "dpi": 300.0,
                "bbox_inches": "tight",
                "pad_inches": 0.02,
            },
        )
    ]
