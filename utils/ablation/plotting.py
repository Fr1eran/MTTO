"""Small plotting adapters shared by the ablation commands."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from utils.plot_utils import save_sci_figure


def save_ablation_figure(
    figure: Figure | None,
    output_file: str | Path | None,
    *,
    dpi: float,
    pad_inches: float = 0.05,
) -> Path | None:
    """Save a figure with the project-wide tight-layout export policy."""
    if figure is None or output_file is None:
        return None
    path = Path(output_file)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_sci_figure(figure, path, dpi=dpi, pad_inches=pad_inches)
