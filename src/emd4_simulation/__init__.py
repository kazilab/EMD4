"""EMD4 simulation of senescence after exposure withdrawal."""

from pathlib import Path

FIGURE_STEM = "emd4_figure6_senescence"
FIGURE_FORMATS = ("pdf", "svg", "png", "eps", "tif")


def remove_previous_figure(outdir: Path) -> None:
    """Remove earlier Figure 6 exports from the selected run directory."""
    for ext in FIGURE_FORMATS:
        (outdir / f"figure2c_emd4_simulation.{ext}").unlink(missing_ok=True)
