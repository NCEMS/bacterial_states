"""Matplotlib conventions shared by the figures (Agg backend, seaborn 'talk' context)."""
from __future__ import annotations

from pathlib import Path

DEFAULT_FORMATS = ("pdf", "png")
UNLABELLED = "Unlabeled"
GREY = (0.7, 0.7, 0.7)


def apply_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("talk")
    plt.rcParams["figure.dpi"] = 110


def savefig(fig, out_dir: str | Path, name: str, formats=DEFAULT_FORMATS, dpi: int = 300) -> list:
    """Write ``<out_dir>/<name>.<fmt>`` for every format (tight bounding box)."""
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def pretty(col: str) -> str:
    return col.replace("_", " ").title()
