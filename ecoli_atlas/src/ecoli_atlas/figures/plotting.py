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


RED, BLUE = "#de2d26", "#2c7fb8"


def tab20_pairs(cats) -> tuple[dict, dict]:
    """Dark / light tab20 colour per category (consecutive tab20 pairs)."""
    import matplotlib.pyplot as plt
    dark = {c: plt.cm.tab20(i * 2) for i, c in enumerate(cats)}
    light = {c: plt.cm.tab20(i * 2 + 1) for i, c in enumerate(cats)}
    return dark, light


def correlate_barh(ax, corr, n_each: int, title: str, xlabel: str, fontsize: int = 8, xlim=None):
    """Horizontal bars of the ``n_each`` strongest positive (red) and negative (blue) correlates."""
    import pandas as pd
    corr = corr.sort_values()
    top = pd.concat([corr.tail(n_each).iloc[::-1], corr.head(n_each)])
    ax.barh(range(len(top)), top.values, color=[RED if v > 0 else BLUE for v in top.values])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=fontsize)
    ax.invert_yaxis()
    ax.axhline(n_each - 0.5, color="k", lw=.5, ls=":")
    ax.axvline(0, color="k", lw=.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    return top


def spearman_series(X, y, genes):
    """Spearman rho of every column of X with y, indexed by ``genes``; constant columns get 0."""
    import numpy as np
    import pandas as pd
    from ..utils import spearman_vec
    return pd.Series(spearman_vec(np.asarray(X, dtype=float), np.asarray(y, dtype=float)), index=list(genes)).fillna(0.0)
