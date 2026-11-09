"""Matplotlib conventions shared by the figures and case studies (Agg backend, notebook style)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FORMATS = ("pdf", "png")
RED, BLUE = "#de2d26", "#2c7fb8"


def apply_style() -> None:
    """Headless backend plus the notebook's seaborn 'talk' context."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("talk")
    plt.rcParams["figure.dpi"] = 110


def savefig(fig, out_dir: str | Path, name: str, formats=DEFAULT_FORMATS, dpi: int = 300) -> list:
    """Write ``<out_dir>/<name>.<fmt>`` for every format (tight bounding box, like the notebook)."""
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
    base = col[:-2] if col.endswith("_c") else col
    return base.replace("_", " ").title()


def tab20_pairs(cats) -> tuple[dict, dict]:
    """Dark / light tab20 colour per category (consecutive tab20 pairs)."""
    import matplotlib.pyplot as plt
    dark = {c: plt.cm.tab20(i * 2) for i, c in enumerate(cats)}
    light = {c: plt.cm.tab20(i * 2 + 1) for i, c in enumerate(cats)}
    return dark, light


def top_and_bottom(corr: pd.Series, n_each: int) -> pd.Series:
    """The ``n_each`` most positive (descending) followed by the ``n_each`` most negative values."""
    corr = corr.sort_values()
    return pd.concat([corr.tail(n_each).iloc[::-1], corr.head(n_each)])


def correlate_barh(ax, corr: pd.Series, n_each: int, title: str, xlabel: str, fontsize: int = 8,
                   xlim=None) -> pd.Series:
    """Horizontal bars of the strongest positive (red) and negative (blue) correlates."""
    top = top_and_bottom(corr, n_each)
    ax.barh(range(len(top)), top.values, color=[RED if v > 0 else BLUE for v in top.values])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=fontsize)
    ax.invert_yaxis()
    ax.axhline(n_each - 0.5, color="k", lw=.5, ls=":")
    ax.axvline(0, color="k", lw=.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=10 if "\n" not in title else 11)
    ax.set_xlabel(xlabel)
    return top


def spearman_series(X: np.ndarray, y: np.ndarray, genes) -> pd.Series:
    """Spearman rho of every column of X with y; constant columns get 0, as in the notebook."""
    from ..utils import spearman_vec
    return pd.Series(spearman_vec(np.asarray(X, dtype=float), np.asarray(y, dtype=float)), index=list(genes)).fillna(0.0)
