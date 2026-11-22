"""Corpus composition pies: what the samples of the model are."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import pretty, savefig

# (title, column of the frame given to the figure)
PLOT_VARS = ("phase", "medium", "carbon_source", "genotype_class", "stress_family", "strain_group")
MIN_WEDGE_SHARE = 0.05   # categories below this share are grouped as "Other"


def composition_tables(frame: pd.DataFrame) -> dict:
    """column -> {"n": labelled samples, "wedges": {class: n}}. Empty values are left out; for
    ``stress_family`` only treated samples are shown (the value ``none`` is left out)."""
    out = {}
    for col in PLOT_VARS:
        if col not in frame.columns:
            continue
        series = frame[col].astype("object")
        series = series[series.notna() & (series.astype(str).str.strip() != "")]
        if col == "stress_family":
            series = series[series != "none"]
        if series.empty:
            out[col] = {"n": 0, "wedges": {}}
            continue
        vc = series.astype(str).value_counts()
        share = vc / vc.sum()
        keep, small = vc[share >= MIN_WEDGE_SHARE], vc[share < MIN_WEDGE_SHARE]
        data = pd.concat([keep, pd.Series({"Other": small.sum()})]) if len(small) else keep
        out[col] = {"n": int(series.shape[0]), "wedges": {str(k): int(v) for k, v in data.items()}}
    return out


def fig_corpus_composition(frame: pd.DataFrame, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    tables = composition_tables(frame)
    cols = [c for c in PLOT_VARS if c in tables]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 5))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        t = tables[col]
        if not t["wedges"]:
            ax.axis("off")
            continue
        data = pd.Series(t["wedges"])
        ax.pie(data, labels=[str(i) for i in data.index], autopct="%1.1f%%", startangle=140,
               colors=plt.cm.Paired.colors, textprops={"fontsize": 8})
        ax.set_title(f"{pretty(col)}\n(n={t['n']})", fontsize=11)
    fig.suptitle(f"{len(frame)} samples in {frame['GSE'].nunique()} studies", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_corpus_composition", formats)
    return {"n_samples": int(len(frame)), "composition": tables}
