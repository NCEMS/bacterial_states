"""Corpus composition pies (notebook cell 11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import pretty, savefig

PLOT_VARS = ["growth_phase_c", "medium_type_c", "carbon_source_c", "strain_background_c", "is_lab_strain_c", "stress"]
MIN_WEDGE_SHARE = 0.10   # categories below this share are grouped as "Other"


def composition_tables(meta: pd.DataFrame) -> dict:
    out = {}
    for col in PLOT_VARS:
        if col not in meta.columns:
            continue
        series = meta.loc[meta["stress"] != "", col] if col == "stress" else meta[col].dropna()
        if series.empty:
            out[col] = {"n": 0, "wedges": {}}
            continue
        vc = series.astype(str).value_counts()
        share = vc / vc.sum()
        keep, small = vc[share >= MIN_WEDGE_SHARE], vc[share < MIN_WEDGE_SHARE]
        data = pd.concat([keep, pd.Series({"Other": small.sum()})]) if len(small) else keep
        out[col] = {"n": int(series.shape[0]), "wedges": {str(k): int(v) for k, v in data.items()}}
    return out


def fig_corpus_composition(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    meta = ctx.meta
    tables = composition_tables(meta)
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
    fig.tight_layout()
    savefig(fig, out_dir, "fig_corpus_composition", formats)
    return {"n_samples": int(len(meta)), "composition": tables}
