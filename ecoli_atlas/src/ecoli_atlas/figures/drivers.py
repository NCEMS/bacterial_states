"""Figure of `ecoli_atlas drivers`: the genes with the largest attributions of one contrast."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import GREY, savefig


UP, DOWN = "#b2182b", "#2166ac"


def fig_drivers(table: pd.DataFrame, head: str, low: str, high: str, out_dir, formats, top: int = 25,
                label: str = "symbol", name: str = "drivers") -> None:
    """The genes whose change moves the score most towards the high class. Left: those that are
    higher in the high class; right: those that are lower. Bars are the attribution; the number
    beside a bar is the log2 fold change in the data. With several seeds the bar is the mean, the
    line its standard deviation, and a row that is among the top rows in fewer than three of four
    seeds is drawn pale."""
    import matplotlib.pyplot as plt
    pos = table[table["attribution"] > 0]
    parts = [(pos[pos["direction"] == "up"].head(top), f"higher in {high}", UP),
             (pos[pos["direction"] == "down"].head(top), f"lower in {high}", DOWN)]
    n = max(len(p[0]) for p in parts)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.0, 0.34 * n + 1.8)))
    xmax = float(pos["attribution"].max()) if len(pos) else 1.0
    for ax, (sub, title, color) in zip(axes, parts):
        names = sub[label] if label in sub.columns else sub["feature"]
        if "feature" in sub.columns:
            names = names.where(names != "", sub["feature"])
        y = np.arange(len(sub))[::-1]
        solid = np.ones(len(sub), bool) if "seeds_in_top" not in sub.columns else (sub["seeds_in_top"] >= 0.75 * sub["n_seeds"]).to_numpy()
        ax.barh(y, sub["attribution"], color=color, height=0.75, alpha=None)
        for bar, ok in zip(ax.patches[-len(sub):] if len(sub) else [], solid):
            bar.set_alpha(1.0 if ok else 0.35)                     # pale: among the top rows in fewer than 3 of 4 seeds
        if "attribution_sd" in sub.columns:
            ax.errorbar(sub["attribution"], y, xerr=sub["attribution_sd"], fmt="none", ecolor="black", elinewidth=0.8)
        for yi, a, fc in zip(y, sub["attribution"], sub["log2fc"]):
            ax.text(a + 0.01 * xmax, yi, f"{fc:+.1f}", va="center", fontsize=9, color=GREY)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=11, style="italic")
        ax.set_xlim(0, 1.15 * xmax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"attribution to the {head} score ({low} → {high})", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out_dir, f"fig_{name}_{head}_{low}_to_{high}", formats)
