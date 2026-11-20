"""Figures of `ecoli_atlas axes`: where the categories of an ordered axis lie in the embedding."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import GREY, pretty, savefig

METHOD_LABEL = {"ordinal": "ordinal score", "anchor": "anchor score"}


def _colors(categories: list) -> dict:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    return {c: cmap(i / max(len(categories) - 1, 1)) for i, c in enumerate(categories)}


def fig_axis_distributions(scores: pd.DataFrame, spec, thresholds: dict, out_dir, formats) -> None:
    """One panel per method: the score of every sample by category. Samples the embedding has not
    fitted (TEST, projected) are drawn with a black edge; held-out categories are hatched in the
    tick label with an asterisk; dashed lines are the ordinal thresholds."""
    import matplotlib.pyplot as plt
    cats = [c for c in spec.categories if (scores["category"] == c).any()]
    colors = _colors(spec.categories)
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, 2, figsize=(1.5 * len(cats) + 9, 6), squeeze=False)
    for ax, method in zip(axes[0], ("ordinal", "anchor")):
        for i, c in enumerate(cats):
            sub = scores[(scores["category"] == c) & scores[method].notna()]
            x = i + rng.uniform(-0.3, 0.3, len(sub))
            seen = (sub["split"] == 1).to_numpy()
            ax.scatter(x[seen], sub[method].to_numpy()[seen], s=9, color=colors[c], alpha=0.45, linewidths=0)
            ax.scatter(x[~seen], sub[method].to_numpy()[~seen], s=16, color=colors[c], edgecolors="black", linewidths=0.6)
            if len(sub):
                ax.hlines(sub[method].median(), i - 0.38, i + 0.38, color="black", lw=2)
        if method == "ordinal":
            for t in thresholds.values():
                ax.axhline(t, color=GREY, ls="--", lw=1)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axhline(1, color=GREY, lw=0.8)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{c.replace('_', ' ')}{'*' if c in spec.holdout else ''}\n(n={int((scores['category'] == c).sum())})" for c in cats],
                           rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(METHOD_LABEL[method])
        ax.set_title(f"{pretty(spec.name)}: {METHOD_LABEL[method]}", fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, f"fig_axis_{spec.name}_distributions", formats)


def fig_axis_series(scores: pd.DataFrame, series: pd.DataFrame, spec, out_dir, formats, method: str = "ordinal") -> None:
    """One panel per study of ``series`` (columns sample, series, x and optionally x_label): the
    score against the position in the series, one line per series, points coloured by category."""
    import matplotlib.pyplot as plt
    s = series.join(scores[["gse", "category", method]], on="sample", how="inner").dropna(subset=[method])
    studies = [g for g in s["gse"].unique()]
    if not studies:
        return
    colors = _colors(spec.categories)
    ncol = min(3, len(studies))
    nrow = int(np.ceil(len(studies) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow), squeeze=False)
    for ax in axes.ravel()[len(studies):]:
        ax.axis("off")
    for ax, g in zip(axes.ravel(), studies):
        sub = s[s["gse"] == g]
        for _, line in sub.groupby("series"):
            line = line.sort_values("x")
            mean = line.groupby("x")[method].mean()
            ax.plot(mean.index, mean.to_numpy(), color=GREY, lw=1.5, zorder=1)
            ax.scatter(line["x"], line[method], c=[colors.get(c, GREY) for c in line["category"]], s=40,
                       edgecolors="black", linewidths=0.5, zorder=2)
        ax.axhline(0, color=GREY, lw=0.8, ls=":")
        ax.axhline(1, color=GREY, lw=0.8, ls=":")
        if sub["x"].min() > 0 and sub["x"].max() / sub["x"].min() > 20:
            ax.set_xscale("log")
        label = sub["x_label"].iloc[0] if "x_label" in sub.columns else "position"
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel(METHOD_LABEL[method], fontsize=11)
        ax.set_title(f"{g} (n={len(sub)})", fontsize=12)
        ax.tick_params(labelsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], markeredgecolor="black", label=c.replace("_", " "))
               for c in spec.categories if (s["category"] == c).any()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=GREY, markeredgecolor="black", label="no category"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, out_dir, f"fig_axis_{spec.name}_series", formats)


def fig_axes_map(scores_x: pd.DataFrame, spec_x, scores_y: pd.DataFrame, spec_y, out_dir, formats, method: str = "ordinal") -> None:
    """Two axes against each other, twice: coloured by the categories of the first and of the second axis."""
    import matplotlib.pyplot as plt
    both = pd.DataFrame({"x": scores_x[method], "y": scores_y[method], "cx": scores_x["category"], "cy": scores_y["category"]}).dropna(subset=["x", "y"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, col, spec in ((axes[0], "cx", spec_x), (axes[1], "cy", spec_y)):
        colors = _colors(spec.categories)
        none = both[both[col] == ""]
        ax.scatter(none["x"], none["y"], s=4, color=GREY, alpha=0.3, linewidths=0)
        for c in spec.categories:
            sub = both[both[col] == c]
            ax.scatter(sub["x"], sub["y"], s=8, color=colors[c], alpha=0.6, linewidths=0, label=f"{c.replace('_', ' ')} ({len(sub)})")
        ax.set_xlabel(f"{pretty(spec_x.name)} ({METHOD_LABEL[method]})")
        ax.set_ylabel(f"{pretty(spec_y.name)} ({METHOD_LABEL[method]})")
        ax.set_title(f"coloured by {pretty(spec.name)}", fontsize=13)
        ax.legend(fontsize=9, markerscale=2, frameon=False)
    fig.tight_layout()
    savefig(fig, out_dir, f"fig_axes_{spec_x.name}_by_{spec_y.name}", formats)
