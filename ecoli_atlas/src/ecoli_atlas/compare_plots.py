"""Figures for `ecoli_atlas compare`: how the new samples relate to each other and to the atlas.

  fig_query_distances   heatmap of the distances between the new samples
  fig_umap_groups       atlas UMAP with the new samples coloured by group (needs ``--umap``)
  fig_factor_by_group   predicted class probability per factor along the ordered groups
  fig_neighbor_labels   what the atlas neighbours of each group are (curated labels, studies)

The last two need groups (``--group-by``). Colours: one blue ramp for ordered (numeric) groups, a
fixed categorical order otherwise; at most eight categories get a colour, the rest are "other".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .figures.plotting import savefig

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")       # light -> dark, one hue
ORDINAL = RAMP[1:]                                                    # ordered marks: skip the palest step
OTHER, ATLAS_GREY, INK, MUTED = "#b5b4ac", "#d9d8d2", "#2b2b29", "#6f6e68"
MAX_CATEGORIES = len(CATEGORICAL)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
    return plt


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(colors))


def is_ordered(order) -> bool:
    """True when every group label is a number (a time, a dose): colour with one ramp."""
    vals = pd.to_numeric(pd.Series(list(order), dtype=object), errors="coerce")
    return len(vals) > 0 and bool(vals.notna().all())


def group_colors(order, ordered: bool | None = None) -> dict:
    """Colour per group: a light-to-dark blue ramp for ordered groups, else the categorical slots
    in the given order; categories beyond the eighth share the "other" grey."""
    order = list(order)
    ordered = is_ordered(order) if ordered is None else ordered
    if ordered and len(order) > 1:
        cm = _cmap(ORDINAL, "ordinal")
        return {g: cm(i / (len(order) - 1)) for i, g in enumerate(order)}
    return {g: (CATEGORICAL[i] if i < MAX_CATEGORIES else OTHER) for i, g in enumerate(order)}


def _sample_order(index, labels, order) -> list:
    if labels is None:
        return list(index)
    pos = {g: i for i, g in enumerate(order)}
    lab = labels.reindex(index)
    return sorted(index, key=lambda s: (pos.get(lab[s], len(pos)), str(s)))


def plot_query_distances(distances: pd.DataFrame, labels=None, order=None, group_name: str | None = None):
    plt = _plt()
    ids = _sample_order(list(distances.index), labels, order)
    D = distances.loc[ids, ids].values
    n = len(ids)
    side = float(np.clip(0.22 * n + 2.2, 4.5, 13))
    fig, ax = plt.subplots(figsize=(side + 0.8, side))
    im = ax.imshow(D, cmap=_cmap(RAMP, "seq"), vmin=0, vmax=float(D.max()) if D.max() > 0 else 1.0, interpolation="nearest")
    fs = 8 if n <= 24 else 6 if n <= 48 else 4
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ids, rotation=90, fontsize=fs)
    ax.set_yticklabels(ids, fontsize=fs)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if labels is not None:
        lab = labels.reindex(ids).values
        for i in range(1, n):
            if lab[i] != lab[i - 1]:
                ax.axhline(i - 0.5, color="white", lw=1.5)
                ax.axvline(i - 0.5, color="white", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("distance between samples")
    cb.outline.set_visible(False)
    ax.set_title("Distances between the new samples" + (f" (ordered by {group_name})" if group_name and labels is not None else ""))
    return fig


def plot_umap_groups(atlas_xy: pd.DataFrame, query_xy: pd.DataFrame, color=None, color_order=None,
                     color_name: str = "group", labels=None, order=None, trajectory: bool = False):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.scatter(atlas_xy["umap1"], atlas_xy["umap2"], s=4, c=ATLAS_GREY, linewidths=0, label=f"atlas (n={len(atlas_xy)})", zorder=1)
    if color is None:
        ax.scatter(query_xy["umap1"], query_xy["umap2"], s=42, c=CATEGORICAL[0], edgecolor="white", linewidth=0.8,
                   label=f"new samples (n={len(query_xy)})", zorder=3)
    else:
        col = group_colors(color_order)
        c = color.reindex(query_xy.index)
        shown_other = False
        for i, g in enumerate(color_order):
            m = (c == g).values
            name = str(g)
            if col[g] == OTHER:
                name = None if shown_other else "other"
                shown_other = True
            ax.scatter(query_xy.loc[m, "umap1"], query_xy.loc[m, "umap2"], s=42, color=col[g], edgecolor="white",
                       linewidth=0.8, label=name, zorder=3)
    if trajectory and labels is not None and order is not None and len(order) > 1:
        lab = labels.reindex(query_xy.index)
        cen = query_xy.groupby(lab.values)[["umap1", "umap2"]].mean().reindex(order).dropna()
        ax.plot(cen["umap1"], cen["umap2"], color=INK, lw=1.2, zorder=4)
        ax.annotate("", xy=cen.iloc[-1].values, xytext=cen.iloc[-2].values,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=5)
        for g in (cen.index[0], cen.index[-1]):
            ax.annotate(str(g), cen.loc[g].values, xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK, zorder=6)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(title=None if color is None else color_name, fontsize=7.5, title_fontsize=8, frameon=False,
              loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("New samples on the atlas UMAP" + (" (line: group centroids in order)" if trajectory and labels is not None else ""))
    return fig


def _prob_columns(predictions: pd.DataFrame, factors) -> dict:
    """Per factor, the probability column to show: the last class in sorted order (for the usual
    two-class factors that is `stationary`, `defined`, `True`)."""
    out = {}
    for f in factors:
        cols = sorted(c for c in predictions.columns if c.startswith(f"{f}_prob_"))
        if cols:
            out[f] = cols[-1]
    return out


def plot_factor_by_group(predictions: pd.DataFrame, labels: pd.Series, order, factors, group_name: str = "group"):
    plt = _plt()
    preds = predictions.copy()
    preds.index = preds.index.astype(str)
    shown = _prob_columns(preds, factors)
    if not shown:
        return None
    lab = labels.reindex(preds.index)
    pos = {g: i for i, g in enumerate(order)}
    x = lab.map(pos)
    fig, axes = plt.subplots(len(shown), 1, figsize=(max(5.5, 0.55 * len(order) + 3), 2.3 * len(shown)), sharex=True, squeeze=False)
    rng = np.random.RandomState(0)
    for ax, (f, c) in zip(axes[:, 0], shown.items()):
        ok = x.notna() & preds[c].notna()
        jitter = rng.uniform(-0.12, 0.12, int(ok.sum()))
        ax.scatter(x[ok].values + jitter, preds.loc[ok, c].values, s=22, color=CATEGORICAL[0], edgecolor="white", linewidth=0.6, zorder=3)
        mean = preds.loc[ok, c].groupby(x[ok].values).mean()
        ax.plot(mean.index, mean.values, color=INK, lw=1.4, zorder=2)
        ax.axhline(0.5, color=ATLAS_GREY, lw=0.8, zorder=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("probability")
        ax.set_title(f"P({f.replace('_', ' ')} = {c.split('_prob_', 1)[1]})", loc="left")
    axes[-1, 0].set_xticks(range(len(order)))
    axes[-1, 0].set_xticklabels([str(g) for g in order], rotation=0 if max(len(str(g)) for g in order) <= 5 else 45,
                                ha="center" if max(len(str(g)) for g in order) <= 5 else "right")
    axes[-1, 0].set_xlabel(group_name)
    fig.suptitle("Encoder calls along the groups (dots: samples, line: group mean)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def neighbor_composition(neighbors: pd.DataFrame, values: pd.Series, labels: pd.Series, order) -> pd.DataFrame:
    """Fraction of each group's pooled neighbours per value (rows: groups in order). The most
    common values keep their own column, the rest are summed into ``other``; unlabelled neighbours
    are left out."""
    df = pd.DataFrame({"group": neighbors["query"].map(labels.to_dict()), "value": values.values}).dropna()
    if df.empty:
        return pd.DataFrame(index=list(order))
    df["value"] = df["value"].astype(str)
    tab = pd.crosstab(df["group"], df["value"], normalize="index").reindex(list(order)).fillna(0.0)
    keep = list(tab.sum().sort_values(ascending=False).index)
    if len(keep) > MAX_CATEGORIES:
        head = keep[:MAX_CATEGORIES - 1]
        tab = tab[head].assign(other=tab[keep[MAX_CATEGORIES - 1:]].sum(axis=1))
    else:
        tab = tab[keep]
    return tab


def plot_neighbor_labels(neighbors: pd.DataFrame, atlas, labels: pd.Series, order, factors, group_name: str = "group"):
    plt = _plt()
    panels = []
    for f in factors:
        src = atlas.label_source(f)
        if src is not None:
            panels.append((f"neighbours' {f.replace('_', ' ')}", neighbors["atlas_sample"].map(src.to_dict())))
    if "gse" in neighbors.columns and neighbors["gse"].notna().any():
        panels.append(("neighbours' study", neighbors["gse"]))
    panels = [(t, neighbor_composition(neighbors, v, labels, order)) for t, v in panels]
    panels = [(t, tab) for t, tab in panels if tab.shape[1] > 0]
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(6.0, 0.55 * len(order) + 4.2), 2.2 * len(panels)), sharex=True, squeeze=False)
    xs = np.arange(len(order))
    for ax, (title, tab) in zip(axes[:, 0], panels):
        bottom = np.zeros(len(order))
        for i, col in enumerate(tab.columns):
            color = OTHER if col == "other" and i == len(tab.columns) - 1 and len(tab.columns) == MAX_CATEGORIES else CATEGORICAL[i]
            ax.bar(xs, tab[col].values, bottom=bottom, width=0.72, color=color, edgecolor="white", linewidth=1.0, label=str(col))
            bottom += tab[col].values
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("fraction")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    long = max(len(str(g)) for g in order) > 5
    axes[-1, 0].set_xticks(xs)
    axes[-1, 0].set_xticklabels([str(g) for g in order], rotation=45 if long else 0, ha="right" if long else "center")
    axes[-1, 0].set_xlabel(group_name)
    fig.suptitle("Atlas neighbours of each group", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def write_all(out_dir, atlas, neighbors: pd.DataFrame, summary: pd.DataFrame, distances: pd.DataFrame,
              predictions: pd.DataFrame | None = None, labels: pd.Series | None = None, order=None,
              color: pd.Series | None = None, color_order=None, color_name: str = "group",
              group_name: str | None = None, atlas_xy: pd.DataFrame | None = None, query_xy: pd.DataFrame | None = None,
              factors=(), formats=("pdf", "png"), trajectory: bool | None = None) -> list:
    """Write every figure the inputs allow; returns the names written."""
    out = Path(out_dir)
    written = []
    group_name = group_name or "group"

    def save(fig, name):
        if fig is not None:
            savefig(fig, out, name, formats=formats, dpi=200)
            written.append(name)

    if len(distances) > 1:
        save(plot_query_distances(distances, labels, order, group_name), "fig_query_distances")
    if atlas_xy is not None and query_xy is not None:
        traj = (labels is not None and is_ordered(order)) if trajectory is None else trajectory
        save(plot_umap_groups(atlas_xy, query_xy, color, color_order, color_name, labels, order, traj), "fig_umap_groups")
    if labels is not None:
        if predictions is not None:
            save(plot_factor_by_group(predictions, labels, order, factors, group_name), "fig_factor_by_group")
        save(plot_neighbor_labels(neighbors, atlas, labels, order, factors, group_name), "fig_neighbor_labels")
    return written
