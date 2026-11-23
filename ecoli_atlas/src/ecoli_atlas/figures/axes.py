"""Figures of `ecoli_atlas axes`: where the categories of an ordered axis lie in the embedding."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import BLUE, GREY, RED, pretty, savefig

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


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
MAX_CATEGORICAL_X = 12           # series figures: up to this many positions are drawn as categories


def _x_label(x: float) -> str:
    return f"{x:g}"


SERIES_X = ("auto", "categorical", "linear", "log")


def _draw_series(ax, s: pd.DataFrame, col: str, seeds: pd.DataFrame | None = None, categories: list | None = None,
                 x_axis: str = "auto") -> list:
    """One panel of a series figure. ``s``: sample, series, x and ``col`` (and ``category``). ``x_axis``: categorical
    (the positions as categories, labelled with their values, the series side by side within each), linear, log, or
    auto (categorical with at most ``MAX_CATEGORICAL_X`` positions, else log when the positions span more than 20-fold
    and are all positive, else linear). One colour per
    series and a line through its points; ``seeds`` (samples x seeds) adds every seed's value as a small dot of the same
    colour, jittered around the sample. Marker shape = the sample's category of the axis (``categories``).
    Returns the legend handles of the series."""
    import matplotlib.pyplot as plt
    xs = sorted(s["x"].unique())
    names = list(dict.fromkeys(s.sort_values(["series"])["series"]))
    if x_axis not in SERIES_X:
        raise ValueError(f"x_axis must be one of {SERIES_X}, not {x_axis!r}")
    if x_axis == "log" and xs[0] <= 0:
        raise ValueError("a log x axis needs positive positions")
    categorical = x_axis == "categorical" or (x_axis == "auto" and len(xs) <= MAX_CATEGORICAL_X)
    log = x_axis == "log" or (x_axis == "auto" and not categorical and xs[0] > 0 and xs[-1] / xs[0] > 20)
    if categorical:
        pos = {x: float(i) for i, x in enumerate(xs)}
        slot = 0.75 / max(len(names), 1)
        offset = {n: (i - (len(names) - 1) / 2) * slot for i, n in enumerate(names)}
    else:
        pos = {x: float(x) for x in xs}
        offset = {n: 0.0 for n in names}

    def jitter(x0, k):
        if categorical:
            return x0 + rng.uniform(-0.3, 0.3, k) * slot
        if log:
            return x0 * np.exp(rng.uniform(-0.03, 0.03, k))
        return x0 + rng.uniform(-0.008, 0.008, k) * (xs[-1] - xs[0] or 1.0)
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    color = {n: cmap(i % cmap.N) for i, n in enumerate(names)}
    marker = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(categories or [])}
    rng = np.random.RandomState(0)
    handles = []
    for n in names:
        line = s[s["series"] == n].sort_values("x")
        px = line["x"].map(pos).to_numpy() + offset[n]
        mean = line.groupby("x")[col].mean()
        ax.plot(mean.index.map(pos).to_numpy() + offset[n], mean.to_numpy(), color=color[n], lw=1.2, alpha=0.6, zorder=1)
        if seeds is not None:
            for x0, sample in zip(px, line["sample"]):
                if sample in seeds.index:
                    v = seeds.loc[sample].dropna().to_numpy(dtype=float)
                    ax.scatter(jitter(x0, len(v)), v, s=10, color=color[n], alpha=0.55, linewidths=0, zorder=2)
        cats = line["category"] if "category" in line.columns else pd.Series("", index=line.index)
        for c in cats.unique():
            m = (cats == c).to_numpy()
            ax.scatter(px[m], line[col].to_numpy()[m], s=42, color=color[n], marker=marker.get(c, "o"), edgecolors="black",
                       linewidths=0.6, zorder=3)
        handles.append(plt.Line2D([], [], marker="o", ls="-", color=color[n], markeredgecolor="black", label=n))
    if categorical:
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([_x_label(x) for x in xs])
        ax.set_xlim(-0.6, len(xs) - 0.4)
    elif log:
        ax.set_xscale("log")
    return handles


def _short_names(names: list) -> dict:
    """Series names without the words they all start with ("GSE67402 replicate 1" -> "1")."""
    words = [n.split(" ") for n in names]
    k = 0
    while len(names) > 1 and all(len(w) > k + 1 for w in words) and len({w[k] for w in words}) == 1:
        k += 1
    return {n: " ".join(w[k:]) for n, w in zip(names, words)}


def _series_legend(fig, handles: list, extra: list) -> None:
    import matplotlib.pyplot as plt
    seen, uniq = set(), []
    for h in handles:
        if h.get_label() not in seen:
            seen.add(h.get_label())
            uniq.append(h)
    allh = uniq + extra
    fig.legend(handles=allh, loc="lower center", ncol=min(len(allh), 4), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    rows = int(np.ceil(len(allh) / 4))
    fig.tight_layout(rect=(0, min((0.28 * rows + 0.25) / fig.get_figheight(), 0.35), 1, 1))


def fig_axis_series(scores: pd.DataFrame, series: pd.DataFrame, spec, out_dir, formats, method: str = "ordinal",
                    seeds: pd.DataFrame | None = None, x_axis: str = "auto") -> None:
    """One panel per study of ``series`` (columns sample, series, x and optionally x_label): the score at each position
    of the series (positions as categories, the series side by side, one colour each; marker shape = the sample's
    category of the axis). ``seeds`` (samples x seeds, `axes --seeds`) adds each seed's score as small dots."""
    import matplotlib.pyplot as plt
    s = series.join(scores[["gse", "category", method]], on="sample", how="inner").dropna(subset=[method])
    by_panel = "panel" in s.columns
    s["_panel"] = s["panel"] if by_panel else s["gse"]
    studies = list(dict.fromkeys(s["_panel"]))
    if not studies:
        return
    ncol = min(3, len(studies))
    nrow = int(np.ceil(len(studies) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(6.2, 0.55 * s.groupby("_panel")["x"].nunique().max() + 3) * ncol, 4.6 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[len(studies):]:
        ax.axis("off")
    handles = []
    for ax, g in zip(axes.ravel(), studies):
        sub = s[s["_panel"] == g]
        handles += _draw_series(ax, sub, method, seeds, spec.categories, x_axis)
        ax.axhline(0, color=GREY, lw=0.8, ls=":")
        ax.axhline(1, color=GREY, lw=0.8, ls=":")
        ax.set_xlabel(sub["x_label"].iloc[0] if "x_label" in sub.columns else "position", fontsize=11)
        ax.set_ylabel(f"{pretty(spec.name)} ({METHOD_LABEL[method]})", fontsize=11)
        ax.set_title(f"{g} (n={len(sub)})", fontsize=12)
        ax.tick_params(labelsize=10)
    extra = [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], ls="", color=GREY, markeredgecolor="black", label=c.replace("_", " "))
             for i, c in enumerate(spec.categories) if (s["category"] == c).any()]
    if seeds is not None:
        extra.append(plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=3, label=f"each of {seeds.shape[1]} seeds"))
    _series_legend(fig, handles if (len(studies) == 1 or by_panel) else [], extra)
    savefig(fig, out_dir, f"fig_axis_{spec.name}_series", formats)


def fig_axes_map(scores_x: pd.DataFrame, spec_x, scores_y: pd.DataFrame, spec_y, out_dir, formats, method: str = "ordinal",
                 query=None, series: pd.DataFrame | None = None) -> None:
    """Two axes against each other, twice: coloured by the categories of the first and of the second axis. ``query``
    samples (`axes --query`) are drawn on top, joined along their ``series`` in the order of x."""
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
        if query:
            qs = both.loc[both.index.intersection(list(query))]
            if series is not None:
                path = series.set_index("sample").join(qs[["x", "y"]], how="inner", rsuffix="_score")
                for _, line in path.groupby("series"):
                    line = line.groupby("x")[["x_score", "y"]].mean() if "x_score" in line.columns else None
                    if line is not None:
                        ax.plot(line["x_score"], line["y"], color="black", lw=1.0, alpha=0.7, zorder=4)
            ax.scatter(qs["x"], qs["y"], s=36, color="white", edgecolors="black", linewidths=1.0, zorder=5,
                       label=f"query ({len(qs)})")
        ax.legend(fontsize=9, markerscale=2, frameon=False)
    fig.tight_layout()
    savefig(fig, out_dir, f"fig_axes_{spec_x.name}_by_{spec_y.name}", formats)


HEAD_PROB = {"phase": "p_stationary", "medium": "p_minimal", "lab_strain": "p_lab"}


def fig_heads_series(heads: pd.DataFrame, series: pd.DataFrame, out_dir, formats, seeds: dict | None = None,
                     x_axis: str = "auto") -> None:
    """The calls of the model's heads (probability of the second class, 0-1, as in `embed`'s predictions.tsv) along
    the series: one panel per head, the positions as categories, the series side by side, one colour each. ``seeds``
    ({seed name: predictions frame}, `embed --seeds`) adds each seed's probability as small dots."""
    import matplotlib.pyplot as plt
    from ..config import HEADS
    cols = [h for h in HEADS if HEAD_PROB[h] in heads.columns]
    s = series.join(heads[[HEAD_PROB[h] for h in cols]], on="sample", how="inner")
    if s.empty or not cols:
        return
    panels = list(dict.fromkeys(s["panel"])) if "panel" in s.columns else [None]
    nx = s["x"].nunique()
    if panels == [None]:                         # one column of heads
        fig, axes = plt.subplots(len(cols), 1, figsize=(max(7, 0.8 * nx + 3), 3.4 * len(cols)), squeeze=False)
        cell = {(h, None): axes[r, 0] for r, h in enumerate(cols)}
    else:                                        # one row per panel, one column per head
        fig, axes = plt.subplots(len(panels), len(cols), figsize=(max(6, 0.6 * nx + 2.5) * len(cols), 3.2 * len(panels)), squeeze=False)
        cell = {(h, pan): axes[r, c] for r, pan in enumerate(panels) for c, h in enumerate(cols)}
    handles = []
    for (h, pan), ax in cell.items():
        p = HEAD_PROB[h]
        per_seed = pd.DataFrame({n: f[p] for n, f in seeds.items() if p in f.columns}) if seeds else None
        sub = s if pan is None else s[s["panel"] == pan]
        handles = _draw_series(ax, sub.dropna(subset=[p]), p, per_seed, x_axis=x_axis) or handles
        ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel(s["x_label"].iloc[0] if "x_label" in s.columns else "position", fontsize=11)
        ax.set_ylabel(f"P({HEADS[h][1].replace('_', ' ')})", fontsize=11)
        ax.set_title((f"{pan}: " if pan is not None else "")
                     + f"{pretty(h)} call: {HEADS[h][0].replace('_', ' ')} (0) to {HEADS[h][1].replace('_', ' ')} (1)", fontsize=11)
        ax.tick_params(labelsize=10)
    extra = [plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=3, label=f"each of {len(seeds)} seeds")] if seeds else []
    _series_legend(fig, handles, extra)
    savefig(fig, out_dir, "fig_heads_series", formats)


def fig_heads_seed_votes(heads: pd.DataFrame, series: pd.DataFrame, seeds: dict, out_dir, formats) -> None:
    """How many of the seeds make each call, along the series: one panel per head, one bar per sample (height 1),
    split into the share of seeds that call the second class (probability at least 0.5) and the share that call the
    first. Positions as categories, the series side by side within each and named above the bars (without the words
    all series names share). With a ``panel`` column one row of panels per value. ``seeds``: {seed name: predictions
    frame} of `embed --seeds`."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ..config import HEADS
    cols = [h for h in HEADS if any(HEAD_PROB[h] in f.columns for f in seeds.values())]
    s = series[series["sample"].isin(heads.index)]
    if s.empty or not cols:
        return
    panels = list(dict.fromkeys(s["panel"])) if "panel" in s.columns else [None]
    nx = s["x"].nunique()
    if panels == [None]:
        fig, axes = plt.subplots(len(cols), 1, figsize=(max(7, 0.8 * nx + 3), 3.0 * len(cols)), squeeze=False)
        cell = {(h, None): (axes[r, 0], True) for r, h in enumerate(cols)}
    else:
        fig, axes = plt.subplots(len(panels), len(cols), figsize=(max(6, 0.6 * nx + 2.5) * len(cols), 3.2 * len(panels)), squeeze=False)
        cell = {(h, pan): (axes[r, c], r == 0) for r, pan in enumerate(panels) for c, h in enumerate(cols)}
    names = list(dict.fromkeys(s.sort_values(["series"])["series"]))
    label = _short_names(names)
    for (h, pan), (ax, show_legend) in cell.items():
        sp = s if pan is None else s[s["panel"] == pan]
        xs = sorted(sp["x"].unique())
        pos = {x: i for i, x in enumerate(xs)}
        here = [n for n in names if (sp["series"] == n).any()]
        slot = 0.8 / max(len(here), 1)
        offset = {n: (i - (len(here) - 1) / 2) * slot for i, n in enumerate(here)}
        p = HEAD_PROB[h]
        P = pd.DataFrame({n: f[p] for n, f in seeds.items() if p in f.columns})
        high = (P >= 0.5).sum(axis=1) / P.notna().sum(axis=1)
        for n in here:
            line = sp[sp["series"] == n]
            x = line["x"].map(pos).to_numpy() + offset[n]
            v = high.reindex(line["sample"]).to_numpy(dtype=float)
            ok = ~np.isnan(v)
            ax.bar(x[ok], v[ok], width=slot * 0.8, color=RED, edgecolor="black", linewidth=0.6)
            ax.bar(x[ok], 1 - v[ok], bottom=v[ok], width=slot * 0.8, color=BLUE, edgecolor="black", linewidth=0.6)
            for x0 in x[ok]:
                ax.text(x0, 1.02, label[n], ha="center", va="bottom", fontsize=6, rotation=90 if len(label[n]) > 2 else 0,
                        clip_on=False)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([_x_label(x) for x in xs])
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_xlabel(s["x_label"].iloc[0] if "x_label" in s.columns else "position", fontsize=11)
        ax.set_ylabel(f"share of {P.shape[1]} seeds", fontsize=11)
        tall = max(len(v) for v in label.values()) > 2
        ax.set_title((f"{pan}: " if pan is not None else "") + f"{pretty(h)} call", fontsize=11, pad=30 if tall else 14)
        ax.tick_params(labelsize=10)
        if show_legend:
            ax.legend(handles=[Patch(facecolor=RED, label=HEADS[h][1].replace("_", " ")),
                               Patch(facecolor=BLUE, label=HEADS[h][0].replace("_", " "))],
                      loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8, frameon=False)
    prefix = names[0][: len(names[0]) - len(label[names[0]])].strip()
    if prefix:
        fig.suptitle(f"bars labelled by {prefix}", fontsize=10, y=1.0)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_heads_seed_votes", formats)


def fig_heads_map(atlas: pd.DataFrame | None, heads: pd.DataFrame, out_dir, formats, series: pd.DataFrame | None = None,
                  x: str = "phase", y: str = "medium") -> None:
    """Two head calls (probabilities) against each other: the atlas samples of the bundle coloured by their labels of
    the two calls, the new samples on top, joined along their ``series``. The probabilities pile up at 0 and 1, so the
    atlas is drawn with a little jitter."""
    import matplotlib.pyplot as plt
    from ..config import HEADS
    px, py = HEAD_PROB[x], HEAD_PROB[y]
    if px not in heads.columns or py not in heads.columns:
        return
    q = heads[[px, py]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rng = np.random.RandomState(0)
    for ax, head in zip(axes, (x, y)):
        if atlas is not None and {px, py} <= set(atlas.columns):
            a = atlas.dropna(subset=[px, py])
            jx, jy = rng.normal(0, 0.008, len(a)), rng.normal(0, 0.008, len(a))
            lab = a.get(f"{head}_label", pd.Series("", index=a.index)).fillna("").astype(str)
            colors = {HEADS[head][0]: "#2c7fb8", HEADS[head][1]: "#de2d26"}
            none = (~lab.isin(list(colors))).to_numpy()
            ax.scatter(a[px][none] + jx[none], a[py][none] + jy[none], s=4, color=GREY, alpha=0.3, linewidths=0,
                       label=f"no {head.replace('_', ' ')} label ({int(none.sum())})")
            for c, col in colors.items():
                m = (lab == c).to_numpy()
                ax.scatter(a[px][m] + jx[m], a[py][m] + jy[m], s=8, color=col, alpha=0.5, linewidths=0, label=f"{c.replace('_', ' ')} ({int(m.sum())})")
        if series is not None:
            path = series.set_index("sample").join(q, how="inner")
            for _, line in path.groupby("series"):
                line = line.groupby("x")[[px, py]].mean()
                ax.plot(line[px], line[py], color="black", lw=1.0, alpha=0.7, zorder=4)
        ax.scatter(q[px], q[py], s=36, color="white", edgecolors="black", linewidths=1.0, zorder=5, label=f"query ({len(q)})")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"{pretty(x)} call: P({HEADS[x][1]})")
        ax.set_ylabel(f"{pretty(y)} call: P({HEADS[y][1]})")
        ax.set_title(f"atlas coloured by its {pretty(head)} labels", fontsize=13)
        ax.legend(fontsize=9, markerscale=2, frameon=False)
    fig.tight_layout()
    savefig(fig, out_dir, f"fig_heads_{x}_by_{y}", formats)
