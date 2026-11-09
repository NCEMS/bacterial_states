"""Study leakage on TEST: same-study fraction among k nearest neighbours for the z-scored input,
the concatenated factor blocks, the final embedding and (optionally) reComBat (notebook cell 44)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import same_study_rate
from ..utils import boot_ci, paired_boot_ci
from .plotting import savefig

K = 25
ORDER = ["input_zcpm", "concat", "top_ae", "reComBat"]
PLOT_ONLY = ["input_zcpm", "top_ae", "reComBat"]
TRANSDUCTIVE = {"input_zcpm": False, "concat": False, "top_ae": False, "reComBat": True}
STYLE = {"input_zcpm": "0.6", "concat": "#1f77b4", "top_ae": "#2ca02c", "reComBat": "crimson"}


def leakage_stats(ctx, recombat: pd.DataFrame | None = None, k: int = K) -> dict:
    meta = ctx.meta
    pool = np.where((meta["split"] == 3).values)[0]
    if recombat is not None:
        rc_pos = set(np.where(meta.index.isin(recombat.index))[0])
        pool = np.array([p for p in pool if p in rc_pos], dtype=int)
    if len(pool) < 3:
        raise ValueError("fewer than 3 TEST samples; the leakage figure needs a TEST set")
    kk = int(min(k, len(pool) - 1))
    gse, cid, ann = ctx.data.gse_codes[pool], ctx.data.cond_id[pool], ctx.data.is_annot[pool]
    sizes = pd.Series(gse).value_counts()
    chance = float(np.mean([(sizes[g] - 1) / max(len(pool) - 1, 1) for g in gse]))
    cands = {"input_zcpm": ctx.Z[pool], "concat": ctx.concat_blocks[pool], "top_ae": ctx.embedding[pool]}
    if recombat is not None:
        cands["reComBat"] = recombat.loc[meta.index[pool]].values
    data = {n: same_study_rate(E, gse, cid, ann, kk) for n, E in cands.items()}
    order = [n for n in ORDER if n in data]
    stats = {}
    for n in order:
        lo, hi = boot_ci(data[n], n_boot=2000, seed=0)
        stats[n] = {"mean": float(np.mean(data[n])), "ci": [lo, hi], "n": int(len(data[n])),
                    "transductive": TRANSDUCTIVE[n]}
    paired = {}
    for a, b in [("top_ae", "concat"), ("top_ae", "input_zcpm"), ("concat", "input_zcpm")]:
        if a in data and b in data:
            d, lo, hi = paired_boot_ci(data[a], data[b])
            paired[f"{a}-{b}"] = {"diff": d, "ci": [lo, hi], "includes_zero": bool(lo <= 0 <= hi)}
    return {"n_test": int(len(pool)), "n_studies": int(len(sizes)), "k": kk, "chance": chance,
            "rates": stats, "paired": paired, "_data": data}


def fig_study_leakage(ctx, out_dir, formats, recombat: pd.DataFrame | None = None) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    st = leakage_stats(ctx, recombat)
    stats, chance = st["rates"], st["chance"]
    plot = [n for n in PLOT_ONLY if n in stats]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(plot))
    means = [stats[n]["mean"] for n in plot]
    lo = [stats[n]["mean"] - stats[n]["ci"][0] for n in plot]
    hi = [stats[n]["ci"][1] - stats[n]["mean"] for n in plot]
    bars = ax.bar(x, means, color=[STYLE[n] for n in plot], alpha=0.85, width=0.65, edgecolor="k", linewidth=1.0,
                  yerr=[lo, hi], capsize=5, ecolor="k")
    for b, n in zip(bars, plot):
        if TRANSDUCTIVE[n]:
            b.set_hatch("//")
    for i, n in enumerate(plot):
        ax.text(i, stats[n]["ci"][1] + 0.006, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(chance, color="k", ls=":", lw=1.4)
    ax.text(len(plot) - 0.5, chance, f" chance = {chance:.3f}", va="bottom", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot, rotation=12)
    ax.set_ylabel(f"same-study fraction of k={st['k']} neighbours")
    ax.set_ylim(0, max(max(m + h for m, h in zip(means, hi)), chance) + 0.06)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(handles=[Patch(facecolor="white", edgecolor="k", hatch="//", label="transductive")], loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_study_leakage", formats)
    st.pop("_data")
    return st
