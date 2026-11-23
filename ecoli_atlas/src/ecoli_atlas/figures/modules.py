"""Figure of `drivers --modules`: the curated modules that move a score most, per source (regulons, iModulons,
pathways, ...)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import GREY, savefig

UP, DOWN = "#b2182b", "#2166ac"


def fig_modules(table: pd.DataFrame, title: str, high: str, low: str, out_dir, name: str, formats, top: int = 10,
                namespaces=None) -> None:
    """One panel per source (``namespaces`` restricts them): the ``top`` modules by |z_net|, bar = z_net, red when the module moves the score
    towards ``high``, blue towards ``low``; drawn pale when it is among the top modules in fewer than 3 of 4 study draws."""
    import matplotlib.pyplot as plt
    nss = [ns for ns in (namespaces or sorted(table["source"].unique())) if (table["source"] == ns).any()]
    if not nss:
        return
    ncol = min(3, len(nss))
    nrow = int(np.ceil(len(nss) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 0.32 * top * nrow + 1.4 * nrow + 0.6), squeeze=False)
    for ax, ns in zip(axes.flat, nss):
        sub = table[table["source"] == ns].head(top)
        y = np.arange(len(sub))[::-1]
        bars = ax.barh(y, sub["z_net"], color=[UP if v > 0 else DOWN for v in sub["net"]], height=0.75)
        for bar, b in zip(bars, sub["boot_in_top"]):
            bar.set_alpha(1.0 if b >= 0.75 else 0.35)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{n} ({s})" for n, s in zip(sub["name"], sub["size"])], fontsize=9)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_title(ns, fontsize=11)
        ax.set_xlabel("z of the module's attribution")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(nss):]:
        ax.axis("off")
    fig.suptitle(f"{title}  (red: towards {high}, blue: towards {low}; pale: not stable over study draws)", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, name, formats)
