"""Panel: one regulon set against the ribosomal proteins, one point per sample (figures/fig_regulon_<set>.pdf/.png).
Fill = genotype group, ring = IS1H state (figlib); dashed line = the line through the study means that the residuals
are taken from (fig_batch_tracking, without the excluded study).
    python scripts/fig_regulon.py --set FlhDC"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import figlib as F

NAMES = {"FlhDC": "FlhDC regulon", "FliA": "FliA (σ28) regulon"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="FlhDC", choices=F.SETS)
    a = p.parse_args()
    plt = F.style()
    d = F.samples()
    k, b = F.trend(a.set)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    xs = np.linspace(d.ribosome.min() - 0.05, d.ribosome.max() + 0.05, 10)
    ax.plot(xs, k * xs + b, color="#9c9c9c", lw=1.2, ls="--", zorder=1)
    F.draw_points(ax, d, "ribosome", a.set, size=48)
    ax.set_xlabel("ribosomal proteins, mean log2 expression")
    ax.set_ylabel(f"{NAMES.get(a.set, a.set + ' regulon')}, mean log2 expression")
    ax.grid(color="#ececec", lw=0.6, zorder=0)
    ax.legend(handles=F.legend_handles(plt, set(d.group)), loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=7.5)
    n = (~d.mixed).sum()
    ax.set_title(f"{len(d)} LB samples ({d.gse.nunique()} studies); dashed: line through the study means",
                 fontsize=8, color="#4d4d4d", loc="left")
    F.savefig(fig, f"fig_regulon_{a.set}")


if __name__ == "__main__":
    main()
