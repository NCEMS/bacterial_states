"""Panel: MG1655 only, CGSC 7740-type against ATCC 47076-type, the difference of the study means of every set's residual
with a 95% bootstrap CI over studies (figures/fig_mg1655_stock.pdf/.png), and the tests (figures/stats_mg1655_stock.tsv;
figlib.two_level; q = Benjamini-Hochberg over the 9 sets). Non-mixed MG1655-like samples with a resolved stock.
    python scripts/fig_stock.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import figlib as F

A, B = "CGSC7740-type", "ATCC47076-type"


def main():
    plt = F.style()
    d = F.samples()
    x = d[~d.mixed & (d.lineage == "MG1655-like")]
    t = pd.DataFrame([{"set": s, **F.two_level(x, f"res_{s}", "mg1655_stock", A, B)} for s in F.SETS])
    for c in ("p_sample", "p_study", "p_perm", "p_mixed"):
        t["q" + c[1:]] = F.bh(t[c])
    t.to_csv(f"{F.FIGURES}/stats_mg1655_stock.tsv", sep="\t", index=False, float_format="%.4g")
    print(t[["set", "n_a", "n_b", "studies_a", "studies_b", "diff_study", "ci_lo", "ci_hi", "p_study", "q_study",
             "p_perm", "p_mixed"]].to_string(index=False))
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.axvline(0, color="#9c9c9c", lw=1, ls="--")
    y = np.arange(len(t))
    ax.hlines(y, t.ci_lo, t.ci_hi, color="#4d4d4d", lw=1.4)
    ax.scatter(t.diff_study, y, s=40, c=F.GROUPS["MG1655, CGSC 7740-type"], edgecolors="#4d4d4d", linewidths=0.6,
               zorder=3)
    ax.set_yticks(y, t.set)
    for yi, r in zip(y, t.itertuples()):
        mark = "*" if r.q_study < 0.05 else ""
        ax.text(1.02, yi, f"q = {r.q_study:.2g}{mark}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=7.5, color="#262626")
    ax.set_xlabel("CGSC 7740-type minus ATCC 47076-type\n(difference of study means of the residual, log2)")
    r0 = t.iloc[0]
    ax.set_title(f"MG1655: {int(r0.studies_a)} CGSC 7740-type vs {int(r0.studies_b)} ATCC 47076-type studies "
                 f"({int(r0.n_a)} vs {int(r0.n_b)} samples)\nbars: 95% bootstrap CI over studies; q: study-level "
                 "Mann-Whitney, BH", fontsize=7.5, color="#4d4d4d", loc="left")
    ax.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
    F.savefig(fig, "fig_mg1655_stock")


if __name__ == "__main__":
    main()
