"""Panel: a set's residual by IS1H state, one point per sample (figures/fig_is1h_<set>.pdf/.png), and the tests of
IS1H present against absent for every set (figures/stats_is1h.tsv; figlib.two_level), in two scopes:
  K-12      all non-mixed samples of the K-12 lab strains (MG1655, BW25113 lineage, other K-12)
  MG1655    MG1655-like samples only
q = Benjamini-Hochberg over the 9 sets within a scope and test.
    python scripts/fig_is1h.py --set FlhDC --set FliA"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import figlib as F

ORDER = ["present", "absent", "no_info"]
SHORT = {"present": "present", "absent": "absent", "no_info": "no reads"}


def tests(d):
    k12 = d[~d.mixed & (d.group != "E. coli B")]
    rows = []
    for scope, x in (("K-12", k12), ("MG1655", k12[k12.lineage == "MG1655-like"])):
        for s in F.SETS:
            rows.append({"scope": scope, "set": s, **F.two_level(x, f"res_{s}", "IS1H_state", "present", "absent")})
    t = pd.DataFrame(rows)
    for c in ("p_sample", "p_study", "p_perm", "p_mixed"):
        t["q" + c[1:]] = t.groupby("scope")[c].transform(F.bh)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", default=None)
    a = p.parse_args()
    sets = a.set or ["FlhDC", "FliA"]
    plt = F.style()
    d = F.samples()
    t = tests(d)
    t.to_csv(f"{F.FIGURES}/stats_is1h.tsv", sep="\t", index=False, float_format="%.4g")
    print(t[t.set.isin(sets)][["scope", "set", "n_a", "n_b", "studies_a", "studies_b", "diff_study", "p_sample",
                                "p_study", "p_perm", "p_mixed", "r2"]].to_string(index=False))
    rng = np.random.default_rng(1)
    for s in sets:
        x = d[d.group != "E. coli B"].copy()
        x["xpos"] = x.IS1H_state.map({k: i for i, k in enumerate(ORDER)}) + rng.uniform(-0.22, 0.22, len(x))
        fig, ax = plt.subplots(figsize=(3.6, 3.8))
        ax.axhline(0, color="#9c9c9c", lw=1, ls="--", zorder=1)
        F.draw_points(ax, x, "xpos", f"res_{s}", size=40)
        for i, k in enumerate(ORDER):
            v = x[(x.IS1H_state == k) & ~x.mixed][f"res_{s}"]
            ax.plot([i - 0.3, i + 0.3], [v.median()] * 2, color="#262626", lw=1.6, zorder=5)
        ax.set_xticks(range(3), [f"{SHORT[k]}\n(n = {((x.IS1H_state == k) & ~x.mixed).sum()})" for k in ORDER],
                      fontsize=7.5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel(f"{s} regulon, residual from the trend (log2)")
        ax.set_xlabel("IS1H upstream of flhDC")
        r = t[(t.scope == "K-12") & (t.set == s)].iloc[0]
        ax.set_title(f"present vs absent ({r.studies_a} vs {r.studies_b} studies): study-level p = {r.p_study:.2g},\n"
                     f"permutation between studies p = {r.p_perm:.2g}; R² = {r.r2:.2f}",
                     fontsize=7.5, color="#4d4d4d", loc="left")
        ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
        ax.legend(handles=F.legend_handles(plt, set(x.group)), loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=False, fontsize=7.5)
        F.savefig(fig, f"fig_is1h_{s}")


if __name__ == "__main__":
    main()
