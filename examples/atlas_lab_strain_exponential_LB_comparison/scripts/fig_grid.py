"""Panel: genotype grid, one row per study (figures/fig_genotype_grid.pdf/.png, figures/genotype_grid.tsv).
Columns: the genotype group (colour strip), the derived / ancestral state at IS1H (flhDC upstream), IS1I (crl), IS5U
(oppA-ychE), glpR and gatC, and the study mean FlhDC residual (bar). Derived states: IS1H, IS1I, IS5U present; the
glpR and gatC frameshifts of CGSC 7740 (the reference alleles of NC_000913.3). A cell gives the state of the study's
informative samples (mixed libraries included; `*` marks studies with mixed libraries): derived, ancestral, both, or
no reads. glpR and gatC are MG1655 stock markers: greyed out (not applicable) for the other groups. Rows sorted by group, then by the FlhDC residual.
    python scripts/fig_grid.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import figlib as F

LOCI = [("IS1H\n(flhDC)", "IS1H_state", {"present": 1, "absent": 0}),
        ("IS1I\n(crl)", "IS1I_state", {"present": 1, "absent": 0}),
        ("IS5U\n(oppA)", "IS5U_state", {"present": 1, "absent": 0}),
        ("glpR", "glpR_3560455_insG", {0.0: 1, 1.0: 0}),
        ("gatC", "gatC_2173363_del2", {0.0: 1, 1.0: 0})]
CELL = {"derived": "#262626", "ancestral": "#e6e6e6", "both": "#9c9c9c", "none": "#ffffff", "na": "#f5f5f5"}
MG1655_ONLY = ("glpR", "gatC")


def cell(v):
    v = [x for x in v if x in (0, 1)]
    return "none" if not v else "derived" if min(v) == 1 else "ancestral" if max(v) == 0 else "both"


def main():
    plt = F.style()
    from matplotlib.patches import Rectangle, Patch
    d = F.samples()
    rows = []
    for gse, x in d.groupby("gse"):
        g = x.group.value_counts().index[0]
        r = {"gse": gse, "group": g, "n": len(x), "mixed": bool(x.mixed.any()),
             "FlhDC": x[~x.mixed].res_FlhDC.mean() if (~x.mixed).any() else x.res_FlhDC.mean()}
        for name, col, m in LOCI:
            r[name] = cell(x[col].map(m).tolist()) if col in x else "none"
            if name in MG1655_ONLY and not g.startswith("MG1655"):
                r[name] = "na"
        rows.append(r)
    t = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(F.GROUPS)}
    t = t.sort_values(["group", "FlhDC"], key=lambda c: c.map(order) if c.name == "group" else -c).reset_index(drop=True)
    t.rename(columns={n: n.replace("\n", " ") for n, _, _ in LOCI}).to_csv(f"{F.FIGURES}/genotype_grid.tsv", sep="\t",
                                                                            index=False, float_format="%.3f")
    n = len(t)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.6, 0.19 * n + 1.4), sharey=True,
                                 gridspec_kw={"width_ratios": [len(LOCI) + 1, 2.4], "wspace": 0.05})
    for i, r in t.iterrows():
        y = n - 1 - i
        ax.add_patch(Rectangle((0, y), 0.6, 0.9, color=F.GROUPS[r.group], lw=0))
        for j, (name, _, _) in enumerate(LOCI):
            na = r[name] == "na"
            ax.add_patch(Rectangle((1 + j, y), 0.9, 0.9, facecolor=CELL[r[name]], edgecolor="#d0d0d0", lw=0.5,
                                   hatch="////" if na else None))
        bx.barh(y + 0.45, r.FlhDC, height=0.7, color=F.GROUPS[r.group], edgecolor="#6f6f6f", lw=0.4)
    ax.set_xlim(0, len(LOCI) + 1); ax.set_ylim(0, n)
    ax.set_xticks([0.3] + [1.45 + j for j in range(len(LOCI))], ["group"] + [l[0] for l in LOCI], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n) + 0.45, [f"{g}{' *' if m else ''}" for g, m in zip(t.gse[::-1], t.mixed[::-1])],
                  fontsize=6.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    bx.axvline(0, color="#9c9c9c", lw=0.8)
    bx.set_xlabel("FlhDC residual,\nstudy mean (log2)", fontsize=7.5)
    bx.grid(axis="x", color="#ececec", lw=0.6)
    bx.spines["left"].set_visible(False); bx.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=CELL[k], edgecolor="#d0d0d0", label=l) for k, l in
               (("derived", "derived (as in NC_000913.3)"), ("ancestral", "ancestral"), ("both", "both"),
                ("none", "no reads"))]
    handles.append(Patch(facecolor=CELL["na"], edgecolor="#d0d0d0", hatch="////", label="not applicable (not MG1655)"))
    handles += [Patch(facecolor=c, label=g) for g, c in F.GROUPS.items() if g in set(t.group)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.92, 0.9), frameon=False, fontsize=7)
    F.savefig(fig, "fig_genotype_grid")


if __name__ == "__main__":
    main()
