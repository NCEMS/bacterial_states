"""Shared parts of the figure scripts (scripts/fig_*.py): the sample table, genotype groups and colours, the IS1H
border, and the tests. Reads sample_states.tsv (the per-sample states; tracked, or rewritten by scripts/collect.py)
and results/residuals.tsv (scripts/residuals.py, from the atlas).

Samples: the LB samples of sample_states.tsv without the excluded studies (common.EXCLUDE_STUDIES) and the controls. Mixed libraries
(>= 20 clustered calls) are drawn hollow and left out of every test.

Tests of a two-level factor (e.g. IS1H present / absent) on a residual, with samples nested in studies:
  p_sample   Mann-Whitney over samples
  p_study    Mann-Whitney over study means (one value per study and level)
  p_perm     permutation of the levels between whole studies (each study keeps its samples; statistic: difference of
             the sample means; 20,000 permutations); studies with both levels are left out of it
  coef, p_mixed   linear mixed model residual ~ level + (1 | study) (statsmodels MixedLM, ML fit)
  r2         share of the residual variance explained by the level (samples)
"""
import os
import numpy as np, pandas as pd
from scipy import stats

from common import FIGURES, RESULTS, STATES, SETS, EXCLUDE_STUDIES
import residuals as _residuals

# genotype groups in a fixed order; the three hues are slots 1-3 of the validated reference palette (light mode:
# CVD worst adjacent dE 12.6, normal-vision 28.1); contrast < 3:1 for two of them, so every mark has a dark edge or a
# label and the legend names every group
GROUPS = {
    "MG1655, ATCC 47076-type": "#257de0",
    "MG1655, CGSC 7740-type": "#f47230",
    "BW25113 lineage": "#00b894",
    "MG1655, stock unresolved": "#9dc8f7",
    "other K-12": "#9c9c9c",
    "E. coli B": "#d9d9d9",
}
# IS1H border: present = thick black ring, absent = thin grey ring, no information = no ring
IS1H_EDGE = {"present": ("black", 1.5), "absent": ("#6f6f6f", 0.6), "no_info": ("none", 0.0)}
IS1H_LABEL = {"present": "IS1H present", "absent": "IS1H absent", "no_info": "no reads across IS1H"}


def group(r):
    if r.lineage == "BW25113-like":
        return "BW25113 lineage"
    if r.lineage == "MG1655-like":
        return {"ATCC47076-type": "MG1655, ATCC 47076-type", "CGSC7740-type": "MG1655, CGSC 7740-type"}.get(
            r.mg1655_stock, "MG1655, stock unresolved")
    if r.lineage == "B DSM 613":
        return "E. coli B"
    return "other K-12"


def samples():
    t = pd.read_csv(STATES, sep="\t")
    t = t[~t.gse.isin(EXCLUDE_STUDIES) & (t.strain != "control")].copy()
    t["group"] = t.apply(group, axis=1)
    res = pd.read_csv(RESULTS / "residuals.tsv", sep="\t")
    res = res[["gsm", "ribosome"] + SETS + [f"res_{k}" for k in SETS]]
    return t.merge(res, on="gsm", how="inner")


def trend(set_name):
    """slope, intercept of the line the residuals are taken from (scripts/residuals.py)"""
    return _residuals.trend(set_name)


def two_level(d, col, level_col, a, b, n_perm=20000, seed=0):
    """Tests of level a against b (see the module docstring) on column col of d (samples)."""
    d = d[d[level_col].isin([a, b])].dropna(subset=[col])
    xa, xb = d[d[level_col] == a][col], d[d[level_col] == b][col]
    out = {"n_a": len(xa), "n_b": len(xb), "studies_a": d[d[level_col] == a].gse.nunique(),
           "studies_b": d[d[level_col] == b].gse.nunique(), "median_a": xa.median(), "median_b": xb.median(),
           "diff_mean": xa.mean() - xb.mean()}
    out["p_sample"] = stats.mannwhitneyu(xa, xb).pvalue if len(xa) and len(xb) else np.nan
    st = d.groupby(["gse", level_col])[col].mean().reset_index()
    sa, sb = st[st[level_col] == a][col], st[st[level_col] == b][col]
    out["diff_study"] = sa.mean() - sb.mean()
    out["p_study"] = stats.mannwhitneyu(sa, sb).pvalue if len(sa) > 1 and len(sb) > 1 else np.nan
    # bootstrap over studies for a CI of the difference of study means
    rng = np.random.default_rng(seed)
    bs = [rng.choice(sa.values, len(sa)).mean() - rng.choice(sb.values, len(sb)).mean() for _ in range(5000)]
    out["ci_lo"], out["ci_hi"] = np.percentile(bs, [2.5, 97.5])
    # permutation between whole studies
    lv = d.groupby("gse")[level_col].agg(lambda x: x.iloc[0] if x.nunique() == 1 else None).dropna()
    dd = d[d.gse.isin(lv.index)]
    groups = {g: v[col].values for g, v in dd.groupby("gse")}
    gs = list(lv.index); labels = lv.values
    def stat(lab):
        A = np.concatenate([groups[g] for g, l in zip(gs, lab) if l == a] or [np.array([])])
        B = np.concatenate([groups[g] for g, l in zip(gs, lab) if l == b] or [np.array([])])
        return A.mean() - B.mean() if len(A) and len(B) else np.nan
    obs = stat(labels)
    null = np.array([stat(rng.permutation(labels)) for _ in range(n_perm)])
    out["p_perm"] = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm) if np.isfinite(obs) else np.nan
    try:                                         # optional: statsmodels is not in the ecoli_atlas environment
        import statsmodels.formula.api as smf
        m = smf.mixedlm(f"{col} ~ C({level_col}, Treatment('{b}'))", d, groups=d["gse"]).fit(reml=False)
        name = [k for k in m.params.index if k.startswith("C(")][0]
        out["coef"], out["p_mixed"] = m.params[name], m.pvalues[name]
    except Exception:                            # no statsmodels, or a singular fit
        out["coef"], out["p_mixed"] = np.nan, np.nan
    tot = ((d[col] - d[col].mean()) ** 2).sum()
    within = d.groupby(level_col)[col].transform(lambda v: v - v.mean()).pow(2).sum()
    out["r2"] = 1 - within / tot if tot else np.nan
    return out


def bh(p):
    p = np.asarray(p, dtype=float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok])
    return q


def savefig(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", f"{FIGURES}/{name}.pdf")


def style():
    import matplotlib
    matplotlib.use("Agg")
    FIGURES.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "hatch.color": "#c8c8c8", "hatch.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#6f6f6f", "axes.labelcolor": "#262626", "xtick.color": "#4d4d4d",
                         "ytick.color": "#4d4d4d", "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def legend_handles(plt, groups_present, with_is1h=True, with_mixed=True):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=GROUPS[g], markeredgecolor="#6f6f6f",
                markeredgewidth=0.6, label=g) for g in GROUPS if g in groups_present]
    if with_is1h:
        for s in ("present", "absent", "no_info"):
            c, w = IS1H_EDGE[s]
            h.append(Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor="#ffffff",
                            markeredgecolor=c if c != "none" else "#ffffff", markeredgewidth=max(w, 0.01),
                            label=IS1H_LABEL[s]))
    if with_mixed:
        h.append(Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor="none", markeredgecolor="#4d4d4d",
                        markeredgewidth=0.8, label="mixed library (not tested)"))
    return h


def draw_points(ax, d, x, y, size=36, jitter=None, rng=None):
    """samples of d at (x, y): fill by group, ring by IS1H state; mixed libraries hollow and smaller"""
    for mixed in (False, True):
        for s in ("no_info", "absent", "present"):          # present drawn last, on top
            for g in GROUPS:
                m = (d.group == g) & (d.IS1H_state == s) & (d.mixed == mixed)
                if not m.any():
                    continue
                xs = d.loc[m, x].values if isinstance(x, str) else x[m.values]
                ys = d.loc[m, y].values
                if mixed:
                    ax.scatter(xs, ys, s=size * 0.5, facecolors="none", edgecolors=GROUPS[g] if g != "E. coli B"
                               else "#9c9c9c", linewidths=0.9, zorder=2)
                    continue
                c, w = IS1H_EDGE[s]
                ax.scatter(xs, ys, s=size, c=GROUPS[g], edgecolors=c, linewidths=w, zorder=3 + (s == "present"))
