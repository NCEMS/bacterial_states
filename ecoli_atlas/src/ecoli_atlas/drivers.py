"""`ecoli_atlas drivers`: which genes move a head's call.

For a head (default: phase) and a pair of its classes, low -> high, every gene gets an attribution
by integrated gradients: the head's score (the log-odds of its call; for the ordinal phase head the
phase score minus its upper threshold) is followed along the straight line from a baseline, the
mean expression profile of the low class, to each sample of the high class, and the change of the
score is shared out among the genes. The attributions of one sample add up to its score minus the
score of the baseline. They are averaged over the high-class samples with every study counting
once.

The sign of an attribution says what the gene's change does to the score, not which way the gene
changes: a ribosomal protein gene that falls in stationary phase and thereby raises the phase score
has a positive attribution. The direction of the gene is in ``direction`` (`up` / `down` in the
high class, from the log2 fold change in the data). A negative attribution is a gene whose change
pulls the score back towards the low class.

Next to the attribution, which describes the model, the table gives what the data show: the
log2 fold change of the gene between the two classes and the rank correlation of its expression
with the score.

Two things make a list of single genes from a single model unreliable, and both are dealt with here.

Seeds. Models that differ only in their seed are equally good and weigh the genes differently.
With several bundles (the ``seed_<n>`` directories of a sweep) every gene gets the mean over the
seeds, and the number of seeds in which it has the same sign and in which it is among the top genes.

Co-transcribed genes. Genes that always change together carry the same information, and a model
may put its weight on any of them. Neighbouring genes on the same strand whose expression is
correlated within studies are therefore joined into expression units (``expression_units``), a
stand-in for operons made from the atlas itself, and the attributions of a unit are added up. The
unit table is the one to read: a unit that ranks high in most seeds is a finding, whichever of
its genes a given model happened to use.

Studies. The high class is a sample of studies, and for a head with few of them (stationary phase:
about 30) the choice of studies moves the ranking more than the seed does. ``study_bootstrap``
draws the studies again and reports, per gene and unit, how often it stays among the top rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from .data import TrainingData


def head_score(model, x: torch.Tensor, c: torch.Tensor, head: str) -> torch.Tensor:
    """Log-odds of the head's call, with the block encoders in the graph."""
    logits = model(x, c, sample=False, train_blocks=True)["logits"][head]
    return logits[:, 1] - logits[:, 0]


def study_balanced_weights(gse: np.ndarray) -> np.ndarray:
    """Weights that sum to 1 and give every study the same total."""
    n = pd.Series(gse).map(pd.Series(gse).value_counts()).to_numpy(dtype=float)
    return (1.0 / n) / (1.0 / n).sum()


def integrated_gradients(model, X: torch.Tensor, C: torch.Tensor, base: torch.Tensor, head: str, device,
                         steps: int = 32, chunk: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attribution of every feature for every row of ``X`` relative to ``base`` (midpoint rule
    over ``steps`` points of the line). Returns (attributions, score of the sample, score of the
    baseline with the sample's covariates)."""
    model.eval()
    base = base.to(device)
    attr, s_end, s_base = [], [], []
    for i in range(0, X.shape[0], chunk):
        x, c = X[i:i + chunk].to(device), C[i:i + chunk].to(device)
        delta = x - base
        total = torch.zeros_like(x)
        for k in range(steps):
            xi = (base + (k + 0.5) / steps * delta).detach().requires_grad_(True)
            total += torch.autograd.grad(head_score(model, xi, c, head).sum(), xi)[0]
        attr.append((delta * total / steps).detach().cpu())
        with torch.no_grad():
            s_end.append(head_score(model, x, c, head).cpu())
            s_base.append(head_score(model, base.expand_as(x), c, head).cpu())
    return torch.cat(attr).numpy(), torch.cat(s_end).numpy(), torch.cat(s_base).numpy()


def spearman_columns(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Spearman correlation of every column of ``M`` with ``v`` (0 for a constant column)."""
    R = np.apply_along_axis(rankdata, 0, M)
    r = rankdata(v)
    R, r = R - R.mean(0), r - r.mean()
    den = np.sqrt((R ** 2).sum(0) * (r ** 2).sum())
    return np.where(den > 0, (R * r[:, None]).sum(0) / np.where(den > 0, den, 1.0), 0.0)


def default_contrasts(classes: tuple) -> list:
    """First against last class, and for more than two classes also every neighbouring pair."""
    out = [(classes[0], classes[-1])]
    if len(classes) > 2:
        out += [(classes[i], classes[i + 1]) for i in range(len(classes) - 1)]
    return out


def gene_drivers(model, data: TrainingData, head: str, low: str, high: str, device, steps: int = 32) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """The driver table of one contrast (one row per model feature, strongest first), a summary, and
    the mean attribution of every high-class study (studies x model features, studies in the order
    of their codes; the table's ``attribution`` is the mean of its rows)."""
    classes = data.fit_classes[head]
    for name in (low, high):
        if name not in classes:
            raise ValueError(f"{head}: {name!r} is not one of its classes {classes}")
    y, gse = data.y_fit[head], data.gse_codes
    lo, hi = np.where(y == classes.index(low))[0], np.where(y == classes.index(high))[0]
    if not len(lo) or not len(hi):
        raise ValueError(f"{head}: no sample of {low!r} or of {high!r}")
    w_lo, w_hi = study_balanced_weights(gse[lo]), study_balanced_weights(gse[hi])
    X = data.X.numpy()
    base = torch.tensor(w_lo @ X[lo], dtype=torch.float32)[None, :]
    attr, s_hi, s_base = integrated_gradients(model, data.X[torch.as_tensor(hi)], data.C[torch.as_tensor(hi)], base, head, device, steps)
    mean_attr = w_hi @ attr
    logx = X * data.zscore.sd + data.zscore.mu                              # log2 normalised expression
    both = np.concatenate([lo, hi])
    with torch.no_grad():
        score = np.concatenate([head_score(model, data.X[torch.as_tensor(both[i:i + 2048])].to(device),
                                           data.C[torch.as_tensor(both[i:i + 2048])].to(device), head).cpu().numpy()
                                for i in range(0, len(both), 2048)])
    table = pd.DataFrame({
        "feature": data.fs.ids,
        "attribution": mean_attr,
        "share": np.abs(mean_attr) / np.abs(mean_attr).sum(),
        "same_sign": ((np.sign(attr) == np.sign(mean_attr)[None, :]) * w_hi[:, None]).sum(0),
        "log2fc": w_hi @ logx[hi] - w_lo @ logx[lo],
        "score_spearman": spearman_columns(logx[both], score),
    })
    table["direction"] = np.where(table["log2fc"] >= 0, "up", "down")
    table = table.sort_values("share", ascending=False).reset_index(drop=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    explained, gap = float(w_hi @ attr.sum(1)), float(w_hi @ (s_hi - s_base))
    pos = table["attribution"] > 0
    summary = {"head": head, "low": low, "high": high, "steps": steps,
               "n_samples": {low: int(len(lo)), high: int(len(hi))},
               "n_studies": {low: int(len(np.unique(gse[lo]))), high: int(len(np.unique(gse[hi])))},
               "score_gap_high_minus_baseline": gap, "sum_of_attributions": explained,
               "share_of_attribution": {f"top_{k}": float(table["share"].head(k).sum()) for k in (10, 50, 200)},
               "share_of_attribution_towards_high": float(table.loc[pos, "share"].sum()),
               "towards_high_up_vs_down": {d: float(table.loc[pos & (table["direction"] == d), "share"].sum()) for d in ("up", "down")}}
    hi_gse = gse[hi]
    per_study = np.stack([attr[hi_gse == st].mean(0) for st in np.unique(hi_gse)])
    return table, summary, per_study


def study_bootstrap(per_study: np.ndarray, top: int, n_boot: int = 200, seed: int = 0) -> dict:
    """Stability of the attributions when the high-class studies are drawn again with replacement
    (every study counting once, as in the attribution itself). ``per_study`` is studies x rows.
    Returns per row: ``study_sd``, the spread of the mean attribution over the draws;
    ``study_in_top``, the share of draws in which the row is among the ``top`` rows by absolute
    attribution; ``study_same_sign``, the share of draws with the sign of the full mean. The
    baseline (the low class) is held fixed, so this is the part of the uncertainty that comes from
    which high-class studies the atlas holds. The draws depend only on ``seed`` and the number of
    studies, so gene and unit tables of one contrast use the same draws."""
    J, n = per_study.shape
    rng = np.random.default_rng(seed)
    full = per_study.mean(0)
    top = min(top, n)
    boot = np.empty((n_boot, n))
    in_top = np.zeros(n)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, J, J), minlength=J) / J
        boot[b] = w @ per_study
        in_top[np.argpartition(-np.abs(boot[b]), top - 1)[:top]] += 1
    return {"study_sd": boot.std(0, ddof=1) if n_boot > 1 else np.zeros(n), "study_in_top": in_top / n_boot,
            "study_same_sign": (np.sign(boot) == np.sign(full)[None, :]).mean(0)}


def unit_matrix(unit: np.ndarray) -> np.ndarray:
    """Features x units 0/1 matrix in the unit order of ``np.unique(unit)``; ``A @ G`` adds the
    attributions of the members of every unit."""
    _, inv = np.unique(unit, return_inverse=True)
    G = np.zeros((len(unit), inv.max() + 1))
    G[np.arange(len(unit)), inv] = 1.0
    return G


def expression_units(feat: pd.DataFrame, ids: list, logx: np.ndarray, gse: np.ndarray, min_r: float = 0.6,
                     max_gap: int = 500, min_study: int = 4) -> np.ndarray:
    """Unit number of every model feature. Two features that follow each other on the same strand of
    the same replicon, at most ``max_gap`` bp apart, are joined when their log expression has a
    Pearson correlation of at least ``min_r`` after the mean of every study has been removed
    (studies with at least ``min_study`` samples). Features without a position are units of their own."""
    cols = {"Chr", "Start", "End", "Strand"}
    unit = np.arange(len(ids))
    if not cols <= set(feat.columns):
        return unit
    pos = feat.reindex(ids)
    start, end = pd.to_numeric(pos["Start"], errors="coerce").to_numpy(), pd.to_numeric(pos["End"], errors="coerce").to_numpy()
    sizes = pd.Series(gse).map(pd.Series(gse).value_counts()).to_numpy()
    keep = sizes >= min_study
    Z = logx[keep] - pd.DataFrame(logx[keep]).groupby(gse[keep]).transform("mean").to_numpy()
    Z = Z - Z.mean(0)
    Z = Z / np.maximum(np.sqrt((Z ** 2).sum(0)), 1e-12)
    order = [i for i in np.lexsort((start, pos["Chr"].astype(str).to_numpy())) if not np.isnan(start[i])]
    for a, b in zip(order[:-1], order[1:]):
        if (pos["Chr"].iloc[a] == pos["Chr"].iloc[b] and pos["Strand"].iloc[a] == pos["Strand"].iloc[b]
                and start[b] - end[a] <= max_gap and float(Z[:, a] @ Z[:, b]) >= min_r):
            unit[b] = unit[a]
    return unit


def unit_name(symbols: list) -> str:
    """`pspABCD` for genes that share their first three letters, otherwise `first-last`, with the
    number of genes when there are more than two."""
    symbols = [s for s in symbols if s]
    if len(symbols) <= 1:
        return symbols[0] if symbols else ""
    if all(len(s) == 4 and s[:3] == symbols[0][:3] for s in symbols):
        return symbols[0][:3] + "".join(s[3] for s in symbols)
    return f"{symbols[0]}-{symbols[-1]}" + (f" ({len(symbols)})" if len(symbols) > 2 else "")


def combine_seeds(tables: list, key: str, top: int) -> pd.DataFrame:
    """One table from the tables of several seeds (same rows, any order): ``attribution`` is the mean
    over the seeds and the ranking is made from it; ``seeds_same_sign`` and ``seeds_in_top`` count
    the seeds in which a row has the sign of the mean and is among that seed's ``top`` rows."""
    first = tables[0].set_index(key)
    A = np.column_stack([t.set_index(key)["attribution"].reindex(first.index).to_numpy() for t in tables])
    R = np.column_stack([t.set_index(key)["rank"].reindex(first.index).to_numpy() for t in tables])
    out = first.drop(columns=[c for c in ("rank", "share", "attribution", "attribution_per_gene", "same_sign", "score_spearman")
                              if c in first.columns]).copy()
    mean = A.mean(1)
    out.insert(0, "attribution", mean)
    out.insert(1, "share", np.abs(mean) / np.abs(mean).sum())
    out.insert(2, "n_seeds", len(tables))
    out.insert(3, "seeds_same_sign", (np.sign(A) == np.sign(mean)[:, None]).sum(1))
    out.insert(4, "seeds_in_top", (R <= top).sum(1))
    out.insert(5, "attribution_sd", A.std(1))
    if "n_genes" in out.columns:
        out.insert(6, "attribution_per_gene", mean / out["n_genes"].to_numpy())
    for col in ("same_sign", "score_spearman"):
        if col in first.columns:
            out[col] = np.column_stack([t.set_index(key)[col].reindex(first.index).to_numpy() for t in tables]).mean(1)
    out = out.sort_values("share", ascending=False).reset_index()
    out.insert(1, "rank", np.arange(1, len(out) + 1))
    return out


def unit_table(genes: pd.DataFrame, unit: np.ndarray, ids: list) -> pd.DataFrame:
    """Attributions added up over the expression units (one row per unit, strongest first).
    ``genes`` is a driver table with ``feature``, ``attribution``, ``log2fc`` and, when present,
    ``symbol`` and ``product``; the members of a unit are listed in genome order. ``unit_id`` is the
    feature id of its first gene, ``lead_feature`` the gene with the largest attribution."""
    g = genes.set_index("feature").reindex(ids)
    g["unit"] = unit
    sym = g["symbol"] if "symbol" in g.columns else pd.Series(g.index, index=g.index)
    sym = sym.where(sym != "", pd.Series(g.index, index=g.index))
    rows = []
    for u, sub in g.groupby("unit", sort=False):
        best = sub["attribution"].abs().idxmax()
        rows.append({"unit": unit_name(list(sym[sub.index])), "n_genes": len(sub), "attribution": float(sub["attribution"].sum()),
                     "log2fc": float(sub["log2fc"].mean()), "genes": ",".join(sym[sub.index]), "unit_id": sub.index[0], "lead_feature": best,
                     "products": "; ".join(dict.fromkeys(p for p in (sub["product"] if "product" in sub.columns else []) if p))})
    out = pd.DataFrame(rows)
    out["share"] = out["attribution"].abs() / out["attribution"].abs().sum()
    out["direction"] = np.where(out["log2fc"] >= 0, "up", "down")
    out["attribution_per_gene"] = out["attribution"] / out["n_genes"]
    out = out.sort_values("share", ascending=False).reset_index(drop=True)
    out.insert(1, "rank", np.arange(1, len(out) + 1))
    return out[["unit", "rank", "n_genes", "attribution", "attribution_per_gene", "share", "direction", "log2fc", "genes", "unit_id", "lead_feature", "products"]]
