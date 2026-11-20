"""`ecoli_atlas axes`: one dimension of the embedding per ordered set of categories.

An axis is a metadata column and an ordered list of its values, for example growth phase from lag
to late stationary, or medium from complex broth to minimal medium. Two scores are made for every
sample from its embedding:

    ordinal   a proportional-odds (cumulative logit) model: one linear score of the embedding and one
              threshold between each pair of neighbouring categories. The score is not bounded, so
              a sample can lie beyond the last category that was fitted. For samples of the fitted
              categories the score is out of fold: the model that scores a sample has seen no sample
              of its study.
    anchor    the direction that separates the two end categories (linear discriminant), fitted on
              those two alone. Every category in between is placed without having been fitted.

Both are scaled so that the median of the first fitted category is 0 and that of the last is 1.
Categories named in ``holdout`` are left out of both fits and scored with the model of all other
studies' samples: this is the test of whether the ordinal score extrapolates (late stationary
beyond stationary) or interpolates (an in-between category left out).

Only the embedding itself is taken as given. It was trained with the samples of the training pool,
so every table also reports the samples the embedding has never fitted (TEST and projected).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SPLIT_NAMES = {1: "training_pool", 3: "test", 0: "projected"}
MAX_PER_STUDY_CATEGORY = 25      # a study's samples of one category beyond this number are downweighted


@dataclass
class AxisSpec:
    name: str
    column: str
    categories: list          # in order
    holdout: tuple = ()       # categories left out of the fits

    @classmethod
    def parse(cls, text: str, holdout=()) -> "AxisSpec":
        """``name:column:cat1,cat2,...``"""
        parts = text.split(":")
        if len(parts) != 3 or not parts[2]:
            raise ValueError(f"axis must be name:column:cat1,cat2,... , not {text!r}")
        cats = [c.strip() for c in parts[2].split(",") if c.strip()]
        spec = cls(parts[0].strip(), parts[1].strip(), cats, tuple(h for h in holdout if h in cats))
        if len(cats) < 2 or len(set(cats)) != len(cats):
            raise ValueError(f"axis {spec.name}: at least two different categories are needed")
        if len(spec.fitted) < 2:
            raise ValueError(f"axis {spec.name}: fewer than two categories are left to fit")
        return spec

    @property
    def fitted(self) -> list:
        return [c for c in self.categories if c not in self.holdout]


def block_columns(manifest: dict, blocks: str, n_columns: int) -> list:
    """Column positions of the embedding for ``blocks`` (``all`` or comma-separated block names)."""
    if blocks in (None, "", "all"):
        return list(range(n_columns))
    layout = {b["block"]: range(b["start"], b["end"]) for b in manifest.get("embedding_blocks", [])}
    missing = [b for b in blocks.split(",") if b not in layout]
    if missing:
        raise ValueError(f"the bundle has no embedding block {missing}; it has {sorted(layout)}")
    return [i for b in blocks.split(",") for i in layout[b]]


def sample_weights(rank: np.ndarray, study: np.ndarray) -> np.ndarray:
    """Every category has the same total weight; within a category a study counts at most as
    MAX_PER_STUDY_CATEGORY samples."""
    frame = pd.DataFrame({"rank": rank, "study": study})
    n = frame.groupby(["rank", "study"])["rank"].transform("size").to_numpy(dtype=float)
    w = np.minimum(1.0, MAX_PER_STUDY_CATEGORY / n)
    total = pd.Series(w).groupby(rank).transform("sum").to_numpy()
    return w / total


class OrdinalModel:
    """Proportional odds: P(rank <= k) = sigmoid(theta_k - x.w), thresholds theta increasing."""

    def __init__(self, l2: float = 1.0, max_iter: int = 300):
        self.l2, self.max_iter = l2, max_iter

    def fit(self, X: np.ndarray, rank: np.ndarray, weight: np.ndarray) -> "OrdinalModel":
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        levels = np.unique(rank)                         # the ranks present, in order
        code = np.searchsorted(levels, rank)
        k = len(levels)
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        y = torch.tensor(code, dtype=torch.long)
        wt = torch.tensor(weight / weight.sum(), dtype=torch.float64)
        w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        first = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        gaps = torch.zeros(max(k - 2, 0), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, first, gaps], max_iter=self.max_iter, line_search_fn="strong_wolfe")
        inf = torch.tensor([float("inf")], dtype=torch.float64)

        def loss_fn():
            theta = torch.cat([first, first + torch.cumsum(torch.nn.functional.softplus(gaps), 0)])
            edges = torch.cat([-inf, theta, inf])
            s = x @ w
            p = torch.sigmoid(edges[y + 1] - s) - torch.sigmoid(edges[y] - s)
            return -(wt * torch.log(p.clamp_min(1e-12))).sum() + self.l2 * (w ** 2).sum() / len(y)

        def closure():
            opt.zero_grad()
            value = loss_fn()
            value.backward()
            return value

        opt.step(closure)
        self.w = w.detach().numpy()
        self.levels = levels
        self.thresholds = torch.cat([first, first + torch.cumsum(torch.nn.functional.softplus(gaps), 0)]).detach().numpy()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w


class AnchorModel:
    """Linear discriminant direction between two groups of samples."""

    def fit(self, X: np.ndarray, is_last: np.ndarray, weight: np.ndarray) -> "AnchorModel":
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        x = (X - self.mu) / self.sd
        w = weight / weight.sum()
        m0 = (w[~is_last, None] * x[~is_last]).sum(0) / w[~is_last].sum()
        m1 = (w[is_last, None] * x[is_last]).sum(0) / w[is_last].sum()
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x, is_last.astype(int))
        direction = lda.coef_[0]
        self.w = direction if (m1 - m0) @ direction >= 0 else -direction
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w


def unit_scale(score: np.ndarray, rank: np.ndarray, rows: np.ndarray, lo: int, hi: int):
    """(offset, scale) that put the median of rank ``lo`` at 0 and of rank ``hi`` at 1 among ``rows``."""
    a = np.median(score[rows][rank[rows] == lo])
    b = np.median(score[rows][rank[rows] == hi])
    return a, (b - a) if b != a else 1.0


def study_folds(study: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """A fold per sample, whole studies together, about the same number of samples per fold."""
    sizes = pd.Series(study).value_counts()
    order = list(np.random.RandomState(seed).permutation(sizes.index.to_numpy()))
    order.sort(key=lambda g: -sizes[g])
    load, fold_of = np.zeros(n_folds), {}
    for g in order:
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += sizes[g]
    return pd.Series(study).map(fold_of).to_numpy()


def axis_scores(Z: np.ndarray, frame: pd.DataFrame, spec: AxisSpec, n_folds: int = 5, seed: int = 0, l2: float = 1.0) -> dict:
    """Scores of every sample on one axis.

    Z: samples x embedding columns. frame: per sample ``gse``, ``split`` and the category (column
    ``category``, '' when the sample has none of the axis' categories or is filtered out).
    Returns {"scores": frame, "thresholds": dict, "weights": dict}.
    """
    rank_of = {c: i for i, c in enumerate(spec.categories)}
    rank = frame["category"].map(rank_of).fillna(-1).astype(int).to_numpy()
    fitted_ranks = [rank_of[c] for c in spec.fitted]
    lo, hi = fitted_ranks[0], fitted_ranks[-1]
    study = frame["gse"].to_numpy()
    is_fit = np.isin(rank, fitted_ranks)
    fit_rows = np.where(is_fit)[0]
    present = [r for r in fitted_ranks if (rank[fit_rows] == r).any()]
    if lo not in present or hi not in present:
        raise ValueError(f"axis {spec.name}: no sample of its first or last fitted category")
    n_studies = len(np.unique(study[fit_rows]))
    if n_studies < 2:
        raise ValueError(f"axis {spec.name}: samples of the fitted categories come from one study only")
    for r in (lo, hi):
        if len(np.unique(study[fit_rows][rank[fit_rows] == r])) < 2:
            raise ValueError(f"axis {spec.name}: the end category {spec.categories[r]!r} has samples of one study only, so it cannot "
                             "be told from that study; leave it out of the axis or name it in --holdout")
    k = min(n_folds, n_studies)
    fold = np.full(len(frame), -1)
    fold[fit_rows] = study_folds(study[fit_rows], k, seed)

    def fit_pair(rows):
        w = sample_weights(rank[rows], study[rows])
        ordinal = OrdinalModel(l2).fit(Z[rows], rank[rows], w)
        ends = rows[np.isin(rank[rows], (lo, hi))]
        anchor = AnchorModel().fit(Z[ends], rank[ends] == hi, sample_weights(rank[ends], study[ends]))
        out = {}
        for name, model in (("ordinal", ordinal), ("anchor", anchor)):
            s = model.score(Z)
            a, b = unit_scale(s, rank, rows, lo, hi)
            out[name] = (model, (s - a) / b, a, b)
        return out

    scores = {name: np.full(len(frame), np.nan) for name in ("ordinal", "anchor")}
    full = fit_pair(fit_rows)
    fold_models = {}
    for f in range(k):
        train = fit_rows[fold[fit_rows] != f]
        if (rank[train] == lo).any() and (rank[train] == hi).any():   # else the fold holds an end category entirely
            fold_models[f] = fit_pair(train)
    # a sample is scored by the model that has not seen its study: the fold's model for a study with
    # fitted samples, the model of all fitted samples for every other study
    study_fold = dict(zip(study[fit_rows], fold[fit_rows]))
    sample_fold = np.array([study_fold.get(g, -1) for g in study])
    for f in np.unique(sample_fold):
        idx = np.where(sample_fold == f)[0]
        if f >= 0 and f not in fold_models:
            idx = idx[~is_fit[idx]]                       # no out-of-fold score for the fitted samples of such a fold
            res = full
        else:
            res = fold_models[f] if f >= 0 else full
        for name in scores:
            scores[name][idx] = res[name][1][idx]
    out = frame[["gse", "split", "category"]].copy()
    out["rank"] = np.where(rank >= 0, rank, np.nan)
    out["role"] = np.where(is_fit, "fitted", np.where(rank >= 0, "holdout", ""))
    for name in scores:
        out[name] = scores[name]
    ordinal_model, _, a, b = full["ordinal"]
    thresholds = {f"{spec.categories[int(r0)]}|{spec.categories[int(r1)]}": float((t - a) / b)
                  for r0, r1, t in zip(ordinal_model.levels[:-1], ordinal_model.levels[1:], ordinal_model.thresholds)}
    return {"scores": out, "thresholds": thresholds,
            "weights": {name: (full[name][0].w / full[name][3]).tolist() for name in full}}


def pair_order(scores: pd.DataFrame, column: str) -> dict:
    """Among pairs of samples of one study in different categories, the share whose scores are in
    the order of the categories (ties count half): over all pairs, as the mean over studies (every
    study counts once, so one large study does not decide the number), and per pair of categories."""
    per, total, right, per_study = {}, 0.0, 0.0, []
    sub = scores[scores["rank"].notna() & scores[column].notna()]
    for _, g in sub.groupby("gse"):
        if g["rank"].nunique() < 2:
            continue
        r, s, c = g["rank"].to_numpy(), g[column].to_numpy(), g["category"].to_numpy()
        i, j = np.where(r[:, None] < r[None, :])
        if not len(i):
            continue
        ok = (s[i] < s[j]) + 0.5 * (s[i] == s[j])
        total, right = total + len(i), right + ok.sum()
        per_study.append(ok.mean())
        pairs = pd.DataFrame({"pair": [f"{a}<{b}" for a, b in zip(c[i], c[j])], "ok": ok}).groupby("pair")["ok"]
        for name, n, m in zip(pairs.size().index, pairs.size().to_numpy(), pairs.mean().to_numpy()):
            cell = per.setdefault(name, [0, 0.0, []])
            cell[0] += int(n)
            cell[1] += float(n * m)
            cell[2].append(float(m))
    return {"n_pairs": int(total), "n_studies": len(per_study), "share_in_order": float(right / total) if total else None,
            "share_in_order_mean_over_studies": float(np.mean(per_study)) if per_study else None,
            "by_pair": {k: {"n_pairs": v[0], "n_studies": len(v[2]), "share_in_order": float(v[1] / v[0]),
                            "share_in_order_mean_over_studies": float(np.mean(v[2]))} for k, v in per.items()}}


def summarize(scores: pd.DataFrame, spec: AxisSpec) -> dict:
    """Numbers for one axis: per method, for all samples and for those the embedding never fitted."""
    out = {"categories": spec.categories, "holdout": list(spec.holdout), "n_samples": {}, "methods": {}}
    has = scores["rank"].notna()
    for c in spec.categories:
        sel = scores["category"] == c
        out["n_samples"][c] = {"samples": int(sel.sum()), "studies": int(scores.loc[sel, "gse"].nunique()),
                               "not_in_embedding_fit": int((sel & (scores["split"] != 1)).sum())}
    unseen = scores["split"] != 1
    for method in ("ordinal", "anchor"):
        block = {}
        for label, rows in (("all", has), ("not_in_embedding_fit", has & unseen)):
            sub = scores[rows & scores[method].notna()]
            rho = spearmanr(sub["rank"], sub[method]).statistic if sub["rank"].nunique() > 1 else np.nan
            block[label] = {"n": int(len(sub)), "spearman": None if np.isnan(rho) else float(rho),
                            "within_study": pair_order(sub, method)}
        block["category_medians"] = {c: float(scores.loc[scores["category"] == c, method].median())
                                     for c in spec.categories if (scores["category"] == c).any()}
        if spec.holdout:
            ranks = {c: i for i, c in enumerate(spec.categories)}
            fitted_ranks = [ranks[c] for c in spec.fitted]
            check = {}
            for c in spec.holdout:
                s = scores.loc[scores["category"] == c, method].dropna()
                if s.empty:
                    continue
                below = [r for r in fitted_ranks if r < ranks[c]]
                above = [r for r in fitted_ranks if r > ranks[c]]
                entry = {"n": int(len(s)), "median": float(s.median())}
                if below:
                    ref = scores.loc[scores["rank"] == max(below), method].median()
                    entry["share_above_" + spec.categories[max(below)]] = float((s > ref).mean())
                if above:
                    ref = scores.loc[scores["rank"] == min(above), method].median()
                    entry["share_below_" + spec.categories[min(above)]] = float((s < ref).mean())
                check[c] = entry
            block["holdout"] = check
        out["methods"][method] = block
    return out


def load_embedding(models_dir, blocks: str = "all") -> tuple:
    """(embedding frame restricted to ``blocks``, splits frame, manifest) of a bundle directory."""
    import json
    from pathlib import Path
    d = Path(models_dir)
    with open(d / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    emb = pd.read_csv(d / "embedding.tsv", sep="\t", index_col=0)
    emb.index = emb.index.astype(str)
    cols = block_columns(manifest, blocks, emb.shape[1])
    splits = pd.read_csv(d / "splits.csv", index_col=0)
    splits.index = splits.index.astype(str)
    return emb.iloc[:, cols], splits.loc[emb.index], manifest


def category_frame(meta: pd.DataFrame, splits: pd.DataFrame, spec: AxisSpec, where: dict) -> pd.DataFrame:
    """gse, split and the axis category per sample ('' for samples outside ``where`` or the axis)."""
    if spec.column not in meta.columns:
        raise ValueError(f"axis {spec.name}: the metadata has no column {spec.column!r}")
    keep = pd.Series(True, index=meta.index)
    for col, values in where.items():
        if col not in meta.columns:
            raise ValueError(f"--where: the metadata has no column {col!r}")
        keep &= meta[col].astype(str).str.strip().isin(values)
    value = meta[spec.column].astype(str).str.strip()
    cat = value.where(value.isin(spec.categories) & keep, "")
    return pd.DataFrame({"gse": splits["gse"].astype(str).to_numpy(), "split": splits["split"].to_numpy(), "category": cat.to_numpy()},
                        index=meta.index)
