"""Study-level train / test split and cross-validation folds.

TEST and the folds are chosen among the reference samples only (``labels.is_reference``: untreated,
reference genotype), whole studies at a time:

    reference   the sample is a reference sample
    split       3 = TEST: reference samples of the TEST studies
                1 = training pool: all samples of the other studies
                0 = projected only: withheld studies, and the other samples of the TEST studies
    cv_fold     1..n for the reference samples of the training pool; empty for its other samples,
                which are never scored. When a fold is held out, every sample of its studies is
                left out of the fit (``data.study_fold``)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEADS, TrainConfig

BALANCE_VARS = tuple(HEADS)
SPLIT_COLUMNS = ["gse", "reference", "split", "cv_fold"]


def _balance_tables(df, gse_idx, n_gses, balance_vars):
    global_props, per_study_counts = {}, {}
    for v in balance_vars:
        s = df[v]
        classes = sorted(s.dropna().unique(), key=str)
        codes = s.map({c: i for i, c in enumerate(classes)}).values
        mat = np.zeros((n_gses, len(classes)))
        valid = ~pd.isna(codes)
        np.add.at(mat, (gse_idx[valid], codes[valid].astype(int)), 1)
        per_study_counts[v] = mat
        global_props[v] = mat.sum(0) / mat.sum()
    return global_props, per_study_counts


def split_test_only(df, test_frac, seed, n_candidates=500, balance_vars=BALANCE_VARS,
                    size_weight=2.0, study_count_weight=3.0, forced_test_studies=(), excluded_test_studies=(),
                    log=None):
    """Randomised search over study permutations for a TEST set of about ``test_frac`` of the samples
    that mirrors the corpus class proportions. Returns a boolean Series over ``df``."""
    balance_vars = tuple(v for v in balance_vars if v in df.columns)
    gses = list(df["gse"].unique())
    n, n_g = len(df), len(gses)
    gse_pos = {g: i for i, g in enumerate(gses)}
    gse_idx = df["gse"].map(gse_pos).values
    sizes = np.bincount(gse_idx, minlength=n_g)
    forced = np.array(sorted({gse_pos[g] for g in forced_test_studies if g in gse_pos}), dtype=int)
    excluded = np.array(sorted({gse_pos[g] for g in excluded_test_studies if g in gse_pos}), dtype=int)
    global_props, per_study_counts = _balance_tables(df, gse_idx, n_g, balance_vars)

    def score(test_studies):
        n_k = sizes[test_studies].sum()
        if n_k == 0:
            return np.inf
        sc = size_weight * ((n_k / n) - test_frac) ** 2
        sc += study_count_weight * ((len(test_studies) / n_g) - test_frac) ** 2
        for v in balance_vars:
            cls = per_study_counts[v][test_studies].sum(0)
            props = cls / cls.sum() if cls.sum() > 0 else cls
            sc += ((props - global_props[v]) ** 2).sum()
        return sc

    best_sc, best_studies, best_i = np.inf, None, -1
    for i in range(n_candidates):
        rng = np.random.RandomState(seed * 1_000_003 + i)
        free = np.setdiff1d(np.arange(n_g), np.union1d(forced, excluded))
        order = np.concatenate([forced, rng.permutation(free), excluded])
        cum = np.cumsum(sizes[order])
        n_max = n_g - len(excluded)
        n_test = max(min(int(np.searchsorted(cum, test_frac * n, side="right")) + 1, n_max), len(forced))
        sc = score(order[:n_test])
        if sc < best_sc:
            best_sc, best_studies, best_i = sc, order[:n_test], i
    chosen = {gses[i] for i in best_studies}
    assert not (chosen & set(excluded_test_studies))
    if log:
        log(f"[TEST search] {n_candidates} candidates -> #{best_i}, score={best_sc:.5f} | "
            f"{len(best_studies)} studies (expected ~{test_frac * n_g:.0f})")
    return df["gse"].isin(chosen)


def kfold_by_study(df, eligible_mask, n_folds, seed, n_candidates=500, balance_vars=BALANCE_VARS,
                   size_weight=2.0, study_count_weight=8.0, study_balance=1.0, log=None):
    """Greedy assignment of whole studies to folds, best of ``n_candidates`` random orders."""
    pool = df.loc[eligible_mask]
    balance_vars = tuple(v for v in balance_vars if v in pool.columns)
    gses = list(pool["gse"].unique())
    n, n_g = len(pool), len(gses)
    gse_pos = {g: i for i, g in enumerate(gses)}
    gse_idx = pool["gse"].map(gse_pos).values
    sizes = np.bincount(gse_idx, minlength=n_g)
    target_frac = 1.0 / n_folds
    target_n = max(n / n_folds, 1.0)
    target_studies = max(n_g / n_folds, 1.0)
    global_props, per_study_counts = _balance_tables(pool, gse_idx, n_g, balance_vars)

    def score(fold_of_study):
        sc = 0.0
        for k in range(n_folds):
            sk = np.where(fold_of_study == k)[0]
            n_k = sizes[sk].sum()
            if n_k == 0 or len(sk) == 0:
                return np.inf
            sc += size_weight * ((n_k / n) - target_frac) ** 2
            sc += study_count_weight * ((len(sk) / n_g) - target_frac) ** 2
            for v in balance_vars:
                cls = per_study_counts[v][sk].sum(0)
                props = cls / cls.sum() if cls.sum() > 0 else cls
                sc += ((props - global_props[v]) ** 2).sum()
        return sc

    best_sc, best_assign, best_i = np.inf, None, -1
    for i in range(n_candidates):
        rng = np.random.RandomState(seed * 1_000_003 + i)
        order = rng.permutation(n_g)
        order = order[np.argsort(-sizes[order], kind="stable")]  # largest study first
        fold_of_study = -np.ones(n_g, dtype=int)
        fold_n, fold_s = np.zeros(n_folds), np.zeros(n_folds)
        for s in order:
            deficit = (fold_n / target_n) + study_balance * (fold_s / target_studies)
            k = int(np.argmin(deficit))
            fold_of_study[s] = k
            fold_n[k] += sizes[s]
            fold_s[k] += 1
        sc = score(fold_of_study)
        if sc < best_sc:
            best_sc, best_assign, best_i = sc, fold_of_study.copy(), i
    if log:
        fn = [int(sizes[np.where(best_assign == k)[0]].sum()) for k in range(n_folds)]
        fs = [int((best_assign == k).sum()) for k in range(n_folds)]
        log(f"[fold search] {n_candidates} candidates -> #{best_i}, score={best_sc:.5f} | samples {fn} | studies {fs}")
    out = pd.Series(np.nan, index=df.index)
    out.loc[eligible_mask] = pool["gse"].map({gses[i]: k + 1 for i, k in enumerate(best_assign)}).values
    return out


def make_splits(gse: pd.Series, labels: pd.DataFrame, reference, cfg: TrainConfig, log=None) -> pd.DataFrame:
    """Assign every sample (see the module text). About ``test_frac`` of the reference samples
    (whole studies, with the forced ones) are TEST, the reference samples of the other studies are
    put in ``n_folds`` study folds; both are balanced on the two head labels. A study with more
    than the TEST share of the reference samples stays in the training pool."""
    df = pd.DataFrame({"gse": gse.astype(str).values}, index=gse.index).join(labels[list(BALANCE_VARS)])
    ref = np.asarray(reference, dtype=bool)
    withheld = df["gse"].isin(cfg.withheld_studies).to_numpy()
    work = df.loc[ref & ~withheld]
    sizes = work["gse"].value_counts()
    too_large = sorted(sizes.index[sizes > cfg.test_frac * len(work)])
    forced = sorted(set(cfg.test_studies) & set(sizes.index))
    is_test = split_test_only(work, cfg.test_frac, seed=cfg.seed, n_candidates=cfg.n_candidates,
                              forced_test_studies=forced, excluded_test_studies=[g for g in too_large if g not in forced], log=log)
    in_test_study = df["gse"].isin(set(work.loc[is_test.to_numpy(), "gse"]) | set(cfg.test_studies)).to_numpy()
    out = pd.DataFrame({"gse": df["gse"], "reference": ref}, index=df.index)
    out["split"] = np.where(withheld, 0, np.where(in_test_study, np.where(ref, 3, 0), 1))
    out["cv_fold"] = kfold_by_study(df, ref & (out["split"] == 1).to_numpy(), cfg.n_folds, seed=cfg.seed,
                                    n_candidates=cfg.n_candidates, study_balance=cfg.fold_study_weight, log=log)
    check_no_leakage(out, cfg)
    return out


def load_splits(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, dtype={0: str})
    df.index = df.index.astype(str).str.strip()
    missing = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: splits file lacks columns {missing}")
    df["reference"] = df["reference"].astype(str).str.strip().str.lower().isin(("true", "1"))
    return df[SPLIT_COLUMNS]


def check_no_leakage(splits: pd.DataFrame, cfg: TrainConfig) -> None:
    ref = splits["reference"].to_numpy(dtype=bool)
    train = set(splits.loc[splits["split"] == 1, "gse"].astype(str))
    test = set(splits.loc[splits["split"] == 3, "gse"].astype(str))
    problems = []
    if train & test:
        problems.append("a study is in both TRAIN and TEST")
    if (train | test) & set(cfg.withheld_studies):
        problems.append("a withheld study is in a split")
    if set(cfg.test_studies) & train:
        problems.append("a forced TEST study is in TRAIN")
    if (~ref & (splits["split"] == 3).to_numpy()).any():
        problems.append("a sample that is not a reference sample is in TEST")
    pool = splits[(splits["split"] == 1).to_numpy() & ref]
    if pool["cv_fold"].isna().any() or (pool.groupby("gse")["cv_fold"].nunique() > 1).any():
        problems.append("a reference sample of the training pool has no fold, or a study is in two folds")
    if splits.loc[~ref | (splits["split"] != 1).to_numpy(), "cv_fold"].notna().any():
        problems.append("a sample outside the reference samples of the training pool has a fold")
    if problems:
        raise ValueError("split leakage: " + "; ".join(problems))


def bucket_labels(splits: pd.DataFrame) -> pd.Series:
    """'fold1'..'foldN' / 'test' per reference sample of the split; 'train_only' for the other
    samples of the training pool and 'projected' for the samples in no split."""
    lab = pd.Series("projected", index=splits.index, dtype="object")
    lab[splits["split"] == 3] = "test"
    lab[splits["split"] == 1] = "train_only"
    fold = (splits["split"] == 1).to_numpy() & splits["cv_fold"].notna().to_numpy()
    lab[fold] = "fold" + splits.loc[fold, "cv_fold"].astype(int).astype(str)
    return lab
