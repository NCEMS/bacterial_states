"""Study-level train / test split and cross-validation folds.

Codes: split 0 = projected only (not used for fitting), 1 = training pool, 3 = TEST.
``*_base`` columns are the assignment over wild-type reference samples; ``split`` / ``cv_fold`` are
the final assignment after adding non-reference samples to the pool (``cv_fold`` NaN) and removing
the withheld study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TrainConfig

BALANCE_VARS = ("growth_phase", "medium_type", "is_lab_strain")
SPLIT_COLUMNS = ["gse", "split_base", "cv_fold_base", "split", "cv_fold"]


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
                   size_weight=2.0, study_count_weight=8.0, study_balance=1.0, forced_fold_studies=None, log=None):
    """Greedy assignment of whole studies to folds, best of ``n_candidates`` random orders."""
    forced_fold_studies = forced_fold_studies or {}
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
    forced = {gse_pos[g]: k for g, k in forced_fold_studies.items() if g in gse_pos}
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
        for s, k in forced.items():
            fold_of_study[s] = k
            fold_n[k] += sizes[s]
            fold_s[k] += 1
        for s in order:
            if fold_of_study[s] != -1:
                continue
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


def _spread_studies(study_list, fold_map, n_folds, k_per_fold=2):
    fresh = [g for g in study_list if g not in fold_map]
    for i, g in enumerate(fresh[:k_per_fold * n_folds]):
        fold_map[g] = i % n_folds


def make_splits(meta: pd.DataFrame, cfg: TrainConfig, log=None) -> pd.DataFrame:
    """Reproduce the notebook's split construction from harmonised labels (``build_labels`` output).
    Returns a frame with ``gse, split_base, cv_fold_base`` indexed like ``meta``."""
    wt_mask = meta["is_wildtype"].values.astype(bool)
    meta_ref = meta[wt_mask].copy()
    poc = {cfg.withheld_study} if cfg.withheld_study else set()
    case = set(cfg.case_studies)

    # spread studies carrying secondary variables across folds
    forced_fold: dict = {}
    cs_n = meta_ref.groupby("gse")["carbon_source_c"].apply(lambda s: s.notna().sum())
    _spread_studies(list(cs_n[cs_n > 0].sort_values(ascending=False).index), forced_fold, cfg.n_folds)
    if meta_ref["is_lab_strain"].notna().any():
        ls_min = meta_ref["is_lab_strain"].value_counts().idxmin()
        ls_n = meta_ref[meta_ref["is_lab_strain"] == ls_min].groupby("gse").size()
        _spread_studies(list(ls_n.sort_values(ascending=False).index), forced_fold, cfg.n_folds)

    # guarantee every class of every reported factor appears in TEST
    force_cov: set = set()
    for v in ["growth_phase", "medium_type", "carbon_source", "is_lab_strain"]:
        for cls in sorted(meta_ref[v].dropna().unique(), key=str):
            if any(int((meta_ref.loc[meta_ref["gse"] == g, v] == cls).sum()) > 0 for g in force_cov):
                continue
            cnt = meta_ref[meta_ref[v] == cls].groupby("gse").size()
            if not len(cnt):
                continue
            elig = cnt[cnt >= cfg.min_class_n].sort_values()
            force_cov.add(elig.index[0] if len(elig) else cnt.sort_values(ascending=False).index[0])

    forced_test = sorted((force_cov | case) - poc)
    if log:
        n_forced = int(meta["gse"].isin(forced_test).sum())
        log(f"forced into TEST: {len(forced_test)} studies, {n_forced} samples ({100 * n_forced / len(meta):.1f}%)")
    forced_fold = {g: k for g, k in forced_fold.items() if g not in set(forced_test)}

    is_test_ref = split_test_only(meta_ref, cfg.test_frac, seed=cfg.seed, n_candidates=cfg.n_candidates,
                                  balance_vars=BALANCE_VARS, forced_test_studies=forced_test,
                                  excluded_test_studies=sorted(poc), log=log)
    test_studies = set(meta_ref.loc[is_test_ref, "gse"])
    out = pd.DataFrame(index=meta.index)
    out["gse"] = meta["gse"].values
    out["split_base"] = np.where(~wt_mask, 0, np.where(meta["gse"].isin(test_studies), 3, 1))
    out["cv_fold_base"] = kfold_by_study(meta, wt_mask & (out["split_base"].values == 1), cfg.n_folds,
                                         seed=cfg.seed, n_candidates=cfg.n_candidates, balance_vars=BALANCE_VARS,
                                         study_balance=cfg.fold_study_weight, forced_fold_studies=forced_fold, log=log)
    return out


def finalize_splits(base: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """From ``split_base`` / ``cv_fold_base`` derive ``split`` / ``cv_fold`` and the z-score fit mask."""
    out = base.copy()
    is_poc = out["gse"].astype(str).values == str(cfg.withheld_study) if cfg.withheld_study else np.zeros(len(out), bool)
    out["split"] = out["split_base"].astype(int)
    out["cv_fold"] = out["cv_fold_base"].astype(float)
    zfit = (out["split_base"].values == 1) & ~is_poc
    out.loc[is_poc, "split"] = 0
    out.loc[is_poc, "cv_fold"] = np.nan
    test_studies = set(out.loc[out["split_base"] == 3, "gse"].astype(str))
    held = test_studies | set(cfg.case_studies)
    add = (out["split"].values == 0) & ~is_poc & ~out["gse"].astype(str).isin(held).values
    out.loc[add, "split"] = 1
    return out, zfit


def load_splits(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    missing = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: splits file lacks columns {missing}")
    return df[SPLIT_COLUMNS]


def check_no_leakage(meta: pd.DataFrame, zfit: np.ndarray, cfg: TrainConfig) -> None:
    acc = meta.index.to_series().str.extract(r"(GSM\d+)")[0]
    train = meta["split"] == 1
    test = meta["split"] == 3
    train_gses = set(meta.loc[train, "gse"].astype(str))
    test_gses = set(meta.loc[test, "gse"].astype(str))
    problems = []
    if set(acc[train].dropna()) & set(acc[test].dropna()):
        problems.append("a GSM accession has rows in both TRAIN and TEST")
    if train_gses & test_gses:
        problems.append("a study is in both TRAIN and TEST")
    if int((zfit & meta["gse"].astype(str).isin(test_gses).values).sum()):
        problems.append("z-score statistics would use TEST studies")
    if cfg.withheld_study and int(((meta["split"] != 0) & (meta["gse"].astype(str) == cfg.withheld_study)).sum()):
        problems.append("withheld study reappeared in a split")
    if set(cfg.case_studies) & train_gses:
        problems.append("a case-study GSE is in TRAIN")
    if problems:
        raise ValueError("split leakage: " + "; ".join(problems))


def bucket_labels(meta: pd.DataFrame) -> pd.Series:
    lab = pd.Series(np.nan, index=meta.index, dtype="object")
    lab[meta["split"] == 3] = "test"
    pool = (meta["split"] == 1).values
    fold = meta["cv_fold"]
    lab[pool & fold.notna().values] = "fold" + fold[pool & fold.notna().values].astype(int).astype(str)
    lab[pool & fold.isna().values] = "train(aug)"
    return lab
