"""Assemble everything training needs into one aligned object (replaces the notebook's globals)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import labels as L
from .config import TrainConfig
from .preprocess import GeneMap, ZScore, clean_counts, cpm
from .splits import check_no_leakage, finalize_splits, make_splits


@dataclass
class TrainingData:
    meta: pd.DataFrame            # harmonised labels + split columns, one row per sample
    genes: list
    cpm: np.ndarray               # samples x genes, float64
    X: torch.Tensor               # z-scored CPM, float32, on CPU
    zscore: ZScore
    zfit: np.ndarray              # rows used to fit the z-score
    gse_codes: np.ndarray         # int64 study code per row
    factors: dict                 # factor -> FactorCodes
    cond_id: np.ndarray
    is_annot: np.ndarray
    ctrl_cond: np.ndarray
    is_ctrl: np.ndarray
    genemap: GeneMap

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    @property
    def tr_rows(self) -> np.ndarray:
        return np.where((self.meta["split"] == 1).values)[0]

    @property
    def te_rows(self) -> np.ndarray:
        return np.where((self.meta["split"] == 3).values)[0]

    def fit_rows(self, k: int) -> np.ndarray:
        """Training pool minus fold k; rows with no fold belong to every fit set."""
        m = (self.meta["split"] == 1).values & (self.meta["cv_fold"].isna().values | (self.meta["cv_fold"] != k).values)
        return np.where(m)[0]

    def val_rows(self, k: int) -> np.ndarray:
        return np.where((self.meta["split"] == 1).values & (self.meta["cv_fold"] == k).values)[0]

    def labels_tensor(self, factor: str) -> torch.Tensor:
        return torch.as_tensor(self.factors[factor].codes, dtype=torch.long)

    def splits_frame(self) -> pd.DataFrame:
        return self.meta[["gse", "split_base", "cv_fold_base", "split", "cv_fold"]].copy()


def build_training_data(counts: pd.DataFrame, meta_raw: pd.DataFrame, feat: pd.DataFrame, cfg: TrainConfig,
                        splits: pd.DataFrame | None = None, max_samples: int | None = None, log=None) -> TrainingData:
    """counts: genes x samples; meta_raw: metadata_revised.csv layout; feat: NCBI feature table."""
    log = log or (lambda *_: None)
    genemap = GeneMap.from_feature_table(feat)
    X = clean_counts(counts, genemap)
    meta = meta_raw.copy()
    meta.index = meta.index.astype(str).str.strip()
    X.index = X.index.astype(str).str.strip()
    common = sorted(X.index.intersection(meta.index))
    if not common:
        raise ValueError("no sample ids shared between the count table and the metadata")
    if max_samples is not None and max_samples < len(common):
        rng = np.random.RandomState(cfg.seed)
        common = sorted(rng.choice(common, max_samples, replace=False))
    X, meta = X.loc[common], meta.loc[common].copy()
    log(f"aligned: {len(common)} samples x {X.shape[1]} genes "
        f"(dropped rRNA/tRNA; {len(genemap.locus_to_symbol)} locus tags renamed, {genemap.n_ambiguous} ambiguous skipped)")

    cpm_df = cpm(X)
    genes = [str(g) for g in X.columns]
    lab = L.build_labels(meta, cfg.min_class_n)

    if splits is None:
        base = make_splits(lab, cfg, log=log)
    else:
        base = splits.copy()
        missing = [s for s in lab.index if s not in base.index]
        if missing:
            raise ValueError(f"splits file lacks {len(missing)} of the aligned samples, e.g. {missing[:3]}")
        base = base.loc[lab.index]
        base["gse"] = lab["gse"].values
    if "split" in base.columns and "cv_fold" in base.columns and splits is not None:
        final = base.copy()
        is_poc = final["gse"].astype(str).values == str(cfg.withheld_study) if cfg.withheld_study else np.zeros(len(final), bool)
        zfit = (final["split_base"].values == 1) & ~is_poc
    else:
        final, zfit = finalize_splits(base[["gse", "split_base", "cv_fold_base"]], cfg)
    for c in ("split_base", "cv_fold_base", "split", "cv_fold"):
        lab[c] = final[c].values
    check_no_leakage(lab, zfit, cfg)
    log(f"TEST {int((lab['split'] == 3).sum())} | training pool {int((lab['split'] == 1).sum())} | "
        f"projected only {int((lab['split'] == 0).sum())} | z-score fit on {int(zfit.sum())} samples")

    cpm_values = cpm_df.values.astype(np.float64)
    zs = ZScore.fit(cpm_values, zfit)
    Xz = torch.tensor(zs.apply(cpm_values), dtype=torch.float32)

    factors = {f: L.factor_codes(lab, f) for f in cfg.factors}
    for f, fc in factors.items():
        log(f"{f:15s} classes={fc.classes} labelled={int((fc.codes >= 0).sum())}")
    gse_codes = np.asarray(pd.Categorical(lab["gse"]).codes, dtype=np.int64)
    return TrainingData(meta=lab, genes=genes, cpm=cpm_values, X=Xz, zscore=zs, zfit=zfit, gse_codes=gse_codes,
                        factors=factors, cond_id=L.condition_ids(lab), is_annot=L.is_annotated(lab),
                        ctrl_cond=L.control_condition_ids(lab), is_ctrl=L.is_control(lab), genemap=genemap)
