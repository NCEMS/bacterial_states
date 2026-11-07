"""Gene filtering and renaming, CPM, z-scoring, and alignment to the model's gene set.

Mirrors the training notebook exactly: drop rRNA/tRNA (matched on locus tag or symbol), rename
locus tags to symbols only when a symbol maps to a single locus, per-sample CPM, then a per-gene
z-score with statistics fit on the training pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GeneMap:
    rna_ids: set = field(default_factory=set)            # rRNA/tRNA locus tags and symbols
    locus_to_symbol: dict = field(default_factory=dict)  # unambiguous locus_tag -> symbol
    n_ambiguous: int = 0

    @classmethod
    def from_feature_table(cls, feat: pd.DataFrame) -> "GeneMap":
        rt = feat[feat["class"].isin(["rRNA", "tRNA"])]
        rna_ids = set(rt["locus_tag"].dropna().astype(str)) | set(rt["symbol"].dropna().astype(str))
        keep = (feat[~feat["class"].isin(["rRNA", "tRNA"])]
                .dropna(subset=["locus_tag", "symbol"])
                .drop_duplicates(subset=["locus_tag", "symbol"]))
        sym_nloci = keep.groupby("symbol")["locus_tag"].nunique()
        safe = keep[keep["symbol"].map(sym_nloci) == 1]
        return cls(rna_ids=rna_ids,
                   locus_to_symbol=dict(zip(safe["locus_tag"].astype(str), safe["symbol"].astype(str))),
                   n_ambiguous=int((sym_nloci > 1).sum()))


def clean_counts(counts: pd.DataFrame, genemap: GeneMap) -> pd.DataFrame:
    """genes x samples -> samples x genes with rRNA/tRNA removed and locus tags renamed to symbols."""
    X = counts.T.copy()
    X = X.drop(columns=[c for c in X.columns if str(c) in genemap.rna_ids], errors="ignore")
    X = X.rename(columns=lambda c: genemap.locus_to_symbol.get(str(c), str(c)))
    X = X.loc[:, ~X.columns.duplicated()]
    X.columns = X.columns.astype(str)
    return X


def cpm(X: pd.DataFrame) -> pd.DataFrame:
    lib = X.sum(axis=1)
    lib = lib.where(lib > 0, 1.0)
    return X.div(lib, axis=0) * 1e6


def cpm_array(X: np.ndarray) -> np.ndarray:
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return X / lib * 1e6


@dataclass
class CoverageReport:
    n_model_genes: int
    n_present: int
    missing: list

    @property
    def fraction(self) -> float:
        return self.n_present / self.n_model_genes if self.n_model_genes else 0.0

    def to_frame(self, genes: list) -> pd.DataFrame:
        missing = set(self.missing)
        return pd.DataFrame({"gene": genes, "present": [g not in missing for g in genes]}).set_index("gene")


def align_to_genes(X: pd.DataFrame, genes: list) -> tuple[pd.DataFrame, CoverageReport]:
    """Reindex samples x genes to the model's gene order; absent genes are filled with 0 counts."""
    present = [g for g in genes if g in X.columns]
    aligned = X.reindex(columns=genes, fill_value=0).astype(np.float64)
    missing = [g for g in genes if g not in X.columns]
    return aligned, CoverageReport(len(genes), len(present), missing)


@dataclass
class ZScore:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, cpm_values: np.ndarray, rows: np.ndarray | None = None) -> "ZScore":
        """Per-gene mean and (population) sd of CPM over the given rows, i.e. the training pool.
        Genes with zero variance get sd = 1 so they contribute a constant, as in the notebook."""
        X = np.asarray(cpm_values, dtype=np.float64)
        if rows is not None:
            X = X[rows]
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        return cls(mu.astype(np.float32), sd.astype(np.float32))

    def apply(self, cpm_values: np.ndarray) -> np.ndarray:
        return ((cpm_values - self.mu) / self.sd).astype(np.float32)
