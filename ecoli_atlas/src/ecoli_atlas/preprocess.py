"""From a count table to model inputs: size factors, log expression, z-scores, composition."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import COMPOSITION, FeatureSet


@dataclass
class CoverageReport:
    n_model_features: int
    n_present: int
    missing: list

    @property
    def fraction(self) -> float:
        return self.n_present / self.n_model_features if self.n_model_features else 0.0

    def to_frame(self, ids: list) -> pd.DataFrame:
        missing = set(self.missing)
        return pd.DataFrame({"feature": ids, "present": [g not in missing for g in ids]}).set_index("feature")


def align_counts(counts: pd.DataFrame, fs: FeatureSet) -> tuple[np.ndarray, np.ndarray, CoverageReport]:
    """features x samples table -> (samples x model features counts, samples x 4 depleted-family
    counts in the order of COMPOSITION, coverage). Features the table lacks are 0."""
    missing = [g for g in fs.ids if g not in counts.index]
    Y = counts.reindex(fs.ids, fill_value=0).to_numpy(dtype=np.float64).T
    fam = np.stack([counts.reindex(fs.families[k], fill_value=0).to_numpy(dtype=np.float64).sum(axis=0)
                    for k in COMPOSITION], axis=1)
    return Y, fam, CoverageReport(fs.n, fs.n - len(missing), missing)


def weighted_stats(X: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and (population) sd, optionally weighted by sample."""
    X = np.asarray(X, dtype=np.float64)
    if w is None:
        return X.mean(axis=0), X.std(axis=0)
    w = np.asarray(w, dtype=np.float64) / np.sum(w)
    mu = w @ X
    return mu, np.sqrt(np.maximum(w @ (X - mu) ** 2, 0.0))


def reference_profile(Y: np.ndarray, is_mrna: np.ndarray, w: np.ndarray | None = None, min_presence: float = 0.5) -> np.ndarray:
    """Per-gene mean of log counts over the samples in which the gene has reads; NaN for genes
    that are not mRNA or have reads in fewer than ``min_presence`` of the samples."""
    pos = Y > 0
    wt = np.ones(len(Y)) if w is None else np.asarray(w, dtype=np.float64)
    weight = wt[:, None] * pos
    total = weight.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = (weight * np.log(np.where(pos, Y, 1.0))).sum(axis=0) / total
    keep = is_mrna & (total / wt.sum() >= min_presence)
    return np.where(keep, ref, np.nan)


def size_factors(Y: np.ndarray, ref_log: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """Median over reference genes with reads of count / reference (median-of-ratios). Genes
    called missing in a sample (``missing``, samples x features) are left out for that sample."""
    use = ~np.isnan(ref_log)
    has = Y[:, use] > 0 if missing is None else (Y[:, use] > 0) & ~missing[:, use]
    with np.errstate(divide="ignore"):
        ratio = np.where(has, np.log(np.where(has, Y[:, use], 1.0)) - ref_log[use], np.nan)
    if np.isnan(ratio).all(axis=1).any():
        raise ValueError("a sample has no reads on any reference gene; no size factor can be given")
    return np.exp(np.nanmedian(ratio, axis=1))


def log_expression(Y: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.log2(Y / s[:, None] + 1.0)


def missing_genes(Y: np.ndarray, s: np.ndarray, low: np.ndarray, order: np.ndarray, min_expected: float = 10.0,
                  low_fraction: float = 0.02) -> np.ndarray:
    """Genes without reads because the strain does not have them (samples x features, bool).

    ``low`` is the low end of every gene's normalised count among samples that have the gene
    (``low_expression_profile``), so ``s * low`` reads are expected even where the gene is weakly
    expressed. A gene is depleted in a sample when its count is at most ``low_fraction`` of that
    (zero when the expectation is small). Neighbouring depleted genes in chromosome order
    (``order``) form a run, and a run is called missing when its genes together expected at least
    ``min_expected`` reads. A deleted island is so called with its weakly expressed genes, a single
    gene only when it is well expressed, and a shallow sample gives no call."""
    e = s[:, None] * np.nan_to_num(low)[None, :]
    depleted = (Y <= low_fraction * e)[:, order]
    e = e[:, order]
    out = np.zeros_like(depleted)
    for i in range(Y.shape[0]):
        d = depleted[i]
        if not d.any():
            continue
        run = np.cumsum(~d)                                   # constant within a run of depleted genes
        total = np.bincount(run[d], weights=e[i][d], minlength=int(run[-1]) + 1)
        out[i, d] = total[run[d]] >= min_expected
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return out[:, back]


def low_expression_profile(Y: np.ndarray, s: np.ndarray, order: np.ndarray, min_expected: float = 10.0, low_fraction: float = 0.02,
                           quantile: float = 0.25, start_quantile: float = 0.25, rounds: int = 2) -> np.ndarray:
    """Per gene, the ``quantile`` of the normalised count over the samples (rows of ``Y``) in which
    the gene is not called missing. The calls need the profile, so it is found in rounds: the
    first calls use ``start_quantile`` over all samples, which is above zero as long as fewer
    than that share of the samples lack the gene."""
    N = Y / s[:, None]
    low = np.quantile(N, start_quantile, axis=0)
    for _ in range(rounds):
        miss = missing_genes(Y, s, low, order, min_expected, low_fraction)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nan_to_num(np.nanquantile(np.where(miss, np.nan, N), quantile, axis=0))
    return low


def composition_covariates(Y: np.ndarray, fam: np.ndarray, is_mrna: np.ndarray) -> np.ndarray:
    """Columns: log10 reads on mRNA features, then log10 ratio of 16S, 23S, 5S and tRNA reads to
    mRNA reads (half a read added to both)."""
    mrna = Y[:, is_mrna].sum(axis=1)
    return np.column_stack([np.log10(mrna + 0.5)] + [np.log10((fam[:, j] + 0.5) / (mrna + 0.5)) for j in range(fam.shape[1])])


@dataclass
class ZScore:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, w: np.ndarray | None = None, sd_floor: float = 0.0, missing: np.ndarray | None = None) -> "ZScore":
        """``missing`` (samples x features): entries left out of a gene's statistics."""
        if missing is None or not missing.any():
            mu, sd = weighted_stats(X, w)
        else:
            W = (np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64))[:, None] * ~missing
            total = W.sum(axis=0)
            safe = np.where(total > 0, total, 1.0)
            mu = (W * X).sum(axis=0) / safe
            sd = np.sqrt(np.maximum((W * (X - mu) ** 2).sum(axis=0) / safe, 0.0))
        sd = np.where(sd == 0, 1.0, np.maximum(sd, sd_floor))
        return cls(mu.astype(np.float32), sd.astype(np.float32))

    def apply(self, X: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
        """z-scores; a missing gene is set to 0, the mean of the samples that have it."""
        Z = ((X - self.mu) / self.sd).astype(np.float32)
        if missing is not None:
            Z[missing] = 0.0
        return Z
