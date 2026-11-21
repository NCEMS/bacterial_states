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


def _called_runs(depleted: np.ndarray, e: np.ndarray, order: np.ndarray, min_expected: float) -> np.ndarray:
    """Runs of depleted genes in chromosome order whose genes together expected ``min_expected`` reads."""
    depleted, e = depleted[:, order], e[:, order]
    out = np.zeros_like(depleted)
    for i in range(depleted.shape[0]):
        d = depleted[i]
        if not d.any():
            continue
        run = np.cumsum(~d)                                   # constant within a run of depleted genes
        total = np.bincount(run[d], weights=e[i][d], minlength=int(run[-1]) + 1)
        out[i, d] = total[run[d]] >= min_expected
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return out[:, back]


def background_rate(antisense: np.ndarray, lengths: np.ndarray, low: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """Per sample, the median number of antisense reads per kb over the weakly expressed quarter of
    the genes (by ``low``) that are not called missing: what a present gene collects on the strand
    it is not transcribed from."""
    weak = np.zeros(len(low), dtype=bool)
    weak[np.argsort(np.nan_to_num(low), kind="stable")[: max(len(low) // 4, 1)]] = True
    per_kb = antisense[:, weak] / (lengths[weak] / 1000.0)[None, :]
    if missing is not None:
        per_kb = np.where(missing[:, weak], np.nan, per_kb)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(per_kb, axis=1))


def missing_genes(Y: np.ndarray, s: np.ndarray, low: np.ndarray, order: np.ndarray, min_expected: float = 10.0,
                  low_fraction: float = 0.02, antisense: np.ndarray | None = None, lengths: np.ndarray | None = None,
                  veto_reads: float = 2.0, veto_fraction: float = 0.25) -> np.ndarray:
    """Genes without reads because the strain does not have them (samples x features, bool).

    ``low`` is the low end of every gene's normalised count among samples that have the gene
    (``low_expression_profile``), so ``s * low`` reads are expected even where the gene is weakly
    expressed. A gene is depleted in a sample when its count is at most ``low_fraction`` of that
    (zero when the expectation is small). Neighbouring depleted genes in chromosome order
    (``order``) form a run, and a run is called missing when its genes together expected at least
    ``min_expected`` reads. A deleted island is so called with its weakly expressed genes, a single
    gene only when it is well expressed, and a shallow sample gives no call.

    With ``antisense`` counts (samples x features, and the gene ``lengths``) a gene that has reads
    on the other strand is present, however silent: it is not depleted when its antisense count is
    above ``veto_reads`` and above ``veto_fraction`` of the sample's ``background_rate`` for a gene
    of its length. Antisense reads only ever remove calls."""
    e = s[:, None] * np.nan_to_num(low)[None, :]
    depleted = Y <= low_fraction * e
    if antisense is not None:
        if lengths is None:
            raise ValueError("the antisense veto needs the gene lengths")
        rate = background_rate(antisense, lengths, low, _called_runs(depleted, e, order, min_expected))
        depleted &= antisense <= np.maximum(veto_reads, veto_fraction * rate[:, None] * (lengths / 1000.0)[None, :])
    return _called_runs(depleted, e, order, min_expected)


def low_expression_profile(Y: np.ndarray, s: np.ndarray, order: np.ndarray, min_expected: float = 10.0, low_fraction: float = 0.02,
                           quantile: float = 0.25, start_quantile: float = 0.25, rounds: int = 2, **veto) -> np.ndarray:
    """Per gene, the ``quantile`` of the normalised count over the samples (rows of ``Y``) in which
    the gene is not called missing. The calls need the profile, so it is found in rounds: the
    first calls use ``start_quantile`` over all samples, which is above zero as long as fewer
    than that share of the samples lack the gene. ``veto``: the antisense arguments of ``missing_genes``."""
    N = Y / s[:, None]
    low = np.quantile(N, start_quantile, axis=0)
    for _ in range(rounds):
        miss = missing_genes(Y, s, low, order, min_expected, low_fraction, **veto)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nan_to_num(np.nanquantile(np.where(miss, np.nan, N), quantile, axis=0))
    return low


def call_groups(missing: np.ndarray, study: np.ndarray, min_calls: int = 20, jaccard: float = 0.5) -> np.ndarray:
    """Group index per sample: the samples of one study that lack the same genes, found from the
    per-sample calls alone. Within a study, samples with fewer than ``min_calls`` calls form one
    group; the others are joined when their call sets overlap (intersection over union at least
    ``jaccard``, single linkage). A study with two strains so falls into two groups."""
    study = np.asarray(study).astype(str)
    size = missing.sum(axis=1)
    groups = np.zeros(len(study), dtype=int)
    nxt = 0
    for g in pd.unique(study):
        r = np.flatnonzero(study == g)
        lab = np.zeros(len(r), dtype=int)
        big = np.flatnonzero(size[r] >= min_calls)
        if len(big):
            X = missing[r[big]].astype(np.float32)
            inter = X @ X.T
            union = size[r[big]][:, None] + size[r[big]][None, :] - inter
            adj = inter / np.maximum(union, 1) >= jaccard
            comp = np.full(len(big), -1)
            c = 0
            for i in range(len(big)):
                if comp[i] >= 0:
                    continue
                stack, comp[i] = [i], c
                while stack:
                    j = stack.pop()
                    for k in np.flatnonzero(adj[j] & (comp < 0)):
                        comp[k] = c
                        stack.append(k)
                c += 1
            lab[big] = comp + 1
        _, lab = np.unique(lab, return_inverse=True)
        groups[r] = lab + nxt
        nxt += int(lab.max()) + 1
    return groups


def pool_groups(groups: np.ndarray, Y: np.ndarray, s: np.ndarray, antisense: np.ndarray | None = None) -> tuple:
    """(summed counts, summed size factors, summed antisense counts or None, samples per group), one row per group."""
    n = int(groups.max()) + 1 if len(groups) else 0
    Yg, Sg = np.zeros((n, Y.shape[1])), np.zeros(n)
    np.add.at(Yg, groups, Y)
    np.add.at(Sg, groups, s)
    Ag = None
    if antisense is not None:
        Ag = np.zeros((n, Y.shape[1]))
        np.add.at(Ag, groups, antisense)
    return Yg, Sg, Ag, np.bincount(groups, minlength=n)


def pooled_low_profile(Y: np.ndarray, s: np.ndarray, groups: np.ndarray, order: np.ndarray, antisense: np.ndarray | None = None,
                       **kw) -> np.ndarray:
    """``low_expression_profile`` of the pooled groups with two or more samples. Summed over a
    group, a weakly expressed gene has reads where a single sample has none, so nearly every gene
    gets a low end above zero. The first round starts from the median over the groups, so that a
    gene may be missing from up to half of them. ``kw``: the arguments of ``low_expression_profile``."""
    kw = {"start_quantile": 0.5} | kw
    Yg, Sg, Ag, n = pool_groups(groups, Y, s, antisense)
    use = n >= 2
    if use.sum() < 4:
        raise ValueError("fewer than 4 groups of two or more samples; no pooled profile can be made")
    if Ag is not None:
        kw = kw | {"antisense": Ag[use]}
    return low_expression_profile(Yg[use], Sg[use], order, **kw)


def group_calls(Y: np.ndarray, s: np.ndarray, groups: np.ndarray, low_pool: np.ndarray, order: np.ndarray,
                antisense: np.ndarray | None = None, **kw) -> np.ndarray:
    """Missing-gene calls of every sample's group (samples x features): the group's counts are
    summed and called like one deep sample against ``low_pool``. A sample alone in its group gets
    none. ``kw``: the arguments of ``missing_genes``."""
    Yg, Sg, Ag, n = pool_groups(groups, Y, s, antisense)
    if Ag is not None:
        kw = kw | {"antisense": Ag}
    Mg = missing_genes(Yg, Sg, low_pool, order, **kw)
    Mg[n < 2] = False
    return Mg[groups]


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
