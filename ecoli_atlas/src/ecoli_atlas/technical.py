"""Counts with the technical part of the library swapped for that of a reference library.

Two products (`embed --corrected-counts`, `--decoded-counts`), both features x samples on the scale of
each sample's own reads on mRNA features:

- **corrected counts**: the observed counts, with
  (1) the decoder's covariate term moved from the sample's covariates to those of the reference
      library: count x exp(W (c_ref - c)); the term is additive in the decoder's log rate, so this
      changes only what the model learned the covariates do;
  (2) the GC bias of the library taken off by its measured fit (inGEST ``gc_bias``: log coverage =
      b1 x + b2 x^2 inside genes, x = (GC - 0.5) * 10; ``gc_slope`` = b1 + 2 b2 mean(x), ``gc_curv`` = b2)
      applied to the GC content of every feature. The GC covariates stay at the sample's own values in
      (1), so the GC bias is not taken off twice.
  Every sample keeps its own detail; batch effects that the covariates do not describe stay.
- **decoded counts**: the decoder's expected counts at the sample's embedding, with the same swap: a
  denoised version. They lose real within-sample detail through the embedding (`work/batch_correction/`),
  so they are a summary, not corrected counts.

A bundle trained with ``--gc-offset`` has the GC bias in its decoder (``models.GCTerm``): the GC offset is then
that term, with its learned scale, and needs no feature table.

The reference library: paired-end, every other numeric covariate at the training mean (0 on the scaled
scale), other flags 0, no GC bias. Genes called missing from a strain stay at their observed count in
the corrected counts and are 0 in the decoded counts.
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd
import torch

GC_COLUMNS = ("gc_slope", "gc_curv")
FEATURE_GC_FILE = "feature_gc_counting_v2.tsv"
GC_FIT_MIN_LENGTH = 600          # the GC fit uses genes of at least this length (gc_bias.py); their mean x stands in for mean(x)


def load_feature_gc(path=None) -> pd.DataFrame:
    """Feature_ID -> Length, GC; the package table for the inGEST counting annotation unless ``path``."""
    if path is None:
        with resources.files("ecoli_atlas").joinpath("reference", FEATURE_GC_FILE).open() as fh:
            return pd.read_csv(fh, sep="\t", index_col="Feature_ID")
    return pd.read_csv(path, sep="\t", index_col="Feature_ID")


def feature_gc_for(fs, table: pd.DataFrame) -> np.ndarray:
    """GC content of the model features, checked against the bundle (same features, same lengths)."""
    lacking = [f for f in fs.ids if f not in table.index]
    if lacking:
        raise ValueError(f"the feature GC table lacks {len(lacking)} model features (e.g. {lacking[:3]}); "
                         "give --feature-gc with a table made for this annotation (ecoli_atlas/scripts/feature_gc.py)")
    t = table.loc[fs.ids]
    if "Length" in t.columns and not np.array_equal(t["Length"].to_numpy(dtype=np.int64), np.asarray(fs.lengths, dtype=np.int64)):
        raise ValueError("the feature GC table does not match the bundle's feature lengths: it was made for another annotation")
    return t["GC"].to_numpy(dtype=np.float64)


def reference_covariates(scaler) -> np.ndarray:
    """The reference library on the scaled covariate scale: paired-end, numeric values at the training mean
    (0), other flags 0, GC bias 0 (on its raw scale)."""
    ref = np.zeros(len(scaler.names))
    if "paired" in scaler.names:
        ref[scaler.names.index("paired")] = 1.0
    for n in GC_COLUMNS:
        if n in scaler.names and n in scaler.numeric:
            k = scaler.numeric.index(n)
            ref[scaler.names.index(n)] = (0.0 - scaler.mu[k]) / scaler.sd[k]
    return ref


def swapped_covariates(C: np.ndarray, scaler) -> np.ndarray:
    """Every sample's covariates replaced by the reference library's, except the GC bias (taken off by
    ``gc_offset`` instead)."""
    out = np.broadcast_to(reference_covariates(scaler), C.shape).copy()
    for n in GC_COLUMNS:
        if n in scaler.names:
            j = scaler.names.index(n)
            out[:, j] = C[:, j]
    return out


@torch.no_grad()
def covariate_shift(model, C: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """W (c_ref - c): the change of the linear covariate term of the decoder's log rate when the covariates go
    from C to C_ref (without the GC columns when the model has a GC term; ``model_gc_offset`` is that term)."""
    dt = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    keep = model.gc_term.keep if hasattr(model, "gc_term") else 1.0
    term = lambda c: model.cov_out(model.cov_in(torch.as_tensor(c, dtype=dt, device=dev) * keep)).cpu().numpy()   # noqa: E731
    return (term(C_ref) - term(C)).astype(np.float64)


@torch.no_grad()
def model_gc_offset(model, C: np.ndarray) -> np.ndarray:
    """The decoder's own GC term (``train --gc-offset``) at the samples' covariates, learned scale included."""
    p = next(model.parameters())
    return model.gc_term(torch.as_tensor(C, dtype=p.dtype, device=p.device)).cpu().numpy().astype(np.float64)


def gc_offset(gc: np.ndarray, slope: np.ndarray, curv: np.ndarray, is_mrna: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """samples x features: the log-coverage change from each sample's GC fit at every feature's GC content,
    centred over the mRNA features (a constant per sample goes into the size factor). 0 where the sample
    has no fit."""
    x = (np.asarray(gc, dtype=np.float64) - 0.5) * 10
    long = np.asarray(is_mrna) & (np.asarray(lengths) >= GC_FIT_MIN_LENGTH)
    xbar = x[long].mean() if long.any() else x.mean()
    slope, curv = np.asarray(slope, dtype=np.float64), np.asarray(curv, dtype=np.float64)
    ok = np.isfinite(slope) & np.isfinite(curv)
    b2 = np.where(ok, curv, 0.0)
    b1 = np.where(ok, slope - 2 * b2 * xbar, 0.0)
    off = b1[:, None] * x[None, :] + b2[:, None] * x[None, :] ** 2
    return off - off[:, np.asarray(is_mrna)].mean(axis=1, keepdims=True)


def _to_mrna_total(counts: np.ndarray, observed: np.ndarray, is_mrna: np.ndarray) -> np.ndarray:
    """Scale every sample to its observed reads on mRNA features."""
    have, want = counts[:, is_mrna].sum(axis=1), observed[:, is_mrna].sum(axis=1)
    return counts * np.where(have > 0, want / np.where(have > 0, have, 1.0), 1.0)[:, None]


def technical_counts(model, fs, scaler, Y: np.ndarray, C: np.ndarray, z: np.ndarray, log_s: np.ndarray, miss: np.ndarray,
                     gc: np.ndarray | None, slope: np.ndarray, curv: np.ndarray, corrected: bool = True, decoded: bool = False) -> dict:
    """``corrected`` / ``decoded`` counts (samples x features, see the module text). A model with its own GC term
    takes the GC bias off with that term (learned scale); otherwise with the measured fit at ``gc`` (None: no GC
    offset)."""
    is_mrna = np.asarray(fs.is_mrna)
    C_ref = swapped_covariates(C, scaler)
    shift = covariate_shift(model, C, C_ref)
    if hasattr(model, "gc_term"):
        shift = shift - model_gc_offset(model, C)
    elif gc is not None:
        shift = shift - gc_offset(gc, slope, curv, is_mrna, fs.lengths)
    out = {}
    if corrected:
        out["corrected"] = _to_mrna_total(Y * np.exp(shift), Y, is_mrna)
    if decoded:
        dt = next(model.parameters()).dtype
        dev = next(model.parameters()).device
        with torch.no_grad():
            rate = model.decode(torch.as_tensor(z, dtype=dt, device=dev), torch.as_tensor(C, dtype=dt, device=dev)).cpu().numpy()
        mean = np.exp(rate.astype(np.float64) + np.asarray(log_s, dtype=np.float64)[:, None] + shift)
        out["decoded"] = _to_mrna_total(np.where(miss, 0.0, mean), Y, is_mrna)
    return out
