"""`ecoli_atlas embed`: place new samples in the atlas embedding with a trained bundle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .artifacts import ModelBundle
from .config import HEADS
from .data import _qc_value, check_qc, covariate_frame, dna_density, dna_reads
from .preprocess import align_counts, call_groups, group_calls, log_expression, missing_genes, size_factors
from .train import encode


def embed_counts(bundle: ModelBundle, counts: pd.DataFrame, qc: pd.DataFrame, device="cpu",
                 min_coverage: float = 0.9, antisense: pd.DataFrame | None = None, studies=None, pool: bool = True) -> dict:
    """counts: features x samples (inGEST feature ids). qc: table indexed by sample with the QC values
    (``qc_record.read_qc_records`` of an inGEST run, or the ``full_*`` columns of ``sample_qc.tsv``);
    required, and every sample must have the values ``data.check_qc`` asks for.
    antisense: optional antisense count table of the same samples (inGEST's
    ``all_samples_antisense_counts.txt``); with it a silent gene that has antisense reads is not
    called missing.
    studies: optional study of every sample (a Series indexed by sample, or a mapping); without it
    all samples of the table count as one study. With a bundle trained with pooled calls, the
    samples of a study that lack the same genes are also called together on their summed counts,
    so a sample's calls can depend on the samples embedded with it; ``pool=False`` calls every
    sample on its own.

    Returns frames indexed by sample: ``embedding`` (z1..zN), ``predictions`` (probability and call
    of each head), ``samples`` (size factor, reads on mRNA features, number of genes called missing,
    reads taken off the mRNA features as genomic DNA, the DNA estimate and whether it is above the
    bundle's limit ``max_dna_fraction``),
    the feature ``coverage``, ``missing`` (one row per sample and gene called missing from its
    strain; such genes enter the encoders as the gene's mean) and ``warnings`` (QC values set to
    the training mean).
    """
    ids = list(counts.columns)
    Y, fam, coverage = align_counts(counts, bundle.fs)
    if coverage.fraction < min_coverage:
        raise ValueError(f"only {coverage.n_present} of {coverage.n_model_features} model features are in the count table "
                         f"({coverage.fraction:.1%}); the table must use the inGEST feature ids (e.g. CDS_b0002_thrA)")
    warnings = check_qc(qc, ids, bundle.scaler.names)
    cfg = bundle.config
    # genomic DNA, when the bundle was trained with corrected inputs: the expected DNA reads (from the QC table) come off the counts
    rho, dlen = dna_density(qc, ids, bundle.fs.genome_length) if cfg.dna_inputs else (np.zeros(len(ids)), np.zeros(len(ids)))
    Yin = np.maximum(Y - dna_reads(rho, dlen, bundle.fs.lengths), 0.0) if cfg.dna_inputs else Y
    s = size_factors(Yin, bundle.ref_log)
    miss = np.zeros(Y.shape, dtype=bool)
    if bundle.low is not None and cfg.mask_missing and bundle.fs.order is not None:
        veto = {}
        if antisense is not None:
            lacking = sorted(set(ids) - set(antisense.columns))
            if lacking:
                raise ValueError(f"{len(lacking)} samples are not in the antisense count table: {lacking[:5]}")
            veto = {"antisense": antisense[ids].reindex(bundle.fs.ids, fill_value=0).to_numpy(dtype=np.float64).T, "lengths": bundle.fs.lengths,
                    "veto_reads": cfg.mask_veto_reads, "veto_fraction": cfg.mask_veto_fraction}
        miss = missing_genes(Y, s, bundle.low, bundle.fs.order, cfg.mask_min_expected, cfg.mask_low_fraction, **veto)
        if pool and cfg.mask_pool and bundle.low_pool is not None and len(ids) > 1:
            study = np.array(["all"] * len(ids), dtype=object) if studies is None else pd.Series(studies).reindex(ids).to_numpy(dtype=object)
            if pd.isna(study).any():
                raise ValueError(f"no study for {int(pd.isna(study).sum())} samples, e.g. {[i for i, x in zip(ids, pd.isna(study)) if x][:5]}")
            groups = call_groups(miss, study, cfg.mask_pool_min_calls, cfg.mask_pool_jaccard)
            miss = miss | group_calls(Y, s, groups, bundle.low_pool, bundle.fs.order, min_expected=cfg.mask_min_expected,
                                      low_fraction=cfg.mask_low_fraction, **veto)
        s = size_factors(Yin, bundle.ref_log, miss)
    X = torch.tensor(bundle.zscore.apply(log_expression(Yin, s), miss))
    C = torch.tensor(bundle.scaler.apply(covariate_frame(Y, fam, bundle.fs, qc, ids)))
    enc = encode(bundle.model, X, C, device)
    emb = pd.DataFrame(enc["z"], index=ids, columns=[f"z{i + 1}" for i in range(enc["z"].shape[1])])
    pred = pd.DataFrame(index=ids)
    for h, (neg, pos) in HEADS.items():
        pred[f"p_{pos}"] = enc["probs"][h]
        pred[h] = np.where(enc["probs"][h] >= 0.5, pos, neg)
    info = pd.DataFrame({"size_factor": s, "mrna_reads": Y[:, bundle.fs.is_mrna].sum(axis=1).astype(np.int64),
                         "n_missing_genes": miss.sum(axis=1),
                         "dna_reads_removed": (Y - Yin)[:, bundle.fs.is_mrna].sum(axis=1).round(0).astype(np.int64)}, index=ids)
    # the genomic-DNA estimate of the QC table; above the bundle's limit the sample would not have been part of the atlas
    dna = pd.to_numeric(_qc_value(qc.reindex(ids).fillna(""), "dna_fraction_est"), errors="coerce").to_numpy()
    info["dna_fraction_est"] = dna
    info["high_dna"] = (np.nan_to_num(dna) > cfg.max_dna_fraction).astype(int)
    rows, cols = np.nonzero(miss)
    missing = pd.DataFrame({"sample": np.array(ids, dtype=object)[rows], "feature": np.array(bundle.fs.ids, dtype=object)[cols]})
    for df in (emb, pred, info):
        df.index.name = "sample"
    return {"embedding": emb, "predictions": pred, "samples": info, "coverage": coverage, "missing": missing,
            "warnings": warnings}
