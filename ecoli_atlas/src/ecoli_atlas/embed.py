"""`ecoli_atlas embed`: place new samples in the atlas embedding with a trained bundle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .artifacts import ModelBundle
from .config import HEADS
from .data import _qc_value, check_qc, qc_covariates, covariate_frame, dna_density, dna_reads, gc_cleaned_counts
from .preprocess import align_counts, call_groups, group_calls, log_expression, missing_genes, size_factors
from .technical import GCProfile, feature_gc_for, load_feature_gc, technical_counts
from .train import encode


GC_FILLS = ("study", "atlas")


def fill_gc_values(qc: pd.DataFrame, ids: list, how: str, scaler, studies=None) -> tuple:
    """Fill the GC bias (gc_slope, gc_curv) of samples whose QC record has none (``embed --fill-missing-gc``): ``study``
    with the median of the samples of the same study that have it (``studies``; without it the whole table is one
    study), ``atlas`` with the training mean of the bundle. Returns (QC table with the values filled, Series sample ->
    how it was filled, '' for the others). A library too shallow for the GC fit (inGEST gc_bias.py: 50 genes at depth 5)
    is usually also below the depth gate (``low_depth``), outside the depths the model was trained on."""
    if how not in GC_FILLS:
        raise ValueError(f"--fill-missing-gc must be one of {GC_FILLS}, not {how!r}")
    cov = qc_covariates(qc, ids)
    lacking = [i for i in ids if pd.isna(cov.at[i, "gc_slope"]) or pd.isna(cov.at[i, "gc_curv"])]
    filled = pd.Series("", index=ids, dtype=object)
    if not lacking:
        return qc, filled
    out = qc.copy()
    study = pd.Series("all", index=ids, dtype=object) if studies is None else pd.Series(studies).reindex(ids).fillna("all").astype(object)
    for name in ("gc_slope", "gc_curv"):
        col = "full_" + name if "full_" + name in out.columns else name
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].astype(object)
        for i in lacking:
            if how == "atlas":
                value = float(scaler.mu[scaler.numeric.index(name)])
            else:
                peers = [j for j in ids if study[j] == study[i] and j not in lacking]
                if not peers:
                    raise ValueError(f"--fill-missing-gc study: no sample of the study of {i} ({study[i]}) has a GC fit; "
                                     "use --fill-missing-gc atlas")
                value = float(cov.loc[peers, name].median())
            out.at[i, col] = f"{value:.6g}"
    filled[lacking] = how
    return out, filled


def bundle_feature_gc(bundle: ModelBundle, table: pd.DataFrame | None = None) -> GCProfile:
    """GC content (and window profile) of the bundle's features for its cleaned inputs: from its GC term when it has
    one, else from ``table`` (default: the bundle's ``feature_gc`` table or the package table,
    ``technical.load_feature_gc``), which must give the profile the bundle was trained with."""
    gt = getattr(bundle.model, "gc_term", None)
    if gt is not None:
        return GCProfile(gt.x.cpu().numpy().astype(np.float64) / 10 + 0.5, gt.P.cpu().numpy().astype(np.float64) if gt.n_bins else None)
    prof = feature_gc_for(bundle.fs, load_feature_gc(bundle.config.feature_gc) if table is None else table)
    want = (bundle.manifest.get("encoder_inputs") or {}).get("gc_profile") or "gene"
    if prof.kind != want:
        raise ValueError(f"the bundle's inputs were cleaned with a {want!r} GC profile; the feature GC table gives {prof.kind!r}")
    return prof


def embed_counts(bundle: ModelBundle, counts: pd.DataFrame, qc: pd.DataFrame, device="cpu",
                 min_coverage: float = 0.9, antisense: pd.DataFrame | None = None, studies=None, pool: bool = True,
                 products: tuple = (), feature_gc: pd.DataFrame | None = None, fill_missing_gc: str | None = None) -> dict:
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
    bundle's limit ``max_dna_fraction``, the reads on CDS features and whether they are below the
    bundle's ``min_cds_reads``, the atlas's depth gate (``low_depth``)), the feature ``coverage``, ``missing`` (one row per sample and gene called missing from its
    strain; such genes enter the encoders as the gene's mean) and ``warnings`` (QC values set to
    the training mean).
    products: ``"corrected"`` and / or ``"decoded"``: counts with the technical part swapped for that of a
    reference library (``technical.py``), returned as ``corrected_counts`` / ``decoded_counts`` (features x
    samples); ``samples`` then has ``gc_offset`` (1 where the sample's GC bias was taken off).
    ``fill_missing_gc``: ``study`` or ``atlas``: samples without a GC bias fit get one instead of
    being refused (``fill_gc_values``); ``samples`` then says which (``gc_filled``).
    ``feature_gc``: GC content of the features (``technical.load_feature_gc``; default the package table for
    the inGEST counting annotation).
    """
    ids = list(counts.columns)
    Y, fam, coverage = align_counts(counts, bundle.fs)
    if coverage.fraction < min_coverage:
        raise ValueError(f"only {coverage.n_present} of {coverage.n_model_features} model features are in the count table "
                         f"({coverage.fraction:.1%}); the table must use the inGEST feature ids (e.g. CDS_b0002_thrA)")
    gc_filled = pd.Series("", index=ids, dtype=object)
    if fill_missing_gc:
        qc, gc_filled = fill_gc_values(qc, ids, fill_missing_gc, bundle.scaler, studies)
    warnings = check_qc(qc, ids, bundle.scaler.names)
    if (gc_filled != "").any():
        warnings = warnings + [f"{int((gc_filled != '').sum())} sample(s) have no GC bias fit; filled with the {fill_missing_gc} "
                               f"{'median' if fill_missing_gc == 'study' else 'training mean'} (gc_filled in samples.tsv): "
                               + ", ".join(gc_filled.index[gc_filled != ""][:10])]
    cfg = bundle.config
    # genomic DNA, when the bundle was trained with corrected inputs: the expected DNA reads (from the QC table) come off the counts
    rho, dlen = dna_density(qc, ids, bundle.fs.genome_length) if cfg.dna_inputs else (np.zeros(len(ids)), np.zeros(len(ids)))
    Yin = np.maximum(Y - dna_reads(rho, dlen, bundle.fs.lengths), 0.0) if cfg.dna_inputs else Y
    cov = covariate_frame(Y, fam, bundle.fs, qc, ids)
    # GC-cleaned counts: the encoder inputs (cfg.clean_inputs) and the size factors (cfg.gc_size_factors), as in training
    gc_sf = cfg.gc_size_factors and (cfg.clean_inputs or cfg.gc_offset)
    Ygc = gc_cleaned_counts(Yin, bundle_feature_gc(bundle, feature_gc), cov, bundle.fs)[0] if (cfg.clean_inputs or gc_sf) else None
    Ysf = Ygc if gc_sf else Yin
    s = size_factors(Ysf, bundle.ref_log)
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
        s = size_factors(Ysf, bundle.ref_log, miss)
    Xin = Ygc if cfg.clean_inputs else Yin
    X = torch.tensor(bundle.zscore.apply(log_expression(Xin, s), miss))
    C = torch.tensor(bundle.scaler.apply(cov))
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
    # reads on CDS features; below the atlas's depth gate the sample is outside the depths the model was trained on
    cds = np.array([f.startswith("CDS_") for f in bundle.fs.ids])
    info["cds_reads"] = Y[:, cds].sum(axis=1).round(0).astype(np.int64)
    info["low_depth"] = (info["cds_reads"] < cfg.min_cds_reads).astype(int)
    info["gc_filled"] = gc_filled.reindex(ids).to_numpy()
    rows, cols = np.nonzero(miss)
    missing = pd.DataFrame({"sample": np.array(ids, dtype=object)[rows], "feature": np.array(bundle.fs.ids, dtype=object)[cols]})
    out = {}
    if products:
        unknown = set(products) - {"corrected", "decoded"}
        if unknown:
            raise ValueError(f"unknown count products {sorted(unknown)}")
        own = hasattr(bundle.model, "gc_term")            # a bundle with the GC term in its decoder needs no feature table
        gc = None if own else feature_gc_for(bundle.fs, load_feature_gc() if feature_gc is None else feature_gc)
        q = qc.reindex(ids).fillna("")
        slope = pd.to_numeric(_qc_value(q, "gc_slope"), errors="coerce").to_numpy(dtype=np.float64)
        curv = pd.to_numeric(_qc_value(q, "gc_curv"), errors="coerce").to_numpy(dtype=np.float64)
        has_gc = np.isfinite(slope) & np.isfinite(curv)
        if not has_gc.all():
            warnings = warnings + [f"{int((~has_gc).sum())} sample(s) have no GC bias fit in the QC table; their counts keep "
                                   f"their GC bias: {', '.join(np.array(ids)[~has_gc][:5])}" + (" ..." if (~has_gc).sum() > 5 else "")]
        res = technical_counts(bundle.model, bundle.fs, bundle.scaler, Yin, C.numpy(), enc["z"], np.log(s), miss, gc, slope, curv,
                               corrected="corrected" in products, decoded="decoded" in products)
        for k, v in res.items():
            frame = pd.DataFrame(v.T, index=bundle.fs.ids, columns=ids)
            frame.index.name = "Feature_ID"
            out[f"{k}_counts"] = frame
        info["gc_offset"] = has_gc.astype(int)
    for df in (emb, pred, info):
        df.index.name = "sample"
    return {"embedding": emb, "predictions": pred, "samples": info, "coverage": coverage, "missing": missing,
            "warnings": warnings} | out
