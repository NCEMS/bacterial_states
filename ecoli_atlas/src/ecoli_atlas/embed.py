"""`ecoli_atlas embed`: place new samples in the atlas embedding with a trained bundle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .artifacts import ModelBundle
from .config import HEADS
from .data import covariate_frame
from .preprocess import align_counts, log_expression, size_factors
from .train import encode


def embed_counts(bundle: ModelBundle, counts: pd.DataFrame, qc: pd.DataFrame | None = None, device="cpu",
                 min_coverage: float = 0.9) -> dict:
    """counts: features x samples (inGEST feature ids). qc: optional table indexed by sample with the
    QC columns of the inGEST QC table; without it the QC covariates are set to the training means.

    Returns frames indexed by sample: ``embedding`` (z1..zN), ``predictions`` (probability and call
    of each head), ``samples`` (size factor, reads on mRNA features) and the feature ``coverage``.
    """
    ids = list(counts.columns)
    Y, fam, coverage = align_counts(counts, bundle.fs)
    if coverage.fraction < min_coverage:
        raise ValueError(f"only {coverage.n_present} of {coverage.n_model_features} model features are in the count table "
                         f"({coverage.fraction:.1%}); the table must use the inGEST feature ids (e.g. CDS_b0002_thrA)")
    s = size_factors(Y, bundle.ref_log)
    X = torch.tensor(bundle.zscore.apply(log_expression(Y, s)))
    C = torch.tensor(bundle.scaler.apply(covariate_frame(Y, fam, bundle.fs, qc, ids)))
    enc = encode(bundle.model, X, C, device)
    emb = pd.DataFrame(enc["z"], index=ids, columns=[f"z{i + 1}" for i in range(enc["z"].shape[1])])
    pred = pd.DataFrame(index=ids)
    for h, (neg, pos) in HEADS.items():
        pred[f"p_{pos}"] = enc["probs"][h]
        pred[h] = np.where(enc["probs"][h] >= 0.5, pos, neg)
    info = pd.DataFrame({"size_factor": s, "mrna_reads": Y[:, bundle.fs.is_mrna].sum(axis=1).astype(np.int64),
                         "has_qc": 0 if qc is None else [int(i in qc.index) for i in ids]}, index=ids)
    for df in (emb, pred, info):
        df.index.name = "sample"
    return {"embedding": emb, "predictions": pred, "samples": info, "coverage": coverage}
