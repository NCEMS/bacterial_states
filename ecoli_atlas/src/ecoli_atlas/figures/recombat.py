"""Optional reComBat baseline for the study-leakage comparison (notebook cell 43). reComBat is not
a dependency; without it ``recombat_embedding`` raises ImportError and the bar is skipped."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["growth_phase_c", "medium_type_c"]
MIN_BATCH = 2


def recombat_embedding(ctx, log=None) -> pd.DataFrame:
    log = log or ctx.log
    try:
        from reComBat import reComBat
    except ImportError as e:
        raise ImportError("reComBat is not installed; `pip install reComBat --no-deps` into the ecoli_atlas "
                          "environment to include the transductive baseline") from e
    from sklearn.decomposition import PCA
    meta = ctx.meta
    latent = ctx.embedding.shape[1]
    islab = "is_lab_strain_c" if "is_lab_strain_c" in meta.columns else "is_lab_strain"
    n0 = len(meta)
    s1 = meta.index[ctx.data.cpm.sum(axis=1) > 0]
    s2 = s1[meta.loc[s1, REQUIRED].notna().all(axis=1).values]
    vc = meta.loc[s2, "gse"].astype(str).value_counts()
    keep = s2[meta.loc[s2, "gse"].astype(str).isin(vc[vc >= MIN_BATCH].index).values]
    log(f"[reComBat] kept {len(keep)}/{n0} (dropped {n0 - len(s1)} zero-count, {len(s1) - len(s2)} unlabelled, "
        f"{len(s2) - len(keep)} singleton-study)")
    X_design = meta.loc[keep, REQUIRED].astype(str)
    if islab in meta.columns:
        d = pd.get_dummies(meta.loc[keep, islab], prefix=islab, drop_first=True).astype(float)
        d.columns = [f"{c}_numerical" for c in d.columns]
        X_design = pd.concat([X_design, d], axis=1)
    logcpm = pd.DataFrame(ctx.logcpm, index=meta.index, columns=ctx.genes)
    rc_data = logcpm.loc[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gvar = rc_data.var(axis=0)
    rc_data = rc_data[gvar[gvar > 1e-8].index]
    rc = reComBat(parametric=True, model="ridge", verbose=False)
    corr = rc.fit_transform(rc_data, meta.loc[keep, "gse"].astype(str), X=X_design)
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    emb = PCA(latent, random_state=ctx.seed).fit_transform(corr.values)
    log(f"[reComBat] corrected {corr.shape[1]} genes -> PCA{emb.shape}")
    return pd.DataFrame(emb, index=keep)
