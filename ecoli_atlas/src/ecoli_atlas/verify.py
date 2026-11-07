"""`ecoli_atlas verify-models`: compare two model bundles on the same count table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .artifacts import ModelBundle, load_splits
from .embed import embed_counts
from .labels import build_labels, factor_codes
from .preprocess import GeneMap
from .utils import l2norm


def _knn_overlap(E1: np.ndarray, E2: np.ndarray, k: int) -> float:
    n1 = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(l2norm(E1)).kneighbors(l2norm(E1), return_distance=False)[:, 1:]
    n2 = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(l2norm(E2)).kneighbors(l2norm(E2), return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(n1, n2)]))


def _test_metrics(bundle: ModelBundle, res, meta: pd.DataFrame, splits: pd.DataFrame | None) -> dict:
    if splits is None:
        return {}
    lab = build_labels(meta.loc[meta.index.intersection(res.embedding.index)])
    sp = splits.reindex(lab.index)
    test = lab.index[(sp["split"] == 3).values]
    out = {}
    for f in bundle.factor_order:
        fc = factor_codes(lab, f)
        codes = pd.Series(fc.codes, index=lab.index).reindex(test)
        keep = codes.values >= 0
        if keep.sum() == 0 or len(np.unique(codes.values[keep])) < 2:
            continue
        y = (codes.values[keep] == fc.pos_idx).astype(int)
        p = res.proba[f][[res.embedding.index.get_loc(s) for s in test[keep]], fc.pos_idx]
        thr = bundle.factors[f].threshold if bundle.factors[f].threshold is not None else 0.5
        out[f] = {"n": int(keep.sum()), "balanced_acc": float(balanced_accuracy_score(y, (p >= thr).astype(int))),
                  "auroc": float(roc_auc_score(y, p))}
    return out


def verify_models(old_dir, new_dir, counts: pd.DataFrame, feat: pd.DataFrame, meta: pd.DataFrame | None = None,
                  device="cpu", k: int = 15, log=None) -> dict:
    log = log or print
    old = ModelBundle.load(old_dir, device)
    new = ModelBundle.load(new_dir, device)
    report: dict = {"old": str(old_dir), "new": str(new_dir)}

    report["genes_identical"] = old.genes == new.genes
    report["factor_order_identical"] = old.factor_order == new.factor_order
    report["arch_identical"] = {f: old.factors[f].model.arch() == new.factors[f].model.arch()
                                for f in old.factor_order if f in new.factors}
    report["residual_arch_identical"] = old.residual.arch() == new.residual.arch()
    if report["genes_identical"]:
        report["zscore_mu_max_abs_diff"] = float(np.max(np.abs(old.zscore.mu - new.zscore.mu)))
        report["zscore_sd_max_abs_diff"] = float(np.max(np.abs(old.zscore.sd - new.zscore.sd)))
    report["thresholds"] = {f: [old.factors[f].threshold, new.factors[f].threshold] for f in old.factor_order if f in new.factors}

    gm = GeneMap.from_feature_table(feat)
    r_old = embed_counts(counts, old, gm)
    r_new = embed_counts(counts, new, gm)
    same_layout = list(r_old.embedding.columns) == list(r_new.embedding.columns)
    report["embedding_layout_identical"] = same_layout
    if same_layout:
        diff = np.abs(r_old.embedding.values - r_new.embedding.values)
        report["embedding_max_abs_diff"] = float(diff.max())
        report["embedding_mean_abs_diff"] = float(diff.mean())
        cos = (l2norm(r_old.embedding.values) * l2norm(r_new.embedding.values)).sum(1)
        report["embedding_cosine_similarity"] = {"mean": float(cos.mean()), "min": float(cos.min())}
        report["knn_overlap"] = {"k": k, "mean_fraction": _knn_overlap(r_old.embedding.values, r_new.embedding.values, k)}
    report["prediction_agreement"] = {
        f: float((r_old.predictions[f"{f}_pred"].values == r_new.predictions[f"{f}_pred"].values).mean())
        for f in old.factor_order if f in new.factors}
    if meta is not None:
        meta = meta.copy()
        meta.index = meta.index.astype(str).str.strip()
        report["test_metrics"] = {"old": _test_metrics(old, r_old, meta, load_splits(old_dir)),
                                  "new": _test_metrics(new, r_new, meta, load_splits(new_dir))}
        if old.manifest and "test_metrics" in old.manifest:
            report["test_metrics"]["old_manifest"] = {f: {k2: v[k2] for k2 in ("n", "balanced_acc", "auroc")}
                                                      for f, v in old.manifest["test_metrics"].items()}
    # seed sweeps: is the other bundle's TEST performance inside the sweep's min..max?
    from .sweep import within_spread
    if new.manifest and "seed_sweep" in new.manifest and old.manifest and "test_metrics" in old.manifest:
        report["old_within_new_seed_spread"] = within_spread(old.manifest["test_metrics"], new.manifest["seed_sweep"])
    if old.manifest and "seed_sweep" in old.manifest and new.manifest and "test_metrics" in new.manifest:
        report["new_within_old_seed_spread"] = within_spread(new.manifest["test_metrics"], old.manifest["seed_sweep"])
    log(json.dumps(report, indent=2, default=str))
    return report
