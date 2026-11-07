"""`ecoli_atlas train`: factor encoders -> thresholds -> TEST metrics -> residual AE -> saved bundle."""
from __future__ import annotations

import datetime
import importlib.metadata as im
import platform
from dataclasses import dataclass

import numpy as np
import torch

from . import __version__
from .artifacts import SCHEMA_VERSION, FactorArtifact, ModelBundle
from .config import FactorHParams, TrainConfig
from .data import TrainingData
from .metrics import classification_metrics, tune_threshold
from .train_factor import embed_all, factor_oof_scores, head_proba, train_factor_encoder
from .train_residual import block_stats, embed_residual, standardize_blocks, train_residual

INFERENCE_PIPELINE = [
    "raw counts (samples x genes)",
    "drop rRNA/tRNA; map locus_tag to gene symbol (ASM584v2 feature table)",
    "reindex columns to preprocessing['genes']; absent genes are filled with 0 counts and become strongly "
    "negative after z-scoring, so check gene coverage before projecting",
    "per-sample CPM; per-gene z-score with zscore_mu and zscore_sd",
    "each factor encoder: input -> block embedding; head softmax with factor_thresholds[f] applied to "
    "P(class at positive_class_index[f])",
    "concatenate blocks in factor_order; standardize with tb_mu and tb_sd",
    "residual_ae(input, standardized blocks) -> [factor blocks | residual]",
]


def _version(pkg: str) -> str:
    try:
        return im.version(pkg)
    except Exception:
        return "unknown"


@dataclass
class TrainResult:
    bundle: ModelBundle
    embedding: np.ndarray            # final embedding of every aligned training sample
    factor_tuning: dict              # factor -> best CV value (if tuned)
    residual_tuning: float | None


def run_training(data: TrainingData, cfg: TrainConfig, device, log=None, compute_thresholds: bool = True,
                 training_data_version: str | None = None) -> TrainResult:
    log = log or (lambda *_: None)
    factor_models, factor_emb, thresholds, threshold_diag, test_metrics, tuning = {}, {}, {}, {}, {}, {}

    for f in cfg.factors:
        fc = data.factors[f]
        hp = cfg.factor_hparams.get(f)
        if cfg.tune:
            from .tune import tune_factor
            log(f"[{f}] tuning {cfg.n_trials_factor} trials x {cfg.n_folds} folds")
            hp, tuning[f] = tune_factor(data, f, cfg, device, log)
            cfg.factor_hparams[f] = hp
        elif hp is None:
            hp = FactorHParams()
            cfg.factor_hparams[f] = hp
            log(f"[{f}] no hyper-parameters given; using defaults {hp}")
        log(f"[{f}] training final encoder {hp}")
        model = train_factor_encoder(data, f, hp, cfg, device, log=log)
        factor_models[f] = model
        factor_emb[f] = embed_all(model, data.X, device)

        if compute_thresholds:
            rows, p = factor_oof_scores(data, f, hp, cfg, device)
            if len(rows):
                y_bin = (fc.codes[rows] == fc.pos_idx).astype(int)
                thr, ba, at_edge = tune_threshold(y_bin, p, cfg.thresh_grid)
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(y_bin, p)) if len(np.unique(y_bin)) > 1 else float("nan")
                threshold_diag[f] = {"n": int(len(rows)), "balanced_acc": ba, "auroc": auroc}
                log(f"[{f}] threshold={thr:.3f} | out-of-fold balanced acc={ba:.3f} | AUROC={auroc:.3f} (n={len(rows)})"
                    + ("  [warning: grid boundary]" if at_edge else ""))
            else:
                thr = 0.5
        else:
            thr = 0.5
        thresholds[f] = float(thr)

        te = data.te_rows
        te = te[fc.codes[te] >= 0]
        if len(te) and len(np.unique(fc.codes[te])) > 1:
            p_pos = head_proba(model, data.X, te, device)[:, fc.pos_idx]
            test_metrics[f] = classification_metrics((fc.codes[te] == fc.pos_idx).astype(int), p_pos, thr)
            tm = test_metrics[f]
            log(f"[{f}] TEST n={tm['n']} | balanced acc={tm['balanced_acc']:.3f} "
                f"[{tm['ba_ci'][0]:.3f}, {tm['ba_ci'][1]:.3f}] | AUROC={tm['auroc']:.3f}")

    factor_order = list(cfg.factors)
    concat = np.concatenate([factor_emb[f] for f in factor_order], axis=1)
    tb_mu, tb_sd = block_stats(concat, data.tr_rows)
    TB = standardize_blocks(concat, tb_mu, tb_sd)
    log(f"concatenated blocks: {concat.shape} (widths {[factor_emb[f].shape[1] for f in factor_order]})")

    residual_tuning = None
    if cfg.tune:
        from .tune import tune_residual
        log(f"[residual] tuning {cfg.n_trials_top} trials x {cfg.n_folds} folds")
        cfg.residual_hparams, residual_tuning = tune_residual(data, TB, cfg, device, log)
    log(f"[residual] training {cfg.residual_hparams}")
    residual = train_residual(data, TB, cfg.residual_hparams, cfg, device, log=log)
    top_emb = embed_residual(residual, data.X, TB, device)
    log(f"final embedding {top_emb.shape} = [factor blocks {TB.shape[1]} | residual {cfg.residual_hparams.res_dim}]")

    # ---- bundle --------------------------------------------------------------------------------
    emb_dims = {f: int(factor_emb[f].shape[1]) for f in factor_order}
    layout, cursor = [], 0
    for f in factor_order:
        layout.append({"block": f, "kind": "factor", "start": cursor, "end": cursor + emb_dims[f]})
        cursor += emb_dims[f]
    res_dim = int(cfg.residual_hparams.res_dim)
    layout.append({"block": "residual", "kind": "residual", "start": cursor, "end": cursor + res_dim})
    classes = {f: list(data.factors[f].classes) for f in factor_order}
    class_dtype = {f: ("bool" if set(classes[f]) == {"False", "True"} else "str") for f in factor_order}
    prep = {
        "encoder_norm": "zscore_cpm",
        "genes": list(data.genes),
        "zscore_mu": data.zscore.mu, "zscore_sd": data.zscore.sd,
        "tb_mu": tb_mu, "tb_sd": tb_sd,
        "factor_order": factor_order,
        "factor_emb_dims": emb_dims,
        "res_dim": res_dim,
        "top_emb_dim": int(top_emb.shape[1]),
        "block_layout": layout,
        "factor_classes": classes,
        "factor_class_dtype": class_dtype,
        "factor_thresholds": thresholds,
        "positive_class_index": {f: int(data.factors[f].pos_idx) for f in factor_order},
        "seed": cfg.seed,
    }
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ecoli_atlas_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "training_data_version": training_data_version,
        "seed": cfg.seed,
        "n_input_genes": len(data.genes),
        "n_training_samples": int(len(data.tr_rows)),
        "factor_order": factor_order,
        "factor_emb_dims": emb_dims,
        "residual_dim": res_dim,
        "final_embedding_dim": int(top_emb.shape[1]),
        "final_embedding_block_layout": layout,
        "factor_classes": classes,
        "factor_class_dtype": class_dtype,
        "factor_thresholds": thresholds,
        "threshold_diagnostics": threshold_diag,
        "test_metrics": test_metrics,
        "tuning": {"factor_cv_auroc": tuning, "residual_objective": residual_tuning},
        "hyperparameters": cfg.hyperparameters_dict(),
        "split_config": cfg.split_config_dict(),
        "environment": {p: _version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "optuna", "umap-learn"]}
        | {"python": platform.python_version(), "device": str(device)},
        "files": {"factor_encoders": {f: f"{f}_encoder.pt" for f in factor_order}, "residual_ae": "residual_ae.pt",
                  "preprocessing": "preprocessing.pt", "splits": "splits.csv"},
        "inference_pipeline": INFERENCE_PIPELINE,
    }
    factors = {f: FactorArtifact(f, factor_models[f], classes[f], thresholds[f], data.factors[f].positive_class)
               for f in factor_order}
    bundle = ModelBundle(factors, residual, prep, manifest).to(device)
    return TrainResult(bundle, top_emb, tuning, residual_tuning)


def reload_check(bundle: ModelBundle, data: TrainingData, embedding: np.ndarray, n: int = 50, atol: float = 1e-4) -> float:
    """Re-embed the first ``n`` samples with the saved-and-reloaded bundle and return the max deviation."""
    from .embed import encode
    Z = data.X[:n].numpy()
    final, _, _ = encode(Z, bundle)
    dev = float(np.max(np.abs(final - embedding[:n])))
    if dev > atol:
        raise RuntimeError(f"reload check failed: max deviation {dev:.2e} > {atol}")
    return dev
