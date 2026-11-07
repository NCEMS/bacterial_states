"""Supervised factor encoders: one small classifier per annotated factor whose penultimate layer is
the factor's embedding block."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import FactorHParams, TrainConfig
from .data import TrainingData
from .losses import class_weights
from .models import FactorEncoder
from .utils import set_seed

WEIGHT_DECAY = 1e-5


def train_factor_encoder(data: TrainingData, factor: str, hp: FactorHParams, cfg: TrainConfig, device,
                         rows: np.ndarray | None = None, seed: int | None = None, log=None) -> FactorEncoder:
    seed = cfg.seed if seed is None else seed
    set_seed(seed)
    fc = data.factors[factor]
    rows = data.tr_rows if rows is None else np.asarray(rows)
    rows = rows[fc.codes[rows] >= 0]
    if len(rows) < 2:
        raise ValueError(f"{factor}: too few labelled training rows")
    lab = data.labels_tensor(factor)
    model = FactorEncoder(data.n_genes, hp.emb_dim, fc.n_classes, hp.width, hp.depth, hp.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=WEIGHT_DECAY)
    rows_t = torch.as_tensor(rows)
    lab_rows = lab[rows_t].numpy()
    w = class_weights(lab_rows, fc.n_classes)
    ce = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32, device=device))
    if log:
        c, n = np.unique(lab_rows, return_counts=True)
        log(f"  [{factor}] counts {dict(zip(c.tolist(), n.tolist()))} -> weights {np.round(w, 3).tolist()}")
    dl = DataLoader(TensorDataset(rows_t), batch_size=min(cfg.factor_batch, len(rows)), shuffle=True,
                    drop_last=len(rows) > cfg.factor_batch)
    model.train()
    for _ in range(cfg.factor_epochs):
        for (ridx,) in dl:
            _, out = model(data.X[ridx].to(device))
            loss = ce(out, lab[ridx.cpu()].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def embed_all(model: FactorEncoder, X: torch.Tensor, device, chunk: int = 1024) -> np.ndarray:
    model.eval()
    return torch.cat([model.encoder(X[i:i + chunk].to(device)).cpu() for i in range(0, X.shape[0], chunk)], 0).numpy()


@torch.no_grad()
def head_proba(model: FactorEncoder, X: torch.Tensor, rows, device, chunk: int = 1024) -> np.ndarray:
    model.eval()
    rows = torch.as_tensor(np.asarray(rows))
    return torch.cat([torch.softmax(model(X[rows[i:i + chunk]].to(device))[1], dim=1).cpu()
                      for i in range(0, len(rows), chunk)], 0).numpy()


def factor_cv_score(data: TrainingData, factor: str, hp: FactorHParams, cfg: TrainConfig, device) -> float | None:
    """Study-level k-fold CV: held-out P(positive) pooled across folds and scored once by AUROC.
    Returns None when no fold could be scored."""
    fc = data.factors[factor]
    y = fc.codes
    oof_true, oof_pred = [], []
    for k in range(1, cfg.n_folds + 1):
        fit_rows, val_rows = data.fit_rows(k), data.val_rows(k)
        va = val_rows[y[val_rows] >= 0]
        fl = fit_rows[y[fit_rows] >= 0]
        if len(fl) < 10 or len(va) < 1 or len(np.unique(y[fl])) < 2:
            continue
        m = train_factor_encoder(data, factor, hp, cfg, device, rows=fit_rows)
        oof_true.append(y[va])
        oof_pred.append(head_proba(m, data.X, va, device)[:, fc.pos_idx])
    if not oof_true:
        return None
    t, p = np.concatenate(oof_true), np.concatenate(oof_pred)
    if len(np.unique(t)) < 2:
        return None
    return float(roc_auc_score(t, p))


def factor_oof_scores(data: TrainingData, factor: str, hp: FactorHParams, cfg: TrainConfig, device):
    """P(positive class) for every labelled pool sample from a model that did not see it."""
    fc = data.factors[factor]
    y = fc.codes
    rows_out, p_out = [], []
    for k in range(1, cfg.n_folds + 1):
        fit_rows, val_rows = data.fit_rows(k), data.val_rows(k)
        va = val_rows[y[val_rows] >= 0]
        if len(va) == 0:
            continue
        m = train_factor_encoder(data, factor, hp, cfg, device, rows=fit_rows)
        rows_out.append(va)
        p_out.append(head_proba(m, data.X, va, device)[:, fc.pos_idx])
    if not rows_out:
        return np.array([], dtype=int), np.array([])
    return np.concatenate(rows_out), np.concatenate(p_out)
