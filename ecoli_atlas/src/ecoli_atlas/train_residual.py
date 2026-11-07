"""The residual autoencoder: reconstructs expression from [frozen factor blocks | residual] with a
within-study push and a cross-study control pull on the final embedding."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import ResidualHParams, TrainConfig
from .data import TrainingData
from .losses import cos_dist, cross_study_control_pull, recon_loss, within_study_push
from .metrics import knn_label_agreement
from .models import ResidualAE
from .utils import set_seed

TOP_KNN_FACTORS = ("growth_phase_c", "medium_type_c")
OBJ_RECON_W, OBJ_KNN_W = 1.0, 2.0


def block_stats(concat: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean / sd of the concatenated factor blocks over the training pool (sd 0 -> 1)."""
    mu = concat[rows].mean(0)
    sd = concat[rows].std(0)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_blocks(concat: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> torch.Tensor:
    return torch.tensor(((concat - mu) / sd).astype(np.float32), dtype=torch.float32)


def study_blocked_batches(rows, gse_of_row, batch_size: int, seed: int) -> list:
    """Pack whole (shuffled) studies into each batch so within-study pairs are always present."""
    rng = np.random.RandomState(seed)
    by: dict = {}
    for r in rows:
        by.setdefault(int(gse_of_row[r]), []).append(int(r))
    order = list(by)
    rng.shuffle(order)
    batch, out = [], []
    for g in order:
        rg = by[g]
        rng.shuffle(rg)
        batch += rg
        if len(batch) >= batch_size:
            out.append(np.array(batch))
            batch = []
    if len(batch) > 1:
        out.append(np.array(batch))
    return out


def train_residual(data: TrainingData, TB: torch.Tensor, hp: ResidualHParams, cfg: TrainConfig, device,
                   epochs: int | None = None, rows: np.ndarray | None = None, seed: int | None = None,
                   log=None) -> ResidualAE:
    seed = cfg.seed if seed is None else seed
    epochs = cfg.top_epochs if epochs is None else epochs
    rows = data.tr_rows if rows is None else np.asarray(rows)
    set_seed(seed)
    gse_t = torch.tensor(np.asarray(data.gse_codes), dtype=torch.long)
    cid_t = torch.tensor(np.asarray(data.cond_id), dtype=torch.long)
    ann_t = torch.tensor(np.asarray(data.is_annot), dtype=torch.bool)
    ccond_t = torch.tensor(np.asarray(data.ctrl_cond), dtype=torch.long)
    ctrl_t = torch.tensor(np.asarray(data.is_ctrl), dtype=torch.bool)
    m = ResidualAE(data.n_genes, TB.shape[1], hp.res_dim, hp.width, hp.depth, hp.dropout).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=hp.lr)
    m.train()
    for ep in range(epochs):
        er = eps = epc = 0.0
        nb = 0
        for ridx in study_blocked_batches(rows, data.gse_codes, cfg.top_batch, seed=seed + ep):
            rt = torch.as_tensor(ridx)
            xr = data.X[rt].to(device)
            bl = TB[rt].to(device)
            _, z, xh = m(xr, bl)
            recon = recon_loss(xh, xr)
            d = cos_dist(z)
            push = within_study_push(d, gse_t[rt].to(device), cid_t[rt].to(device), ann_t[rt].to(device), cfg.push_margin)
            cpull = cross_study_control_pull(d, gse_t[rt].to(device), ccond_t[rt].to(device), ctrl_t[rt].to(device))
            loss = recon + hp.push_w * push + hp.ctrl_w * cpull
            opt.zero_grad()
            loss.backward()
            opt.step()
            er += float(recon.detach())
            eps += float(push.detach())
            epc += float(cpull.detach())
            nb += 1
        if log and nb and (ep % 10 == 0 or ep == epochs - 1):
            log(f"  epoch {ep:2d}  recon={er / nb:.4f}  push={eps / nb:.4f}  pull={epc / nb:.4f}")
    m.eval()
    return m


@torch.no_grad()
def embed_residual(m: ResidualAE, X: torch.Tensor, TB: torch.Tensor, device, rows=None, chunk: int = 1024) -> np.ndarray:
    m.eval()
    idx = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    out = []
    for i in range(0, len(idx), chunk):
        sl = torch.as_tensor(idx[i:i + chunk])
        _, z, _ = m(X[sl].to(device), TB[sl].to(device))
        out.append(z.cpu())
    return torch.cat(out, 0).numpy() if out else np.zeros((0, TB.shape[1] + m.arch()["d_res"]), np.float32)


@torch.no_grad()
def recon_r2(m: ResidualAE, X: torch.Tensor, TB: torch.Tensor, rows, device) -> float:
    vt = torch.as_tensor(np.asarray(rows))
    xr = X[vt].to(device)
    _, _, xh = m(xr, TB[vt].to(device))
    return max(0.0, 1.0 - ((xh - xr) ** 2).mean().item() / xr.var(unbiased=False).item())


def top_cv_score(data: TrainingData, TB: torch.Tensor, hp: ResidualHParams, cfg: TrainConfig, device) -> float | None:
    """Tuning objective: cross-study kNN label agreement (weight 2) plus held-out reconstruction R2
    (weight 1), averaged over study folds. None if no fold could be scored."""
    agr, r2s = [], []
    for k in range(1, cfg.n_folds + 1):
        fit, val = data.fit_rows(k), data.val_rows(k)
        if len(val) < 5:
            continue
        m = train_residual(data, TB, hp, cfg, device, epochs=cfg.top_epochs, rows=fit)
        need = np.union1d(fit, val)
        E_all = np.zeros((data.n_samples, TB.shape[1] + hp.res_dim), dtype=np.float32)
        E_all[need] = embed_residual(m, data.X, TB, device, rows=need)
        r2s.append(recon_r2(m, data.X, TB, val, device))
        fa = []
        for fcol in TOP_KNN_FACTORS:
            if fcol not in data.meta.columns:
                continue
            y = np.asarray(pd.Categorical(data.meta[fcol]).codes)
            tr = fit[y[fit] >= 0]
            va = val[y[val] >= 0]
            if len(tr) < 10 or len(va) < 3 or len(np.unique(y[tr])) < 2:
                continue
            fa.append(knn_label_agreement(E_all[tr], y[tr], data.gse_codes[tr], E_all[va], y[va], data.gse_codes[va], cfg.knn_k))
        if fa:
            agr.append(float(np.mean(fa)))
    if not agr:
        return None
    return (OBJ_KNN_W * float(np.mean(agr)) + OBJ_RECON_W * float(np.clip(np.mean(r2s), 0, 1))) / (OBJ_KNN_W + OBJ_RECON_W)
