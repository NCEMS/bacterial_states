"""Training of the atlas model. Three fits (``TrainConfig.fit``):

two_stage     as for the first atlas model. (1) One block encoder per call, each fitted on the
              training samples that have a label for it, with a class-weighted cross-entropy.
              (2) With the blocks frozen, the residual encoder and the count decoder on all
              training samples, with the loss below without its head term.
one_pass      the same blocks and residual, all fitted together with the whole loss.
unstructured  one embedding without blocks, fitted with the whole loss.

    loss = count likelihood (negative binomial, mean over features)
         + kl_w   * KL from the standard normal (residual block, or the unstructured embedding)
         + head_w * class-weighted cross-entropy of each head (labelled samples only; within a class no
                    study counts for more than the label cap, see ``data.label_weights``)
         + push_w * within-study push   (samples of one study known to differ are kept apart)
         + ctrl_w * cross-study pull    (matching control conditions of different studies are drawn together)

With the ordinal phase head (``TrainConfig.phase_head``) the phase term is a proportional-odds loss
over the levels exponential < transition < stationary instead of a two-class cross-entropy.

Both alignment terms act on the whole embedding. Batches are built from whole studies, so that
they always have pairs to act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import HEADS, TrainConfig
from .data import AtlasTables, TrainingData, build_training, label_weights, study_fold
from .features import FeatureSet
from .losses import class_weights, cos_dist, cross_study_control_pull, kl_normal, masked_mean, nb_nll, within_study_push
from .metrics import classification_metrics, within_study_order
from .models import AtlasModel, OrdinalHead, build_model
from .utils import set_seed

GRAD_CLIP = 5.0


def capped_rows(rows: np.ndarray, gse: np.ndarray, cap: int, rng: np.random.RandomState) -> dict:
    """study code -> its rows, at most ``cap`` of them (drawn at random)."""
    by: dict = {}
    for r in rows:
        by.setdefault(int(gse[r]), []).append(int(r))
    return {g: (list(rng.choice(rg, cap, replace=False)) if len(rg) > cap else rg) for g, rg in by.items()}


def study_blocked_batches(rows, gse, batch_size: int, seed: int, cap: int) -> list:
    """Pack whole (shuffled) studies into each batch; a study larger than ``cap`` contributes a
    fresh random ``cap`` of its samples each time."""
    rng = np.random.RandomState(seed)
    by = capped_rows(np.asarray(rows), gse, cap, rng)
    order = list(by)
    rng.shuffle(order)
    batch, out = [], []
    for g in order:
        rg = list(by[g])
        rng.shuffle(rg)
        batch += rg
        if len(batch) >= batch_size:
            out.append(np.array(batch))
            batch = []
    if len(batch) > 1:
        out.append(np.array(batch))
    return out


def model_input(data: TrainingData, rt: torch.Tensor, pool: np.ndarray, cfg: TrainConfig, rng: np.random.RandomState) -> torch.Tensor:
    """The z-scored expression of the samples ``rt`` for a training step. A share
    ``cfg.mask_transplant`` of them is also given the missing genes of a random sample of ``pool``
    (set to 0, the gene's mean), so that every sample is seen with the gaps of other strains and
    the pattern of gaps says nothing about the sample."""
    x = data.X[rt]
    if not cfg.mask_missing or cfg.mask_transplant <= 0 or data.M is None:
        return x
    take = rng.random_sample(len(rt)) < cfg.mask_transplant
    if not take.any():
        return x
    donors = torch.as_tensor(rng.choice(pool, int(take.sum())))
    x = x.clone()
    idx = torch.as_tensor(np.flatnonzero(take))
    x[idx] = x[idx].masked_fill(data.M[donors], 0.0)
    return x


def head_weighting(data: TrainingData, head: str, rows: np.ndarray, cfg: TrainConfig, device) -> tuple:
    """(weight of every sample's label, class weights) of one call for a fit on ``rows``. The label
    weights cap every study within a class (``label_weights``); the class weights are made from the
    weighted class sizes."""
    w, _ = label_weights(data.y_fit[head], data.gse_codes, rows, cfg.label_cap)
    cw = class_weights(data.y_fit[head], len(data.fit_classes[head]), sample_weight=w)
    return torch.tensor(w, dtype=torch.float32), torch.tensor(cw, dtype=torch.float32, device=device)


def weighted_cross_entropy(logits: torch.Tensor, lab: torch.Tensor, w: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the labelled samples (``lab`` >= 0), each weighted by its label weight and
    its class weight; 0 when no sample of the batch has weight."""
    has = (lab >= 0) & (w > 0)
    if not bool(has.any()):
        return logits.new_tensor(0.0)
    wt = w[has] * cw[lab[has]]
    return (F.cross_entropy(logits[has], lab[has], reduction="none") * wt).sum() / wt.sum()


def weighted_ordinal_loss(cum: torch.Tensor, lab: torch.Tensor, w: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """Proportional-odds loss over the labelled samples: for every threshold k the binary
    cross-entropy of ``cum[:, k]`` (logit of P(level > k)) against ``lab > k``, summed over the
    thresholds and weighted as in ``weighted_cross_entropy``."""
    has = (lab >= 0) & (w > 0)
    if not bool(has.any()):
        return cum.new_tensor(0.0)
    wt = w[has] * cw[lab[has]]
    target = (lab[has][:, None] > torch.arange(cum.shape[1], device=cum.device)[None, :]).to(cum.dtype)
    per = F.binary_cross_entropy_with_logits(cum[has], target, reduction="none").sum(dim=1)
    return (per * wt).sum() / wt.sum()


def fit_block(model: AtlasModel, data: TrainingData, head: str, cfg: TrainConfig, device, rows: np.ndarray, seed: int) -> list:
    """Fit one block encoder on the labelled samples among ``rows``, every study capped within each
    class (``head_weighting``). Returns the mean cross-entropy of every epoch."""
    hp = cfg.hparams
    rows = rows[data.y_fit[head][rows] >= 0]
    if len(np.unique(data.y_fit[head][rows])) < len(data.fit_classes[head]):
        raise ValueError(f"{head}: the samples to fit on do not have all of {data.fit_classes[head]}")
    block = model.blocks[head]
    opt = torch.optim.Adam(block.parameters(), lr=hp.block_lr, weight_decay=hp.weight_decay)
    y = torch.tensor(data.y_fit[head], dtype=torch.long)
    w, cw = head_weighting(data, head, rows, cfg, device)
    loss_fn = weighted_ordinal_loss if isinstance(block.head, OrdinalHead) else weighted_cross_entropy
    rng = np.random.RandomState(seed)
    losses = []
    block.train()
    for _ in range(cfg.block_epochs):
        order = rng.permutation(rows)
        total, n = 0.0, 0
        for i in range(0, len(order), cfg.block_batch):
            rt = torch.as_tensor(order[i:i + cfg.block_batch])
            if len(rt) < 2:
                continue
            loss = loss_fn(block(model_input(data, rt, rows, cfg, rng).to(device))[1], y[rt].to(device), w[rt].to(device), cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + float(loss.detach()), n + 1
        losses.append(total / max(n, 1))
    block.eval()
    return losses


@torch.no_grad()
def set_block_statistics(model: AtlasModel, data: TrainingData, rows: np.ndarray, device, chunk: int = 2048) -> None:
    """Mean and sd of every block dimension over ``rows`` (sd 0 -> 1), kept in the model."""
    raw = torch.cat([model.raw_blocks(data.X[torch.as_tensor(rows[i:i + chunk])].to(device)) for i in range(0, len(rows), chunk)])
    sd = raw.std(0, unbiased=False)
    model.block_mu.copy_(raw.mean(0))
    model.block_sd.copy_(torch.where(sd > 0, sd, torch.ones_like(sd)))


def run_epochs(model, params: list, data: TrainingData, cfg: TrainConfig, device, rows: np.ndarray, seed: int, epochs: int,
               stage: str, heads: bool = False, train_blocks: bool = False, log=None) -> list:
    """Optimise ``params`` over study-blocked batches of ``rows``. The loss is the count likelihood
    (over the genes a sample has), the KL term and the two alignment terms, plus each head's
    cross-entropy when ``heads`` is set. Returns one history row per epoch (means of the terms)."""
    hp = cfg.hparams
    opt = torch.optim.Adam(params, lr=hp.lr, weight_decay=hp.weight_decay)
    gse = torch.tensor(data.gse_codes, dtype=torch.long)
    ident = {k: torch.tensor(v, dtype=torch.bool if v.dtype == bool else torch.long) for k, v in data.identity.items()}
    if heads:
        weighting = {h: head_weighting(data, h, rows, cfg, device) for h in HEADS}
        y = {h: torch.tensor(v, dtype=torch.long) for h, v in data.y_fit.items()}
    history = []
    model.train()
    for ep in range(epochs):
        sums: dict = {}
        nb_batches = 0
        rng = np.random.RandomState(seed + 7919 * (ep + 1))
        for ridx in study_blocked_batches(rows, data.gse_codes, cfg.batch, seed + ep, data.study_cap):
            rt = torch.as_tensor(ridx)
            out = model(model_input(data, rt, rows, cfg, rng).to(device), data.C[rt].to(device), train_blocks=train_blocks)
            nll = nb_nll(data.Y[rt].to(device), out["log_rate"] + data.log_s[rt].to(device)[:, None], model.log_theta)
            terms = {"counts": (nll.mean() if data.M is None else masked_mean(nll, ~data.M[rt].to(device))).float(),
                     "kl": kl_normal(out["kl_mu"], out["kl_logvar"]).mean()}
            loss = terms["counts"] + hp.kl_w * terms["kl"]
            if heads:
                for h in HEADS:
                    lab, w = y[h][rt].to(device), weighting[h][0][rt].to(device)
                    terms[h] = (weighted_ordinal_loss(out["cum"][h], lab, w, weighting[h][1]) if h in out["cum"]
                                else weighted_cross_entropy(out["logits"][h], lab, w, weighting[h][1]))
                    loss = loss + hp.head_w * terms[h]
            d = cos_dist(out["mu"])
            g = gse[rt].to(device)
            terms["push"] = within_study_push(d, g, ident["cond_id"][rt].to(device), ident["is_annot"][rt].to(device), hp.push_margin)
            terms["pull"] = cross_study_control_pull(d, g, ident["ctrl_cond"][rt].to(device), ident["is_ctrl"][rt].to(device))
            loss = loss + hp.push_w * terms["push"] + hp.ctrl_w * terms["pull"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            nb_batches += 1
        history.append({"stage": stage, "epoch": ep, **{k: v / max(nb_batches, 1) for k, v in sums.items()}})
        if log and (ep % 10 == 0 or ep == epochs - 1):
            log(f"  epoch {ep:3d}  " + "  ".join(f"{k}={v:.4f}" for k, v in history[-1].items() if k not in ("epoch", "stage")))
    model.eval()
    return history


def fit(data: TrainingData, cfg: TrainConfig, device, rows: np.ndarray | None = None, seed: int | None = None,
        epochs: int | None = None, log=None) -> tuple:
    """Train on ``rows`` (default: the training pool) in the way ``cfg.fit`` names. Returns the model
    (eval mode) and the history: one row per epoch of every stage (``stage`` is a head name,
    ``residual``, ``one_pass`` or ``unstructured``)."""
    seed = cfg.seed if seed is None else seed
    epochs = cfg.epochs if epochs is None else epochs
    rows = data.tr_rows if rows is None else np.asarray(rows)
    set_seed(seed)
    model = build_model(cfg.fit, data.fs.n, data.C.shape[1], cfg.hparams, torch.tensor(data.log_mean), cfg.phase_head).to(device)
    history = []
    if cfg.fit == "two_stage":
        for h in HEADS:
            losses = fit_block(model, data, h, cfg, device, rows, seed)
            history += [{"stage": h, "epoch": ep, "ce": v} for ep, v in enumerate(losses)]
            if log:
                log(f"  block {h}: {int((data.y_fit[h][rows] >= 0).sum())} labelled samples, loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        set_block_statistics(model, data, rows, device)
        history += run_epochs(model, model.residual_parameters(), data, cfg, device, rows, seed, epochs, "residual", log=log)
    else:
        history += run_epochs(model, list(model.parameters()), data, cfg, device, rows, seed, epochs, cfg.fit, heads=True,
                              train_blocks=cfg.fit == "one_pass", log=log)
    return model, history


@torch.no_grad()
def encode(model, X: torch.Tensor, C: torch.Tensor, device, chunk: int = 2048) -> dict:
    """The embedding (blocks and the mean of the residual), each head's probability of its positive
    class (``probs``) and the log-odds of it (``scores``; not saturated where the probability is)."""
    model.eval()
    zs, probs, scores = [], {h: [] for h in HEADS}, {h: [] for h in HEADS}
    for i in range(0, X.shape[0], chunk):
        out = model(X[i:i + chunk].to(device), C[i:i + chunk].to(device), sample=False)
        zs.append(out["mu"].cpu())
        for h in HEADS:
            probs[h].append(torch.softmax(out["logits"][h], dim=1)[:, 1].cpu())
            scores[h].append((out["logits"][h][:, 1] - out["logits"][h][:, 0]).cpu())
    return {"z": torch.cat(zs).numpy(), "probs": {h: torch.cat(v).numpy() for h, v in probs.items()},
            "scores": {h: torch.cat(v).numpy() for h, v in scores.items()}}


def head_metrics(y: dict, probs: dict, rows: np.ndarray) -> dict:
    """Balanced accuracy (at 0.5) and AUROC of each head on the labelled samples among ``rows``."""
    out = {}
    for h in HEADS:
        r = rows[y[h][rows] >= 0]
        if len(r) and len(np.unique(y[h][r])) == 2:
            out[h] = classification_metrics(y[h][r], probs[h][r], 0.5)
    return out


def cross_validate(tables: AtlasTables, fs: FeatureSet, cfg: TrainConfig, splits, device, seed: int | None = None,
                   log=None) -> dict:
    """Out-of-fold head metrics over the study folds (reference samples of the training pool). Each
    fold's model, and all preprocessing statistics, are fitted without any sample of the held-out
    fold's studies. Scored on the labels of ``y_eval``. ``phase_order`` is the within-study order of
    the phase score (``metrics.within_study_order``) over every sample of the held-out studies.
    ``predictions`` has the out-of-fold scores of those samples (reference or not; the metrics use the
    reference samples, ``phase_order`` all), in the columns of ``predictions.tsv``."""
    folds = sorted(int(k) for k in splits.loc[splits["split"] == 1, "cv_fold"].dropna().unique())
    probs = None
    for k in folds:
        full = build_training(tables, fs, cfg, splits)
        data = build_training(tables, fs, cfg, splits, fit_rows=full.fit_rows(k))
        if log:
            log(f"[cv] fold {k}: fit on {len(data.fit_rows(k))} samples, held out {len(data.val_rows(k))}")
        model, _ = fit(data, cfg, device, rows=data.fit_rows(k), seed=seed)
        enc = encode(model, data.X, data.C, device)
        if probs is None:
            probs = {h: np.full(len(data.ids), np.nan) for h in HEADS}
            held_probs = {h: np.full(len(data.ids), np.nan) for h in HEADS}
            phase_score = np.full(len(data.ids), np.nan)
            y = data.y_eval
        for h in HEADS:
            probs[h][data.val_rows(k)] = enc["probs"][h][data.val_rows(k)]
        held = np.where((data.split == 1) & (study_fold(data.splits) == k))[0]      # reference or not: none was fitted
        for h in HEADS:
            held_probs[h][held] = enc["probs"][h][held]
        phase_score[held] = enc["scores"]["phase"][held]
    pool = np.where((splits["split"].to_numpy() == 1) & splits["cv_fold"].notna().to_numpy())[0]
    scored = np.where(~np.isnan(phase_score))[0]
    pred = pd.DataFrame({"gse": data.splits["gse"].to_numpy(), "cv_fold": study_fold(data.splits),
                         "reference": data.splits["reference"].to_numpy(dtype=bool)}, index=data.ids)
    for h, (neg, pos) in HEADS.items():
        pred[f"{h}_label"] = data.labels[h].to_numpy()
        pred[f"{h}_scored"] = y[h] >= 0
        pred[f"p_{pos}"] = held_probs[h]
    pred["phase_logit"] = phase_score
    pred["phase_level"] = data.phase_level
    pred.index.name = "sample"
    pred = pred.iloc[scored].astype({"cv_fold": int})
    return {"metrics": head_metrics(y, probs, pool), "probs": probs, "predictions": pred,
            "phase_order": within_study_order(phase_score, data.phase_level, data.gse_codes, scored)}
