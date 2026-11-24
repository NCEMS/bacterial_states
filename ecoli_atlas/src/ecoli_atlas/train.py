"""Training of the atlas model. Three fits (``TrainConfig.fit``):

two_stage     (1) One block encoder per call, each fitted on the
              training samples that have a label for it, with a class-weighted cross-entropy.
              (2) With the blocks frozen, the residual encoder and the count decoder on all
              training samples, with the loss below without its head term.
one_pass      the same blocks and residual, all fitted together with the whole loss.
unstructured  one embedding without blocks, fitted with the whole loss.

    loss = count likelihood (negative binomial, mean over features; its mean includes the genomic DNA reads, ``models.count_log_mean``)
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
from .data import AtlasTables, TrainingData, label_weights, prepared, study_fold
from .features import FeatureSet
from .losses import class_weights, cos_dist, cross_study_control_pull, kl_normal, masked_mean, nb_nll, within_group_r2, within_study_push
from .metrics import classification_metrics, within_series_order, within_study_order
from .models import AtlasModel, OrdinalHead, build_model, count_log_mean
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


def replicate_sampler(data: TrainingData, rows: np.ndarray, max_size: int = 6) -> dict | None:
    """Replicate groups among ``rows`` with 2+ members that have a DNA estimate: {group code: rows} (at most
    ``max_size`` rows per group); None when there are fewer than 10 such groups."""
    code = data.extra.get("replicate_group")
    dna = data.extra.get("dna_estimate")
    if code is None or dna is None:
        return None
    rows = np.asarray(rows)
    rows = rows[(code[rows] >= 0) & np.isfinite(dna[rows])]
    groups = pd.Series(rows).groupby(code[rows]).apply(lambda s: s.to_numpy())
    groups = {int(k): v for k, v in groups.items() if len(v) >= 2}
    return groups if len(groups) >= 10 else None


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


def head_score(block_out: torch.Tensor) -> torch.Tensor:
    """The score of a head output: log-odds of the positive class (two classes), or the linear score
    of an ordinal head (its first cumulative logit, up to a constant)."""
    return block_out[:, 1] - block_out[:, 0] if block_out.shape[1] == 2 else block_out[:, 0]


def order_pair_sampler(data: TrainingData, rows: np.ndarray) -> dict | None:
    """The phase-order pairs whose two samples are both among ``rows``, with the probability of
    drawing each so that every (source, study) group counts the same. None without pairs."""
    pairs = data.order_pairs
    if pairs is None or not len(pairs["i"]):
        return None
    inside = np.zeros(len(data.ids), bool)
    inside[rows] = True
    keep = inside[pairs["i"]] & inside[pairs["j"]]
    if not keep.any():
        return None
    i, j, grp, src = pairs["i"][keep], pairs["j"][keep], pairs["group"][keep], pairs["source"][keep]
    _, inv, size = np.unique(grp, return_inverse=True, return_counts=True)
    p = 1.0 / size[inv]
    return {"i": i, "j": j, "p": p / p.sum(), "n_groups": int(len(size)),
            "n_pairs": {"levels": int((src == 0).sum()), "series": int((src == 1).sum())},
            "n_studies": {"levels": int(len(np.unique(grp[src == 0]))), "series": int(len(np.unique(grp[src == 1])))}}


def order_loss(score_i: torch.Tensor, score_j: torch.Tensor) -> torch.Tensor:
    """Pairwise ranking loss: log(1 + exp(-(s_j - s_i))) for pairs where j is the later sample."""
    return F.softplus(score_i - score_j).mean()


def fit_block(model: AtlasModel, data: TrainingData, head: str, cfg: TrainConfig, device, rows: np.ndarray, seed: int) -> list:
    """Fit one block encoder on the labelled samples among ``rows``, every study capped within each
    class (``head_weighting``). For the phase block with ``cfg.phase_order_weight`` > 0, each step
    also draws ``cfg.phase_order_batch`` ordered pairs among ``rows`` (labelled or not;
    ``order_pair_sampler``) and adds the weight times their ranking loss. Returns one row per epoch:
    the mean cross-entropy and, with the order term, the mean order loss."""
    hp = cfg.hparams
    sampler = order_pair_sampler(data, rows) if head == "phase" and cfg.phase_order_weight > 0 else None
    pool = rows
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
        total, total_order, n = 0.0, 0.0, 0
        for i in range(0, len(order), cfg.block_batch):
            rt = torch.as_tensor(order[i:i + cfg.block_batch])
            if len(rt) < 2:
                continue
            loss = loss_fn(block(model_input(data, rt, rows, cfg, rng).to(device))[1], y[rt].to(device), w[rt].to(device), cw)
            ce = float(loss.detach())
            if sampler is not None:
                k = rng.choice(len(sampler["p"]), size=cfg.phase_order_batch, p=sampler["p"])
                both = torch.as_tensor(np.concatenate([sampler["i"][k], sampler["j"][k]]))
                s = head_score(block(model_input(data, both, pool, cfg, rng).to(device))[1])
                lo = order_loss(s[:len(k)], s[len(k):])
                loss = loss + cfg.phase_order_weight * lo
                total_order += float(lo.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + ce, n + 1
        losses.append({"ce": total / max(n, 1)} | ({"order": total_order / max(n, 1)} if sampler is not None else {}))
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
    reps = replicate_sampler(data, rows) if cfg.dna_penalty_w > 0 else None
    if cfg.dna_penalty_w > 0 and reps is None and log:
        log("  DNA penalty left out: fewer than 10 biological replicate groups with a DNA estimate among the training rows")
    history = []
    model.train()
    for ep in range(epochs):
        sums: dict = {}
        nb_batches = 0
        rng = np.random.RandomState(seed + 7919 * (ep + 1))
        for ridx in study_blocked_batches(rows, data.gse_codes, cfg.batch, seed + ep, data.study_cap):
            rt = torch.as_tensor(ridx)
            out = model(model_input(data, rt, rows, cfg, rng).to(device), data.C[rt].to(device), train_blocks=train_blocks)
            log_dna = data.log_dna(rt)
            nll = nb_nll(data.Y[rt].to(device), count_log_mean(model, out["log_rate"], data.log_s[rt].to(device),
                                                               None if log_dna is None else log_dna.to(device)), model.log_theta)
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
            if reps is not None:
                keys = rng.choice(list(reps), min(cfg.dna_penalty_groups, len(reps)), replace=False)
                parts = [reps[k] if len(reps[k]) <= 6 else rng.choice(reps[k], 6, replace=False) for k in keys]
                pr = torch.as_tensor(np.concatenate(parts))
                pg = torch.as_tensor(np.repeat(np.arange(len(parts)), [len(p) for p in parts]), device=device)
                zr, _ = model.encode(model_input(data, pr, rows, cfg, rng).to(device), data.C[pr].to(device))
                dna_y = torch.as_tensor(data.extra["dna_estimate"][pr.numpy()], dtype=zr.dtype, device=device)
                terms["dna_r2"] = within_group_r2(zr, dna_y, pg)
                loss = loss + cfg.dna_penalty_w * terms["dna_r2"]
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


def gc_profile_bins(prof) -> int:
    """Number of GC bins of the GC term's window profile (0 without a profile)."""
    from .technical import GC_BINS
    return len(GC_BINS) if prof is not None and getattr(prof, "P", None) is not None else 0


def configure_gc_term(model, data: TrainingData) -> None:
    """Set the GC term's feature GC content and the scaling of the two GC covariates (``models.GCTerm``)."""
    from .technical import GC_FIT_MIN_LENGTH
    from .technical import GC_BINS
    prof = data.extra["feature_gc"]
    gc = prof.gc
    names, numeric = list(data.scaler.names), list(data.scaler.numeric)
    x = (gc - 0.5) * 10
    long = np.asarray(data.fs.is_mrna) & (np.asarray(data.fs.lengths) >= GC_FIT_MIN_LENGTH)
    k = [numeric.index(n) for n in ("gc_slope", "gc_curv")]
    model.gc_term.configure(gc, data.fs.is_mrna, float(x[long].mean() if long.any() else x.mean()),
                            (names.index("gc_slope"), names.index("gc_curv")), data.scaler.mu[k], data.scaler.sd[k], prof.P, GC_BINS)


def fit(data: TrainingData, cfg: TrainConfig, device, rows: np.ndarray | None = None, seed: int | None = None,
        epochs: int | None = None, log=None) -> tuple:
    """Train on ``rows`` (default: the training pool) in the way ``cfg.fit`` names. Returns the model
    (eval mode) and the history: one row per epoch of every stage (``stage`` is a head name,
    ``residual``, ``one_pass`` or ``unstructured``)."""
    seed = cfg.seed if seed is None else seed
    epochs = cfg.epochs if epochs is None else epochs
    rows = data.tr_rows if rows is None else np.asarray(rows)
    set_seed(seed)
    model = build_model(cfg.fit, data.fs.n, data.C.shape[1], cfg.hparams, torch.tensor(data.log_mean), cfg.phase_head,
                        dna=data.dna_rho is not None, gc=cfg.gc_offset, enc_cov=cfg.encoder_covariates,
                        gc_bins=gc_profile_bins(data.extra.get("feature_gc"))).to(device)
    if cfg.gc_offset:
        configure_gc_term(model, data)
        if cfg.gc_scale == "fixed":
            model.gc_term.scale.requires_grad_(False)          # held at 1 (no optimiser step reaches it without a gradient)
    history = []
    if cfg.fit == "two_stage":
        for h in HEADS:
            losses = fit_block(model, data, h, cfg, device, rows, seed)
            history += [{"stage": h, "epoch": ep, **v} for ep, v in enumerate(losses)]
            if log:
                log(f"  block {h}: {int((data.y_fit[h][rows] >= 0).sum())} labelled samples, loss {losses[0]['ce']:.4f} -> {losses[-1]['ce']:.4f}"
                    + (f"; order loss {losses[0]['order']:.4f} -> {losses[-1]['order']:.4f}" if "order" in losses[-1] else ""))
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
                   log=None, cache: dict | None = None) -> dict:
    """Out-of-fold head metrics over the study folds (reference samples of the training pool). Each
    fold's model, and all preprocessing statistics, are fitted without any sample of the held-out
    fold's studies. Scored on the labels of ``y_eval``. ``phase_order`` is the within-study order of
    the phase score (``metrics.within_study_order``) over every sample of the held-out studies.
    ``predictions`` has the out-of-fold scores of those samples (reference or not; the metrics use the
    reference samples, ``phase_order`` all), in the columns of ``predictions.tsv``. ``counts`` is the
    count fit (``evaluate.count_fit``) of those held-out samples, pooled over the folds and per fold: the
    out-of-fold NLL per feature. ``cache`` (a dict) keeps the training data of every fold for the next seed
    (``data.prepared``)."""
    from .evaluate import count_fit_values
    folds = sorted(int(k) for k in splits.loc[splits["split"] == 1, "cv_fold"].dropna().unique())
    probs = None
    ordered = splits.loc[tables.ids]
    nll, cors, per_fold = [], [], {}
    for k in folds:
        fit_rows = np.where((ordered["split"].to_numpy() == 1) & (study_fold(ordered) != k))[0]     # TrainingData.fit_rows(k)
        data = prepared(cache, ("fold", k), tables, fs, cfg, splits, fit_rows=fit_rows)
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
        if len(held):
            a, b = count_fit_values(model, data, held, device)
            nll.append(a), cors.append(b)
            per_fold[str(k)] = {"n": int(len(held)), "nll_per_feature": float(a.mean()),
                                "log_count_correlation_median": float(np.median(b))}
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
    out = {"metrics": head_metrics(y, probs, pool), "probs": probs, "predictions": pred,
           "phase_order": within_study_order(phase_score, data.phase_level, data.gse_codes, scored)}
    if nll:
        a, b = np.concatenate(nll), np.concatenate(cors)
        out["counts"] = {"n": int(len(a)), "nll_per_feature": float(a.mean()), "log_count_correlation_median": float(np.median(b)),
                         "folds": per_fold}
    if data.series is not None and (data.series >= 0).any():
        out["series_order"] = within_series_order(phase_score, data.series, data.series_x, data.splits["gse"].to_numpy(), scored)
    return out
