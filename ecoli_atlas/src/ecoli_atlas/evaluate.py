"""Evaluation of a trained model, on studies it was not trained on wherever that is possible.

    heads              balanced accuracy and AUROC of the calls on TEST studies; the phase call is
                       scored only where the growth phase is stated in the study's text
    knn_agreement      share of a TEST sample's nearest training neighbours with its label
    same_study_rate    share of a sample's nearest neighbours from its own study (lower = better mixing)
    covariate_r2       how well each technical covariate can be predicted from z (ridge fitted on the
                       training pool, R2 on TEST; lower = less technical signal in the embedding)
    counts             negative binomial deviance term per feature, and the correlation of observed
                       with expected log counts per sample
    phase_lfc          per study with both phase classes: correlation over mRNA features of the observed
                       stationary-vs-exponential log2 fold change with the one the decoder gives at the
                       reference covariates
    medium_check       the medium head on the media between the two classes (minimal medium with a
                       hydrolysate, defined rich media, natural fluids)
    phase_check        the phase head on untreated samples in the phases between the two classes
                       (late_stationary is in neither a class nor this check)
    phase_order        among pairs of untreated samples of one study at different phase levels
                       (exponential < transition < stationary), the share whose phase scores are
                       in that order; stated and inferred phases alike
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from .config import HEADS, TrainConfig
from .data import COMPOSITION_NAMES, QC_NUMERIC, TrainingData
from .labels import MEDIUM_CHECK, PHASE_CHECK, medium_check_group, phase_check_group
from .losses import masked_mean, nb_nll
from .metrics import knn_label_agreement, same_study_rate, within_series_order, within_study_order
from .models import count_log_mean
from .train import encode, head_metrics

MIN_PER_CLASS = 2      # samples of each phase class a study needs for the fold-change check


@torch.no_grad()
def count_fit(model, data: TrainingData, rows: np.ndarray, device, chunk: int = 1024) -> dict:
    nll, cors = count_fit_values(model, data, rows, device, chunk)
    return {"n": int(len(rows)), "nll_per_feature": float(nll.mean()), "log_count_correlation_median": float(np.median(cors))}


@torch.no_grad()
def count_fit_values(model, data: TrainingData, rows: np.ndarray, device, chunk: int = 1024) -> tuple:
    """Per sample of ``rows``: the count NLL per feature (over the genes the sample has) and the correlation of the
    observed with the expected log counts."""
    model.eval()
    nll, cors = [], []
    for i in range(0, len(rows), chunk):
        rt = torch.as_tensor(rows[i:i + chunk])
        out = model(data.X[rt].to(device), data.C[rt].to(device), sample=False)
        log_dna = data.log_dna(rt)
        log_mean = count_log_mean(model, out["log_rate"], data.log_s[rt].to(device), None if log_dna is None else log_dna.to(device))
        y = data.Y[rt].to(device)
        keep = torch.ones_like(y, dtype=torch.bool) if data.M is None else ~data.M[rt].to(device)      # the genes the sample has
        nll.append(masked_mean(nb_nll(y, log_mean, model.log_theta), keep, dim=1).cpu().numpy())
        a, b = torch.log1p(y), torch.log1p(torch.exp(log_mean.clamp(max=30)))
        a, b = (a - masked_mean(a, keep, dim=1)[:, None]) * keep, (b - masked_mean(b, keep, dim=1)[:, None]) * keep
        cors.append(((a * b).sum(1) / torch.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)).clamp_min(1e-12)).cpu().numpy())
    return np.concatenate(nll), np.concatenate(cors)


@torch.no_grad()
def phase_fold_changes(model, data: TrainingData, z: np.ndarray, device) -> pd.DataFrame:
    """One row per study with at least MIN_PER_CLASS samples of each phase class."""
    y, gse = data.y["phase"], data.splits["gse"].to_numpy()
    logx = data.X.numpy() * data.zscore.sd + data.zscore.mu           # log2 normalised expression
    c0 = torch.zeros(1, data.C.shape[1], device=device)               # reference covariates: the training means
    mrna = data.fs.is_mrna
    study_split = data.splits.groupby("gse")["split"].max()      # 3 for a TEST study, also for its projected samples
    rows = []
    for g in pd.unique(gse[y >= 0]):
        e, s = np.where((gse == g) & (y == 0))[0], np.where((gse == g) & (y == 1))[0]
        if len(e) < MIN_PER_CLASS or len(s) < MIN_PER_CLASS:
            continue
        obs = logx[s].mean(0) - logx[e].mean(0)
        dec = lambda r: model.decode(torch.tensor(z[r], device=device), c0.expand(len(r), -1)).mean(0).cpu().numpy()  # noqa: E731
        pred = (dec(s) - dec(e)) / np.log(2.0)
        rows.append({"gse": g, "split": int(study_split[g]), "n_exponential": len(e), "n_stationary": len(s),
                     "pearson": float(np.corrcoef(obs[mrna], pred[mrna])[0, 1])})
    return pd.DataFrame(rows, columns=["gse", "split", "n_exponential", "n_stationary", "pearson"])


def covariate_r2(data: TrainingData, z: np.ndarray) -> dict:
    tr, te = data.tr_rows, data.te_rows
    C = data.C.numpy()
    out = {}
    for name in data.scaler.names:
        if name not in COMPOSITION_NAMES + QC_NUMERIC:
            continue
        j = data.scaler.names.index(name)
        if np.var(C[te, j]) == 0 or np.var(C[tr, j]) == 0:
            continue
        pred = Ridge(alpha=1.0).fit(z[tr], C[tr, j]).predict(z[te])
        out[name] = float(1.0 - ((C[te, j] - pred) ** 2).mean() / np.var(C[te, j]))
    return out


def evaluate(model, data: TrainingData, cfg: TrainConfig, device) -> dict:
    """Returns {"report": nested dict, "embedding": frame, "predictions": frame, "phase_lfc": frame}."""
    enc = encode(model, data.X, data.C, device)
    z, probs = enc["z"], enc["probs"]
    tr, te = data.tr_rows, data.te_rows
    report = {"n_samples": {"training_pool": int(len(tr)), "training_reference": int(data.reference[tr].sum()), "test": int(len(te))},
              "heads": {"test": head_metrics(data.y_eval, probs, te), "training_pool": head_metrics(data.y_eval, probs, tr)}}
    knn = {}
    for h in HEADS:
        fit_r, val_r = tr[data.y[h][tr] >= 0], te[data.y_eval[h][te] >= 0]
        if len(fit_r) >= cfg.knn_k and len(val_r) and len(np.unique(data.y_eval[h][val_r])) == 2:
            knn[h] = knn_label_agreement(z[fit_r], data.y[h][fit_r], data.gse_codes[fit_r], z[val_r], data.y_eval[h][val_r],
                                         data.gse_codes[val_r], cfg.knn_k)
    report["knn_agreement"] = knn
    ident = data.identity
    used = np.concatenate([tr, te])
    if len(used) > 25:
        rate = same_study_rate(z[used], data.gse_codes[used], ident["cond_id"][used], ident["is_annot"][used])
        report["same_study_rate"] = {"n": int(len(rate)), "mean": float(rate.mean()) if len(rate) else None}
    report["covariate_r2"] = covariate_r2(data, z) if len(te) > 1 else {}
    report["counts"] = {"test": count_fit(model, data, te, device) if len(te) else {}, "training_pool": count_fit(model, data, tr, device)}
    lfc = phase_fold_changes(model, data, z, device)
    report["phase_lfc"] = {name: {"n_studies": int(len(sub)), "median_pearson": float(sub["pearson"].median()) if len(sub) else None}
                           for name, sub in (("test", lfc[lfc["split"] == 3]), ("training_pool", lfc[lfc["split"] == 1]))}
    check = phase_check_group(data.meta)
    p = pd.Series(probs["phase"], index=data.ids)
    report["phase_check"] = {g: {"n": int((check == g).sum()), "median_p_stationary": float(p[(check == g).to_numpy()].median())}
                             for g in PHASE_CHECK if (check == g).any()}
    report["phase_order"] = {name: within_study_order(enc["scores"]["phase"], data.phase_level, data.gse_codes, r)
                             for name, r in (("test", np.where(data.splits.groupby("gse")["split"].transform("max").to_numpy() == 3)[0]),
                                             ("training_pool", tr))}
    if data.series is not None and (data.series >= 0).any():
        # time order of the phase score along culture series (time-course table): TEST, withheld and training studies
        names = data.splits["gse"].to_numpy()
        report["series_order"] = {name: within_series_order(enc["scores"]["phase"], data.series, data.series_x, names, np.where(data.split == code)[0])
                                  for name, code in (("test", 3), ("withheld", 0), ("training_pool", 1))}
    mcheck = medium_check_group(data.meta)
    pm = pd.Series(probs["medium"], index=data.ids)
    report["medium_check"] = {g: {"n": int((mcheck == g).sum()), "n_studies": int(data.meta.loc[(mcheck == g).to_numpy(), "GSE"].nunique()),
                                  "median_p_minimal": float(pm[(mcheck == g).to_numpy()].median()),
                                  "share_called_minimal": float((pm[(mcheck == g).to_numpy()] >= 0.5).mean())}
                              for g in MEDIUM_CHECK if (mcheck == g).any()}
    emb = pd.DataFrame(z, index=data.ids, columns=[f"z{i + 1}" for i in range(z.shape[1])])
    emb.index.name = "sample"
    pred = pd.DataFrame({"gse": data.splits["gse"].to_numpy(), "split": data.split}, index=data.ids)
    for h, (neg, pos) in HEADS.items():
        pred[f"{h}_label"] = data.labels[h].to_numpy()
        pred[f"{h}_scored"] = data.y_eval[h] >= 0
        pred[f"p_{pos}"] = probs[h]
    pred["phase_logit"] = enc["scores"]["phase"]        # log-odds of stationary: graded where p_stationary is 0 or 1
    pred.index.name = "sample"
    return {"report": report, "embedding": emb, "predictions": pred, "phase_lfc": lfc}
