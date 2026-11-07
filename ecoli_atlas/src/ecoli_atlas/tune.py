"""Optuna hyper-parameter search (optional dependency: ``pip install ecoli-atlas[tune]``)."""
from __future__ import annotations

import logging

import torch

from .config import FactorHParams, ResidualHParams, TrainConfig
from .data import TrainingData
from .train_factor import factor_cv_score
from .train_residual import top_cv_score

EMB_RANGE = [4, 6, 8, 12, 16]
RES_CHOICES = [8, 16, 24, 32]


def _optuna():
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise SystemExit("hyper-parameter tuning needs optuna: pip install 'ecoli-atlas[tune]'") from e
    optuna.logging.set_verbosity(logging.WARNING)
    return optuna


def tune_factor(data: TrainingData, factor: str, cfg: TrainConfig, device, log=None) -> tuple[FactorHParams, float]:
    optuna = _optuna()

    def objective(trial):
        hp = FactorHParams(emb_dim=trial.suggest_categorical("emb_dim", EMB_RANGE),
                           width=trial.suggest_categorical("width", [128, 256]),
                           depth=trial.suggest_int("depth", 1, 3),
                           dropout=trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
                           lr=trial.suggest_float("lr", 5e-4, 3e-3, log=True))
        score = factor_cv_score(data, factor, hp, cfg, device)
        if score is None:
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.n_trials_factor)
    if log:
        log(f"[{factor}] pooled out-of-fold AUROC = {study.best_value:.3f}  {study.best_params}")
    return FactorHParams(**study.best_params), float(study.best_value)


def tune_residual(data: TrainingData, TB: torch.Tensor, cfg: TrainConfig, device, log=None) -> tuple[ResidualHParams, float]:
    optuna = _optuna()

    def objective(trial):
        hp = ResidualHParams(res_dim=trial.suggest_categorical("res_dim", RES_CHOICES),
                             width=trial.suggest_categorical("width", [128, 256]),
                             depth=trial.suggest_int("depth", 1, 3),
                             dropout=trial.suggest_categorical("dropout", [0.0, 0.1]),
                             lr=trial.suggest_float("lr", 5e-4, 3e-3, log=True),
                             push_w=trial.suggest_float("push_w", 0.0, 2.0),
                             ctrl_w=trial.suggest_float("ctrl_w", 0.0, 2.0))
        score = top_cv_score(data, TB, hp, cfg, device)
        if score is None:
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.n_trials_top)
    if log:
        log(f"[residual] objective = {study.best_value:.4f}  {study.best_params}")
    return ResidualHParams(**study.best_params), float(study.best_value)
