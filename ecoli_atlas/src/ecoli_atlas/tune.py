"""Hyper-parameter search (`ecoli_atlas tune`): Optuna over the network sizes, loss weights and epochs of the model,
scored out of fold on the study folds of a fixed ``splits.csv``, never on TEST.

Two objectives, one Pareto front: the out-of-fold count NLL per feature (``train.cross_validate``, ``counts``;
minimised) and the mean out-of-fold AUROC of the heads (``metrics.mean_auroc``, the score that ranks the seeds of a
sweep; maximised). In the two-stage fit the heads depend on the block encoders only and the NLL mostly on the residual
encoder and the decoder, so the front is close to the best of each.

Everything else (the fit, the phase head, the block sizes, the inputs, the penalties, the caps, the data gates) is
the TrainConfig given on the command line and stays the same in every trial; it is recorded in ``study_config.json``
and a run in the same directory with a different one is refused. The training data of every fold is built once and
shared by the trials (nothing in it depends on these parameters).

The study lives in ``<out>/study.journal`` (Optuna's journal file storage, which works on a network file system), so
it resumes and several workers can share it. After every trial ``trials.tsv`` (all trials), ``pareto.tsv`` (the
front) and ``best/trial_<n>.json`` (each point of the front in the format of `train --hparams`; ``hparams/`` has every
completed trial) are rewritten.
"""
from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HParams, TrainConfig
from .metrics import mean_auroc

STUDY_NAME = "atlas_hpo"
JOURNAL_FILE = "study.journal"
STUDY_CONFIG_FILE = "study_config.json"
TRIALS_FILE = "trials.tsv"
PARETO_FILE = "pareto.tsv"
BEST_DIR = "best"            # the Pareto front, in the format of `train --hparams`
HPARAMS_DIR = "hparams"      # every completed trial, the same format
DIRECTIONS = ("minimize", "maximize")          # out-of-fold NLL, mean out-of-fold AUROC
OBJECTIVES = ("cv_nll", "cv_mean_auroc")

# The search space, one entry per parameter: (kind, values). kind: "cat" (choices), "int" (low, high, step),
# "float" (low, high), "log" (low, high on a log scale). Names are HParams fields or TrainConfig training fields
# (cli.HPARAMS_TRAINING_FIELDS).
SPACE = {
    # residual encoder and count decoder (stage 2)
    "res_dim": ("cat", (8, 16, 24, 32, 48)),
    "enc_width": ("cat", (256, 512, 1024)),
    "enc_depth": ("int", (1, 3, 1)),
    "dec_width": ("cat", (256, 512, 1024)),
    "dec_depth": ("int", (1, 2, 1)),
    "dropout": ("float", (0.0, 0.3)),
    "lr": ("log", (3e-4, 3e-3)),
    "weight_decay": ("log", (1e-7, 1e-3)),
    "kl_w": ("log", (1e-4, 1e-2)),
    "push_w": ("float", (0.0, 1.0)),
    "ctrl_w": ("float", (0.0, 2.0)),
    "push_margin": ("float", (0.25, 1.0)),
    "cov_rank": ("cat", (4, 8, 16)),
    "epochs": ("int", (100, 300, 50)),
    "batch": ("cat", (256, 512, 1024)),
    # block encoders (stage 1)
    "block_width": ("cat", (128, 256, 512)),
    "block_depth": ("int", (1, 3, 1)),
    "block_dropout": ("float", (0.0, 0.5)),
    "block_lr": ("log", (3e-4, 3e-3)),
    "block_epochs": ("int", (10, 60, 5)),
}
TRAINING_FIELDS = ("epochs", "block_epochs", "batch", "block_batch")
# what may differ between trials of one study; every other TrainConfig value is fixed by study_config.json
_VARIABLE = set(SPACE) | {"seed"}


def current_values(cfg: TrainConfig) -> dict:
    """The values of the SPACE parameters in ``cfg`` (the baseline trial)."""
    hp = asdict(cfg.hparams)
    return {k: (getattr(cfg, k) if k in TRAINING_FIELDS else hp[k]) for k in SPACE}


def suggest(trial) -> dict:
    out = {}
    for name, (kind, v) in SPACE.items():
        if kind == "cat":
            out[name] = trial.suggest_categorical(name, list(v))
        elif kind == "int":
            out[name] = trial.suggest_int(name, v[0], v[1], step=v[2])
        elif kind == "float":
            out[name] = trial.suggest_float(name, v[0], v[1])
        elif kind == "log":
            out[name] = trial.suggest_float(name, v[0], v[1], log=True)
        else:
            raise ValueError(f"SPACE[{name!r}]: unknown kind {kind!r}")
    return out


def trial_config(cfg: TrainConfig, params: dict, max_epochs: int | None = None) -> TrainConfig:
    """``cfg`` with a trial's parameters; ``max_epochs`` caps both epoch counts (smoke runs only)."""
    from dataclasses import replace
    c = deepcopy(cfg)
    c.hparams = replace(c.hparams, **{k: v for k, v in params.items() if k not in TRAINING_FIELDS})
    for k in TRAINING_FIELDS:
        if k in params:
            setattr(c, k, int(params[k]))
    if max_epochs:
        c.epochs, c.block_epochs = min(c.epochs, max_epochs), min(c.block_epochs, max_epochs)
    return c


def hparams_file(cfg: TrainConfig, params: dict) -> dict:
    """The full `train --hparams` object of a trial: every HParams field and the training fields."""
    c = trial_config(cfg, params)
    return asdict(c.hparams) | {k: getattr(c, k) for k in TRAINING_FIELDS}


def fixed_config(cfg: TrainConfig) -> dict:
    """The part of ``cfg`` that every trial of a study shares."""
    d = cfg.to_dict()
    d["hparams"] = {k: v for k, v in asdict(cfg.hparams).items() if k not in _VARIABLE}
    return {k: v for k, v in d.items() if k not in _VARIABLE}


def _storage(out: Path):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = str(out / JOURNAL_FILE)
    return optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def _check_study_config(out: Path, cfg: TrainConfig, splits_md5: str, max_epochs: int | None) -> None:
    rec = {"fixed_config": fixed_config(cfg), "splits_md5": splits_md5, "max_epochs": max_epochs,
           "space": {k: [kind, list(v)] for k, (kind, v) in SPACE.items()}, "directions": list(DIRECTIONS),
           "objectives": list(OBJECTIVES)}
    rec = json.loads(json.dumps(rec, default=str))
    path = out / STUDY_CONFIG_FILE
    if path.exists():
        old = json.loads(path.read_text())
        diff = sorted(k for k in rec if old.get(k) != rec[k])
        if diff:
            raise SystemExit(f"{out}: the study there was made with a different {', '.join(diff)}; "
                             f"use another --out (see {path})")
        return
    _atomic_write(path, json.dumps(rec, indent=2))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_tables(study, out: Path, cfg: TrainConfig) -> pd.DataFrame:
    """trials.tsv, pareto.tsv and best/trial_<n>.json from the study as it stands (all workers' trials)."""
    import optuna
    rows = []
    for t in study.get_trials(deepcopy=False):
        r = {"trial": t.number, "state": t.state.name,
             "duration_s": t.duration.total_seconds() if t.duration else np.nan}
        if t.values is not None:
            r |= dict(zip(OBJECTIVES, t.values))
        r |= {f"p_{k}": v for k, v in t.params.items()}
        r |= {k: v for k, v in t.user_attrs.items() if not isinstance(v, (dict, list))}
        rows.append(r)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    front = {t.number for t in study.best_trials} if done else set()
    table["pareto"] = table["trial"].isin(front)
    _atomic_write(out / TRIALS_FILE, table.to_csv(sep="\t", index=False, float_format="%.6g"))
    pareto = table[table["pareto"]].sort_values("cv_nll") if front else table.iloc[:0]
    _atomic_write(out / PARETO_FILE, pareto.to_csv(sep="\t", index=False, float_format="%.6g"))
    best, every = out / BEST_DIR, out / HPARAMS_DIR
    best.mkdir(exist_ok=True)
    every.mkdir(exist_ok=True)
    for t in done:
        text = None
        for d in (every, best) if t.number in front else (every,):
            f = d / f"trial_{t.number}.json"
            if not f.exists():
                text = text or json.dumps(hparams_file(cfg, t.params), indent=2)
                _atomic_write(f, text)
    for f in best.glob("trial_*.json"):
        if int(f.stem.split("_")[1]) not in front:
            f.unlink(missing_ok=True)
    return table


def _n_trials(study) -> int:
    import optuna
    S = optuna.trial.TrialState
    return len(study.get_trials(deepcopy=False, states=(S.COMPLETE, S.RUNNING, S.FAIL, S.PRUNED)))


class Stop(Exception):
    """SIGTERM (the SLURM time limit, scancel): the running trial is marked failed and the worker stops."""


def _on_sigterm(signum, frame):
    raise Stop(f"signal {signum}")


def fail_running(study, log=print) -> int:
    """Mark the trials left RUNNING (a worker killed without SIGTERM) as failed. Only when no other worker runs."""
    import optuna
    stale = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    if stale:
        log(f"[tune] marked {len(stale)} trials left running as failed: {', '.join(str(t.number) for t in stale)}")
    return len(stale)


def run_tune(tables, fs, cfg: TrainConfig, splits: pd.DataFrame, device, out, n_trials: int, max_total: int | None = None,
             trial_seeds=(1,), sampler_seed: int | None = None, max_epochs: int | None = None, reset_running: bool = False,
             log=print) -> dict:
    """Run up to ``n_trials`` trials in this process, stopping once the study holds ``max_total`` trials (over all
    workers and earlier runs; running and failed trials count). Returns a summary."""
    try:
        import optuna
    except ImportError:
        raise SystemExit("`tune` needs optuna: pip install 'ecoli_atlas[tune]' (or mamba install -c conda-forge optuna)") from None
    import hashlib
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5(splits.sort_index().to_csv().encode()).hexdigest()
    _check_study_config(out, cfg, md5, max_epochs)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=sampler_seed, n_startup_trials=16)
    study = optuna.create_study(study_name=STUDY_NAME, storage=_storage(out), directions=list(DIRECTIONS),
                                sampler=sampler, load_if_exists=True)
    if reset_running:
        fail_running(study, log)
    if not study.get_trials(deepcopy=False):
        study.enqueue_trial(current_values(cfg), user_attrs={"baseline": True})
        log(f"[tune] new study in {out}; trial 0 is the current configuration")
    else:
        log(f"[tune] resuming the study in {out} ({len(study.trials)} trials so far)")
    import signal
    previous = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        ran = _trials(study, tables, fs, cfg, splits, device, out, n_trials, max_total, trial_seeds, max_epochs, log)
    finally:
        signal.signal(signal.SIGTERM, previous)
    table = write_tables(study, out, cfg)
    n_front = int(table["pareto"].sum()) if not table.empty else 0
    log(f"[tune] {ran} trials run here; the study holds {len(table)} trials, {n_front} on the Pareto front "
        f"({out / PARETO_FILE})")
    return {"ran": ran, "trials": len(table), "pareto": n_front}


def _trials(study, tables, fs, cfg, splits, device, out, n_trials, max_total, trial_seeds, max_epochs, log) -> int:
    import optuna
    import torch
    from .train import cross_validate
    cache, ran = {}, 0
    for _ in range(n_trials):
        if max_total is not None and _n_trials(study) >= max_total:
            log(f"[tune] the study holds {max_total} trials (--max-total-trials): stopping")
            break
        trial = study.ask()
        params = suggest(trial)
        c = trial_config(cfg, params, max_epochs)
        t0 = time.time()
        log(f"[tune] trial {trial.number}: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()))
        try:
            res = [cross_validate(tables, fs, c, splits, device, seed=s, cache=cache) for s in trial_seeds]
            nll = float(np.mean([r["counts"]["nll_per_feature"] for r in res]))
            auc = float(np.mean([mean_auroc(r["metrics"]) for r in res]))
            if not (math.isfinite(nll) and math.isfinite(auc)):
                raise FloatingPointError(f"objective not finite (NLL {nll}, AUROC {auc})")
        except Stop as exc:
            trial.set_user_attr("error", f"stopped ({exc})")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            log(f"[tune] stopped by {exc} during trial {trial.number} (marked failed)")
            break
        except (RuntimeError, FloatingPointError, ValueError) as exc:          # CUDA OOM is a RuntimeError
            trial.set_user_attr("error", f"{type(exc).__name__}: {str(exc)[:300]}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            log(f"[tune] trial {trial.number} failed: {type(exc).__name__}: {str(exc)[:200]}")
            if device is not None and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            write_tables(study, out, cfg)
            ran += 1
            continue
        attrs = {"seeds": ",".join(map(str, trial_seeds)), "wall_s": round(time.time() - t0, 1),
                 "cv_log_count_correlation": float(np.mean([r["counts"]["log_count_correlation_median"] for r in res]))}
        for h in res[0]["metrics"]:
            for m in ("auroc", "balanced_acc"):
                attrs[f"cv_{h}_{m}"] = float(np.mean([r["metrics"][h][m] for r in res if h in r["metrics"]]))
        attrs["cv_phase_order"] = float(np.mean([r["phase_order"]["share_in_order_mean_over_studies"] for r in res]))
        if all("series_order" in r for r in res):
            attrs["cv_series_order"] = float(np.mean([r["series_order"]["share_in_order"] for r in res]))
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        study.tell(trial, [nll, auc])
        ran += 1
        log(f"[tune] trial {trial.number}: out-of-fold NLL {nll:.4f}, mean AUROC {auc:.4f} "
            f"(phase {attrs.get('cv_phase_auroc', float('nan')):.4f}), {attrs['wall_s']:.0f} s")
        write_tables(study, out, cfg)
    return ran
