"""Training configuration and hyper-parameters, with round-trips to/from ``manifest.json``."""
from __future__ import annotations

import json

import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_FACTORS = ("growth_phase", "medium_type", "is_lab_strain")
# positive class per factor (probability that the operating threshold is applied to)
POSITIVE_CLASS = {"growth_phase": "stationary", "medium_type": "defined", "is_lab_strain": "True"}
PHASE_ORDER = ("exponential", "stationary")


@dataclass
class FactorHParams:
    emb_dim: int = 8
    width: int = 256
    depth: int = 2
    dropout: float = 0.1
    lr: float = 1e-3


@dataclass
class ResidualHParams:
    res_dim: int = 24
    width: int = 128
    depth: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    push_w: float = 0.5
    ctrl_w: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 77
    factors: tuple = DEFAULT_FACTORS
    # study-level splits
    test_frac: float = 0.10
    n_folds: int = 3
    n_candidates: int = 500
    fold_study_weight: float = 1.0
    min_class_n: int = 10
    withheld_study: str | None = "GSE224463"       # proof-of-concept study, excluded from everything
    case_studies: tuple = ("GSE128997", "GSE189154", "GSE78756", "GSE80451")  # forced into TEST
    # training
    factor_epochs: int = 40
    top_epochs: int = 60
    factor_batch: int = 256
    top_batch: int = 512
    push_margin: float = 0.5
    knn_k: int = 15
    # tuning
    tune: bool = False
    n_trials_factor: int = 40
    n_trials_top: int = 40
    # hyper-parameters (filled by tuning or from a manifest)
    factor_hparams: dict = field(default_factory=dict)      # factor -> FactorHParams
    residual_hparams: ResidualHParams = field(default_factory=ResidualHParams)
    # thresholds
    thresh_grid: tuple = tuple(float(x) for x in np.linspace(0.01, 0.99, 197))  # as in the notebook
    apply_thresholds: tuple = ()

    @classmethod
    def from_manifest(cls, path: str | Path, **overrides) -> "TrainConfig":
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        hp = m.get("hyperparameters", {})
        sc = m.get("split_config", {})
        cfg = cls(
            seed=int(m.get("seed", 77)),
            factors=tuple(m.get("factor_order", DEFAULT_FACTORS)),
            test_frac=float(sc.get("test_frac", 0.10)),
            n_folds=int(sc.get("n_folds", 3)),
            fold_study_weight=float(sc.get("fold_study_weight", 1.0)),
            min_class_n=int(sc.get("min_class_n", 10)),
            withheld_study=sc.get("withheld_study"),
            case_studies=tuple(sc.get("case_studies", ())),
            factor_epochs=int(hp.get("factor_epochs", 40)),
            top_epochs=int(hp.get("top_epochs", 60)),
            n_trials_factor=int(hp.get("n_trials_factor", 40)),
            n_trials_top=int(hp.get("n_trials_top", 40)),
            factor_hparams={f: FactorHParams(**v) for f, v in hp.get("factor_encoders", {}).items()},
            residual_hparams=ResidualHParams(**hp["residual_ae"]) if "residual_ae" in hp else ResidualHParams(),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def hyperparameters_dict(self) -> dict:
        return {
            "factor_encoders": {f: asdict(h) for f, h in self.factor_hparams.items()},
            "residual_ae": asdict(self.residual_hparams),
            "factor_epochs": self.factor_epochs,
            "top_epochs": self.top_epochs,
            "n_trials_factor": self.n_trials_factor if self.tune else 0,
            "n_trials_top": self.n_trials_top if self.tune else 0,
        }

    def split_config_dict(self) -> dict:
        return {
            "test_frac": self.test_frac, "n_folds": self.n_folds, "split_seed": self.seed,
            "fold_study_weight": self.fold_study_weight, "min_class_n": self.min_class_n,
            "withheld_study": self.withheld_study, "case_studies": sorted(self.case_studies),
        }
