"""Saving and loading a trained model bundle.

Layout of a models directory (identical to what the original notebook wrote, so old bundles load):

    <factor>_encoder.pt   {"state_dict", "arch", "factor", "classes", "threshold", "positive_class"}
    residual_ae.pt        {"state_dict", "arch"}
    preprocessing.pt      {"genes", "zscore_mu", "zscore_sd", "tb_mu", "tb_sd", "factor_order",
                           "factor_emb_dims", "res_dim", "top_emb_dim", "block_layout", "factor_classes",
                           "factor_class_dtype", "factor_thresholds", "positive_class_index", "seed", ...}
    manifest.json         human-readable record of hyper-parameters, metrics and environment
    splits.csv            study-level train / test assignment used for training

New bundles store arrays as tensors so that ``torch.load(weights_only=True)`` works; old bundles
contain numpy arrays and are loaded with the unsafe fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import FactorEncoder, ResidualAE
from .preprocess import ZScore

SCHEMA_VERSION = 1
PREPROCESSING_FILE = "preprocessing.pt"
RESIDUAL_FILE = "residual_ae.pt"
MANIFEST_FILE = "manifest.json"
SPLITS_FILE = "splits.csv"


def encoder_file(factor: str) -> str:
    return f"{factor}_encoder.pt"


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # legacy bundles hold numpy objects
        return torch.load(path, map_location="cpu", weights_only=False)


def _as_array(v) -> np.ndarray:
    if torch.is_tensor(v):
        v = v.numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def _cpu_state(m: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


@dataclass
class FactorArtifact:
    name: str
    model: FactorEncoder
    classes: list
    threshold: float | None = None
    positive_class: str | None = None

    @property
    def emb_dim(self) -> int:
        return self.model.arch()["emb_dim"]


@dataclass
class ModelBundle:
    factors: dict               # factor name -> FactorArtifact, in factor_order
    residual: ResidualAE
    prep: dict                  # contents of preprocessing.pt with arrays as numpy
    manifest: dict | None = None
    source_dir: Path | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # ---- convenience accessors -------------------------------------------------------------
    @property
    def genes(self) -> list:
        return list(self.prep["genes"])

    @property
    def factor_order(self) -> list:
        return list(self.prep["factor_order"])

    @property
    def block_layout(self) -> list:
        return list(self.prep["block_layout"])

    @property
    def zscore(self) -> ZScore:
        return ZScore(self.prep["zscore_mu"], self.prep["zscore_sd"])

    @property
    def tb_mu(self) -> np.ndarray:
        return self.prep["tb_mu"]

    @property
    def tb_sd(self) -> np.ndarray:
        return self.prep["tb_sd"]

    @property
    def embedding_dim(self) -> int:
        return int(self.block_layout[-1]["end"])

    def embedding_columns(self) -> list:
        cols = []
        for b in self.block_layout:
            prefix = "residual" if b["kind"] == "residual" else b["block"]
            cols += [f"{prefix}[{i}]" for i in range(b["end"] - b["start"])]
        return cols

    def layout_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.block_layout)[["block", "kind", "start", "end"]].copy()
        df["width"] = df["end"] - df["start"]
        return df

    def to(self, device) -> "ModelBundle":
        self.device = torch.device(device)
        for fa in self.factors.values():
            fa.model.to(self.device).eval()
        self.residual.to(self.device).eval()
        return self

    # ---- persistence ------------------------------------------------------------------------
    @classmethod
    def load(cls, models_dir: str | Path, device="cpu") -> "ModelBundle":
        d = Path(models_dir)
        prep = dict(_load_pt(d / PREPROCESSING_FILE))
        for k in ("zscore_mu", "zscore_sd", "tb_mu", "tb_sd"):
            prep[k] = _as_array(prep[k])
        prep["genes"] = [str(g) for g in prep["genes"]]
        factors = {}
        for f in prep["factor_order"]:
            ck = _load_pt(d / encoder_file(f))
            m = FactorEncoder.from_arch(ck["arch"])
            m.load_state_dict(ck["state_dict"])
            thr = ck.get("threshold")
            factors[f] = FactorArtifact(f, m, [str(c) for c in ck["classes"]],
                                        None if thr is None else float(thr), ck.get("positive_class"))
        ck = _load_pt(d / RESIDUAL_FILE)
        residual = ResidualAE.from_arch(ck["arch"])
        residual.load_state_dict(ck["state_dict"])
        manifest = None
        if (d / MANIFEST_FILE).exists():
            with open(d / MANIFEST_FILE, encoding="utf-8") as fh:
                manifest = json.load(fh)
        bundle = cls(factors, residual, prep, manifest, d)
        return bundle.to(device)

    def save(self, models_dir: str | Path, splits: pd.DataFrame | None = None) -> Path:
        d = Path(models_dir)
        d.mkdir(parents=True, exist_ok=True)
        for f in self.factor_order:
            fa = self.factors[f]
            torch.save({"state_dict": _cpu_state(fa.model), "arch": fa.model.arch(), "factor": f,
                        "classes": list(fa.classes),
                        "threshold": None if fa.threshold is None else float(fa.threshold),
                        "positive_class": fa.positive_class}, d / encoder_file(f))
        torch.save({"state_dict": _cpu_state(self.residual), "arch": self.residual.arch()}, d / RESIDUAL_FILE)
        prep = dict(self.prep)
        for k in ("zscore_mu", "zscore_sd", "tb_mu", "tb_sd"):
            prep[k] = torch.tensor(np.asarray(prep[k], dtype=np.float32))
        prep["schema_version"] = SCHEMA_VERSION
        torch.save(prep, d / PREPROCESSING_FILE)
        if self.manifest is not None:
            with open(d / MANIFEST_FILE, "w", encoding="utf-8") as fh:
                json.dump(self.manifest, fh, indent=2, default=str)
        if splits is not None:
            splits.to_csv(d / SPLITS_FILE)
        return d


def load_splits(models_dir: str | Path) -> pd.DataFrame | None:
    p = Path(models_dir) / SPLITS_FILE
    return pd.read_csv(p, index_col=0) if p.exists() else None
