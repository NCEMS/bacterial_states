"""The model bundle: what `train` writes and `embed` / `evaluate` read.

    model.pt            weights (a state dict of tensors)
    preprocessing.pt    tensors: reference profile, z-score and covariate statistics, feature means, and the
                        low end of every feature's expression for the missing-gene calls
    manifest.json       features, covariates, hyper-parameters, split settings, metrics, versions
    splits.csv          study, split and fold of every training-table sample

Both .pt files hold tensors only, so ``torch.load(weights_only=True)`` reads them.
"""
from __future__ import annotations

import datetime
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import HEADS, HParams, TrainConfig
from .data import label_weights, CovariateScaler, TrainingData
from .features import FeatureSet
from .models import build_model
from .technical import GC_BINS
from .preprocess import ZScore

SCHEMA_VERSION = 2
MODEL_FILE, PREPROCESSING_FILE, MANIFEST_FILE, SPLITS_FILE = "model.pt", "preprocessing.pt", "manifest.json", "splits.csv"


def embedding_blocks(hp: HParams, fit: str) -> list:
    """Which columns of the embedding belong to which block (0-based, end exclusive)."""
    if fit == "unstructured":
        return [{"block": "z", "start": 0, "end": hp.z_dim}]
    out, start = [], 0
    for name, dim in [(h, hp.block_dims[h]) for h in HEADS] + [("residual", hp.res_dim)]:
        out.append({"block": name, "start": start, "end": start + dim})
        start += dim
    return out


def environment(device) -> dict:
    import pandas
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__, "device": str(device)}


@dataclass
class ModelBundle:
    model: torch.nn.Module
    fs: FeatureSet
    zscore: ZScore
    scaler: CovariateScaler
    ref_log: np.ndarray
    manifest: dict
    low: np.ndarray | None = None      # low end of every feature's normalised count; None = a bundle without missing-gene calls
    low_pool: np.ndarray | None = None # the same for counts summed over a group of samples; None = no pooled calls

    @property
    def config(self) -> TrainConfig:
        cfg = TrainConfig.from_dict(self.manifest["config"])
        for key in ("dna_decoder", "dna_inputs", "gc_offset", "clean_inputs", "qc_module", "gc_size_factors"):   # bundles made before these
            if key not in self.manifest["config"]:
                setattr(cfg, key, False)
        if "covariates" not in self.manifest["config"]:    # a bundle made before the choice: the covariates it was trained with
            cfg.covariates = tuple(self.manifest["covariates"]["names"])
        return cfg

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "ModelBundle":
        path = Path(path)
        with open(path / MANIFEST_FILE, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: bundle schema {manifest.get('schema_version')!r}, this version reads {SCHEMA_VERSION}")
        if {h: tuple(c) for h, c in (manifest.get("heads") or {}).items()} != {h: tuple(c) for h, c in HEADS.items()}:
            raise ValueError(f"{path}: the bundle was trained for the calls {manifest.get('heads')}, this version makes {dict(HEADS)} "
                             "(a bundle with other calls needs the code it was trained with)")
        pre = torch.load(path / PREPROCESSING_FILE, map_location="cpu", weights_only=True)
        fs = FeatureSet.from_dict(manifest["features"])
        cov = manifest["covariates"]
        hp = HParams(**manifest["config"]["hparams"])
        model = build_model(manifest["config"].get("fit", "two_stage"), fs.n, len(cov["names"]), hp,
                            phase_head=manifest["config"].get("phase_head", "binary"), dna=bool(manifest["config"].get("dna_decoder", False)),
                            gc=bool(manifest["config"].get("gc_offset", False)),
                            enc_cov=bool(manifest["config"].get("encoder_covariates", True)),
                            gc_bins=len(GC_BINS) if (manifest.get("gc_term") or {}).get("profile") == "window300" else 0)
        model.load_state_dict(torch.load(path / MODEL_FILE, map_location="cpu", weights_only=True))
        model.to(device).eval()
        return cls(model, fs, ZScore(pre["z_mu"].numpy(), pre["z_sd"].numpy()),
                   CovariateScaler(cov["names"], cov["numeric"], pre["cov_mu"].numpy(), pre["cov_sd"].numpy()),
                   pre["ref_log"].numpy().astype(np.float64), manifest,
                   pre["missing_low"].numpy().astype(np.float64) if "missing_low" in pre else None,
                   pre["missing_low_pool"].numpy().astype(np.float64) if "missing_low_pool" in pre else None)


def save_bundle(path: str | Path, model, data: TrainingData, cfg: TrainConfig, device, seed: int,
                metrics: dict | None = None, extra: dict | None = None) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path / MODEL_FILE)
    torch.save({"ref_log": torch.tensor(data.ref_log), "z_mu": torch.tensor(data.zscore.mu), "z_sd": torch.tensor(data.zscore.sd),
                "cov_mu": torch.tensor(data.scaler.mu), "cov_sd": torch.tensor(data.scaler.sd),
                "log_mean": torch.tensor(data.log_mean)}
               | ({} if data.low is None else {"missing_low": torch.tensor(data.low)})
               | ({} if data.low_pool is None else {"missing_low_pool": torch.tensor(data.low_pool)}), path / PREPROCESSING_FILE)
    data.splits.to_csv(path / SPLITS_FILE)
    split = data.split
    manifest = {
        "schema_version": SCHEMA_VERSION, "ecoli_atlas_version": __version__,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
        "heads": {h: list(c) for h, c in HEADS.items()},
        "embedding_dim": cfg.hparams.z_dim,
        "fit": cfg.fit,
        "phase_head": cfg.phase_head,
        "fit_classes": {h: list(c) for h, c in data.fit_classes.items()},
        "embedding_blocks": embedding_blocks(cfg.hparams, cfg.fit),
        "n_samples": {"training_pool": int((split == 1).sum()), "training_reference": int(data.reference[split == 1].sum()),
                      "test": int((split == 3).sum()), "projected_only": int((split == 0).sum())},
        "n_studies": {"training_pool": int(data.splits.loc[split == 1, "gse"].nunique()),
                      "test": int(data.splits.loc[split == 3, "gse"].nunique())},
        "n_labelled_training": {h: {c: int((data.y_fit[h][data.tr_rows] == i).sum()) for i, c in enumerate(data.fit_classes[h])} for h in HEADS},
        "study_cap": int(data.study_cap),
        "missing_genes": {"called": data.low is not None, "antisense_veto": bool(data.extra.get("antisense_veto", False)),
                          "pooled": data.low_pool is not None,
                          "median_per_sample": float(np.median(data.M.sum(dim=1).numpy())) if data.M is not None else 0.0,
                          "samples_with_100_or_more": int((data.M.sum(dim=1) >= 100).sum()) if data.M is not None else 0},
        "label_caps": {h: {data.fit_classes[h][c]: k for c, k in label_weights(data.y_fit[h], data.gse_codes, data.tr_rows, cfg.label_cap)[1].items()} for h in HEADS},
        "dna": {"decoder": bool(cfg.dna_decoder), "inputs": bool(cfg.dna_inputs),
                "scale": float(torch.exp(model.log_dna_scale.detach())) if hasattr(model, "log_dna_scale") else None,
                "samples_with_dna": int(data.extra.get("dna_samples", 0)), "genome_length": data.fs.genome_length},
        "gc_term": {"on": hasattr(model, "gc_term"),
                    "scale": float(model.gc_term.scale.detach()) if hasattr(model, "gc_term") else None,
                    "profile": ("window300" if model.gc_term.n_bins else "gene") if hasattr(model, "gc_term") else None},
        "encoder_inputs": {"gc_cleaned": bool(cfg.clean_inputs), "gc_cleaned_samples": int(data.extra.get("gc_cleaned", 0)),
                           "gc_profile": data.extra["feature_gc"].kind if cfg.clean_inputs else None,
                           "covariates": bool(cfg.encoder_covariates)},
        "features": data.fs.to_dict(),
        "covariates": {"names": data.scaler.names, "numeric": data.scaler.numeric},
        "config": cfg.to_dict() | {"hparams": asdict(cfg.hparams)},
        "test_metrics": metrics or {},
        "environment": environment(device),
        "files": [MODEL_FILE, PREPROCESSING_FILE, SPLITS_FILE],
    } | (extra or {})
    with open(path / MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest
