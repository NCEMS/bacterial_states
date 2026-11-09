"""Everything the notebook has in memory after training, rebuilt from the saved bundle without
retraining: aligned training data (with the recorded split), the z-scored input made with the
bundle's saved statistics, and the embedding / factor blocks / class probabilities of every sample."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import MANIFEST_FILE, ModelBundle, load_splits
from ..config import TrainConfig
from ..data import TrainingData, build_training_data
from ..embed import EmbedResult, embed_zscored
from ..preprocess import align_to_genes


@dataclass
class FigureContext:
    data: TrainingData
    bundle: ModelBundle
    cfg: TrainConfig
    Z: np.ndarray                 # samples x model genes, z-scored with the bundle's saved statistics
    res: EmbedResult              # embedding (final), blocks (raw factor blocks), proba
    seed: int
    device: object
    zscore_max_dev: float | None  # max |data.X - Z| when the gene lists coincide
    gene_coverage: float
    log: object = field(default=lambda *_: None, repr=False)
    _logcpm: np.ndarray | None = field(default=None, repr=False)

    @property
    def meta(self) -> pd.DataFrame:
        return self.data.meta

    @property
    def genes(self) -> list:
        return self.data.genes

    @property
    def factors(self) -> list:
        return self.bundle.factor_order

    @property
    def embedding(self) -> np.ndarray:
        return self.res.embedding.values

    @property
    def logcpm(self) -> np.ndarray:
        if self._logcpm is None:
            self._logcpm = np.log1p(self.data.cpm)
        return self._logcpm

    @property
    def concat_blocks(self) -> np.ndarray:
        return np.concatenate([self.res.blocks[f] for f in self.factors], axis=1)

    def positive_index(self, factor: str) -> int:
        fa = self.bundle.factors[factor]
        if fa.positive_class in fa.classes:
            return fa.classes.index(fa.positive_class)
        return int(self.bundle.prep.get("positive_class_index", {}).get(factor, 1))

    def proba_positive(self, factor: str) -> np.ndarray:
        return self.res.proba[factor][:, self.positive_index(factor)]

    def classes(self, factor: str) -> list:
        return list(self.bundle.factors[factor].classes)

    def labels_str(self, factor: str) -> pd.Series:
        """Collapsed labels of a factor as strings (NaN where unlabelled), aligned to the bundle's class names."""
        s = self.meta[factor + "_c"]
        return pd.Series([None if pd.isna(v) else str(v) for v in s], index=s.index, dtype=object)

    def codes(self, factor: str) -> np.ndarray:
        return self.data.factors[factor].codes


def _manifest_config(models_dir: Path, seed: int | None, log) -> TrainConfig:
    mf = models_dir / MANIFEST_FILE
    if mf.exists():
        cfg = TrainConfig.from_manifest(mf)
    else:
        log(f"warning: {mf} not found; using default training configuration")
        cfg = TrainConfig()
    if seed is not None:
        cfg.seed = int(seed)
    return cfg


def build_context(counts: pd.DataFrame, meta_raw: pd.DataFrame, feat: pd.DataFrame, models_dir: str | Path,
                  device="cpu", seed: int | None = None, log=None) -> FigureContext:
    """counts: genes x samples; meta_raw: metadata_revised.csv layout; feat: NCBI feature table."""
    log = log or (lambda *_: None)
    models_dir = Path(models_dir)
    bundle = ModelBundle.load(models_dir, device)
    cfg = _manifest_config(models_dir, seed, log)
    cfg.factors = tuple(bundle.factor_order)
    splits = load_splits(models_dir)
    if splits is None:
        log(f"warning: {models_dir / 'splits.csv'} not found; the study split is searched again (slow, may differ)")
    data = build_training_data(counts, meta_raw, feat, cfg, splits=splits, log=log)

    cpm_df = pd.DataFrame(data.cpm, index=data.meta.index, columns=data.genes)
    aligned, cov = align_to_genes(cpm_df, bundle.genes)
    Z = bundle.zscore.apply(aligned.values.astype(np.float64))
    zdev = float(np.max(np.abs(data.X.numpy() - Z))) if list(data.genes) == list(bundle.genes) else None
    if zdev is None:
        log(f"warning: training-table genes differ from the bundle ({cov.n_present}/{cov.n_model_genes} model genes present)")
    else:
        log(f"z-score with the saved statistics: max deviation from a fresh fit {zdev:.2e}")
    res = embed_zscored(Z, list(data.meta.index), bundle)
    log(f"embedded {len(res.embedding)} samples -> {res.embedding.shape[1]} dims with {models_dir}")
    return FigureContext(data=data, bundle=bundle, cfg=cfg, Z=Z, res=res, seed=cfg.seed, device=device,
                         zscore_max_dev=zdev, gene_coverage=cov.fraction, log=log)
