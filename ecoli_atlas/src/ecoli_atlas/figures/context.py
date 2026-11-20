"""What the model figures and the case studies read: the training tables put through a trained
bundle, as `ecoli_atlas evaluate` does (same preprocessing, same split)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import ModelBundle
from ..config import HEADS, TrainConfig
from ..data import TrainingData
from ..train import encode

EMBEDDING_FILE = "embedding.tsv"


@dataclass
class FigureContext:
    bundle: ModelBundle
    data: TrainingData
    cfg: TrainConfig
    device: object
    z: np.ndarray                   # the embedding, samples x dimensions
    probs: dict                     # head -> probability of its positive class
    scores: dict                    # head -> log-odds of its positive class
    gene_names: list                # one label per model feature: the symbol, or the feature id where a symbol repeats
    log: object = print
    _cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, bundle: ModelBundle, data: TrainingData, device, symbols: pd.Series | None = None, log=print) -> "FigureContext":
        enc = encode(bundle.model, data.X, data.C, device)
        ids = list(data.fs.ids)
        names = ids
        if symbols is not None:
            sym = symbols.reindex(ids).fillna("").astype(str)
            repeated = sym.duplicated(keep=False) | (sym == "")
            names = list(np.where(repeated, ids, sym))
        return cls(bundle, data, bundle.config, device, enc["z"], enc["probs"], enc["scores"], names, log)

    @property
    def meta(self) -> pd.DataFrame:
        return self.data.meta

    @property
    def ids(self) -> list:
        return self.data.ids

    @property
    def split(self) -> np.ndarray:
        return self.data.split

    @property
    def gse(self) -> np.ndarray:
        return self.data.splits["gse"].astype(str).to_numpy()

    @property
    def seed(self) -> int:
        return int(self.cfg.seed)

    @property
    def heads(self) -> list:
        return list(HEADS)

    @property
    def blocks(self) -> dict:
        """block name -> its columns of the embedding (``embedding_blocks`` of the manifest)."""
        return {b["block"]: self.z[:, b["start"]:b["end"]] for b in self.bundle.manifest["embedding_blocks"]}

    @property
    def logx(self) -> np.ndarray:
        """log2 normalised expression, samples x model features."""
        if "logx" not in self._cache:
            self._cache["logx"] = self.data.X.numpy() * self.data.zscore.sd + self.data.zscore.mu
        return self._cache["logx"]

    def labels(self, head: str) -> np.ndarray:
        """Class name of every sample for a head (None where it has no label)."""
        return self.data.labels[head].astype("object").where(self.data.labels[head].notna(), None).to_numpy()

    def column(self, name: str) -> np.ndarray:
        """A metadata column as stripped text ('' everywhere when the table has no such column)."""
        if name not in self.meta.columns:
            return np.full(len(self.ids), "", dtype=object)
        return self.meta[name].astype(str).str.strip().to_numpy(dtype=object)

    def embedding_check(self, models_dir) -> dict:
        """Largest difference between the embedding made here and the bundle's ``embedding.tsv``."""
        path = Path(models_dir) / EMBEDDING_FILE
        if not path.exists():
            return {"skipped": f"no {EMBEDDING_FILE} in the bundle"}
        saved = pd.read_csv(path, sep="\t", index_col=0)
        saved.index = saved.index.astype(str)
        missing = sorted(set(self.ids) - set(saved.index))
        if missing or saved.shape[1] != self.z.shape[1]:
            return {"skipped": f"{EMBEDDING_FILE} does not match the tables ({len(missing)} samples missing, {saved.shape[1]} columns)"}
        diff = np.abs(saved.loc[self.ids].to_numpy() - self.z)
        return {"n_samples": int(len(self.ids)), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
