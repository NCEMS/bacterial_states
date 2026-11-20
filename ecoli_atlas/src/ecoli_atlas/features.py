"""Feature groups of the counting annotation.

The model sees mRNA features (CDS and pseudogenes) and the other RNAs (ncRNA genes and Rfam hits)
as encoder inputs and decoder outputs. Only mRNA features enter the size factor. rRNA and tRNA are
neither input nor output: their abundance relative to mRNA describes the library (rRNA depletion,
size selection) and is passed to the model as covariates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MRNA_TYPES = ("CDS", "pseudo")
OTHER_RNA_TYPES = ("ncRNA",)
DEPLETED_TYPES = ("rRNA", "tRNA")
# rRNA genes by the first letters of their symbol
RRNA_FAMILIES = {"16S": "rrs", "23S": "rrl", "5S": "rrf"}
COMPOSITION = ("16S", "23S", "5S", "tRNA")


def read_feature_table(path: str | Path) -> pd.DataFrame:
    """``*_counting_v2.features.tsv`` of the inGEST annotation, indexed by Feature_ID."""
    feat = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in ("Feature_ID", "Symbol", "Type", "Length") if c not in feat.columns]
    if missing:
        raise ValueError(f"{path}: feature table lacks columns {missing}")
    feat = feat.set_index("Feature_ID")
    feat["Length"] = feat["Length"].astype(int)
    return feat


@dataclass
class FeatureSet:
    """Model features (in model order) and the depleted families used for the composition covariates."""
    ids: list                 # model features: mRNA first, then other RNA
    is_mrna: np.ndarray       # bool over ids
    lengths: np.ndarray       # bp, over ids
    families: dict            # "16S" / "23S" / "5S" / "tRNA" -> list of feature ids
    order: np.ndarray | None = None   # model features in chromosome order (indices into ids); None in bundles made before the missing-gene calls

    @classmethod
    def from_table(cls, feat: pd.DataFrame) -> "FeatureSet":
        unknown = sorted(set(feat["Type"]) - set(MRNA_TYPES + OTHER_RNA_TYPES + DEPLETED_TYPES))
        if unknown:
            raise ValueError(f"feature table has types this model does not know: {unknown}")
        mrna = feat.index[feat["Type"].isin(MRNA_TYPES)]
        other = feat.index[feat["Type"].isin(OTHER_RNA_TYPES)]
        ids = list(mrna) + list(other)
        rrna = feat[feat["Type"] == "rRNA"]
        families = {name: list(rrna.index[rrna["Symbol"].str.startswith(prefix)]) for name, prefix in RRNA_FAMILIES.items()}
        families["tRNA"] = list(feat.index[feat["Type"] == "tRNA"])
        # chromosome order of the model features (the table's own order when it has no positions)
        sub = feat.loc[ids]
        if "Start" in feat.columns:
            keys = [pd.to_numeric(sub["Start"], errors="coerce").fillna(0).to_numpy()]
            if "Chr" in feat.columns:
                keys.append(pd.factorize(sub["Chr"])[0])
            order = np.lexsort(keys)
        else:
            order = np.argsort(feat.index.get_indexer(ids), kind="stable")
        return cls(ids, np.array([True] * len(mrna) + [False] * len(other)), sub["Length"].to_numpy(), families, order.astype(int))

    @property
    def n(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict:
        return {"ids": self.ids, "is_mrna": [bool(x) for x in self.is_mrna], "lengths": [int(x) for x in self.lengths],
                "families": self.families} | ({} if self.order is None else {"order": [int(x) for x in self.order]})

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSet":
        return cls(list(d["ids"]), np.array(d["is_mrna"], dtype=bool), np.array(d["lengths"], dtype=int), dict(d["families"]),
                   np.array(d["order"], dtype=int) if d.get("order") is not None else None)
