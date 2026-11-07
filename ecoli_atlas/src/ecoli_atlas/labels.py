"""Metadata harmonisation into training labels, condition identities and control masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PHASE_ORDER

CORE_META_COLS = ["gse", "strain_background", "strain", "medium_type", "base_medium", "growth_phase",
                  "stress", "carbon_source", "is_lab_strain"]
EXTRA_META_COLS = ["substrain", "OD600", "temperature(C)", "oxygen", "culture_vessel",
                   "culture_volume(mL)", "stress_type", "stress_subtype", "title"]
RARE_COLLAPSED = ["growth_phase", "medium_type", "strain_background", "carbon_source", "is_lab_strain"]
PHASE_COLLAPSE = {"lag": "exponential", "early_exponential": "exponential", "mid_exponential": "exponential",
                  "early_stationary": "stationary", "late_stationary": "stationary"}

# Condition identity = the full metadata record except sample id and study (cell 39 of the notebook)
IDENTITY_FIELDS = ["strain", "strain_background", "is_lab_strain", "medium_type", "base_medium",
                   "growth_phase", "OD600", "carbon_source", "temperature(C)", "oxygen",
                   "culture_vessel", "culture_volume(mL)", "stress", "stress_type", "stress_subtype"]
CONTROL_FIELDS = ["growth_phase", "medium_type", "carbon_source", "base_medium"]
K12_PATTERN = r"mg\s*1655|w3110|bw\s*25113|mc4100|dh5|dh10b|jm10[19]|top\s*10|ncm3722|k[\s_-]?12"


def canonical_stress(x) -> str:
    return ";".join(sorted(s.strip() for s in str(x).split(";") if s.strip()))


def drop_rare(s: pd.Series, min_n: int) -> pd.Series:
    s = s.copy()
    vc = s.value_counts()
    s[s.isin(vc[vc < min_n].index)] = np.nan
    return s


def build_labels(meta: pd.DataFrame, min_class_n: int = 10) -> pd.DataFrame:
    """Harmonise the raw metadata table (metadata_revised.csv layout) into modelling labels."""
    meta = meta.copy()
    meta["strain_background"] = meta["strain"]  # the notebook uses the strain field as background
    extra = [c for c in EXTRA_META_COLS if c in meta.columns]
    missing_core = [c for c in CORE_META_COLS if c not in meta.columns]
    if missing_core:
        raise ValueError(f"metadata lacks required columns: {missing_core}")
    meta = meta[CORE_META_COLS + extra].copy()

    meta["growth_phase"] = meta["growth_phase"].replace(PHASE_COLLAPSE)
    meta["growth_phase"] = meta["growth_phase"].where(meta["growth_phase"].isin(PHASE_ORDER), np.nan)
    meta["is_lab_strain"] = meta["is_lab_strain"].astype("boolean")
    complex_medium = meta["medium_type"] == "complex"
    meta.loc[complex_medium, "carbon_source"] = np.nan
    meta["stress"] = meta["stress"].fillna("").apply(canonical_stress)

    for v in RARE_COLLAPSED:
        meta[v + "_c"] = drop_rare(meta[v], min_class_n)

    meta["is_stressed"] = meta["stress"].str.len() > 0
    annotated = meta["growth_phase_c"].notna()
    for sc in ("strain", "strain_background", "strain_background_c"):
        annotated = annotated | meta[sc].notna()
    meta["is_wildtype"] = (~meta["is_stressed"] & annotated).values
    return meta


def rowkey(df: pd.DataFrame, cols, sep: str = "|") -> pd.Series:
    """Join columns into one identity string; missing values render as 'nan' and therefore match."""
    return df[cols].astype("object").apply(lambda s: s.map(str)).agg(sep.join, axis=1)


def with_collapsed(meta: pd.DataFrame, cols) -> list:
    """Prefer the rare-class-collapsed ``<col>_c`` version of each column when it exists."""
    return [(c + "_c") if (c + "_c") in meta.columns else c
            for c in cols if ((c + "_c") in meta.columns or c in meta.columns)]


@dataclass
class FactorCodes:
    name: str
    classes: list          # sorted category labels as strings
    codes: np.ndarray      # int, -1 = unlabelled
    pos_idx: int = 1       # categories are sorted; index 1 is the positive class

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def positive_class(self) -> str:
        return self.classes[self.pos_idx]


def factor_codes(meta: pd.DataFrame, factor: str) -> FactorCodes:
    cat = pd.Categorical(meta[factor + "_c"])
    fc = FactorCodes(factor, [str(c) for c in cat.categories], np.asarray(cat.codes, dtype=np.int64))
    if fc.n_classes != 2:
        raise ValueError(f"{factor}: expected a binary factor, found classes {fc.classes}")
    return fc


def condition_ids(meta: pd.DataFrame) -> np.ndarray:
    return np.asarray(pd.Categorical(rowkey(meta, with_collapsed(meta, IDENTITY_FIELDS))).codes, dtype=np.int64)


def is_annotated(meta: pd.DataFrame) -> np.ndarray:
    return meta["growth_phase_c"].notna().values


def is_k12(meta: pd.DataFrame) -> np.ndarray:
    bg = meta["strain_background"].astype(str).str.lower()
    return bg.str.contains(K12_PATTERN, regex=True, na=False).values


def control_condition_ids(meta: pd.DataFrame) -> np.ndarray:
    return np.asarray(pd.Categorical(rowkey(meta, with_collapsed(meta, CONTROL_FIELDS))).codes, dtype=np.int64)


def is_control(meta: pd.DataFrame) -> np.ndarray:
    """Wild-type K-12 samples with known growth phase and medium type: cross-study anchors."""
    return (meta["is_wildtype"].fillna(False).astype(bool).values & is_k12(meta)
            & meta["growth_phase_c"].notna().values & meta["medium_type_c"].notna().values)
