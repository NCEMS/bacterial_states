"""Labels from the sample metadata (``metadata_v2.tsv`` of the inGEST atlas build).

Two things are read from it: the classes the heads are trained on, and which samples are known
to be the same or different conditions (for the within-study push and the cross-study control pull).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEADS

# growth_phase value -> phase class. `late_stationary` is deliberately absent: those samples carry
# no phase label and are not part of any phase evaluation.
PHASE_CLASS = {"mid_exponential": "exponential", "stationary": "stationary"}
# growth_phase value -> level of the ordinal phase head (config.PHASE_LEVELS). Here a culture of 24 to
# 72 h counts as stationary; `long_term_stationary`, `early_exponential` and `lag` have no level
PHASE_LEVEL = {"mid_exponential": 0, "transition": 1, "stationary": 2, "late_stationary": 2}
# phases between the two classes: unlabelled, and the head's output on them is reported as a check
PHASE_CHECK = ("lag", "early_exponential", "transition")
MEDIUM_CLASSES = HEADS["medium"]
# media between the two classes (`medium_class` values): unlabelled, the head's output on them is reported as a check
MEDIUM_CHECK = ("minimal_plus_hydrolysate", "defined_rich", "natural_fluid")
# strain_group values of the `lab` class of the lab_strain call: the laboratory lineages. Every other known group
# (pathogens, natural isolates, probiotic strains) is `non_lab`; an empty group has no label
LAB_GROUPS = ("K-12", "B", "C", "W")
# a sample's condition within its study
CONDITION_FIELDS = ("growth_phase", "culture_mode", "base_medium", "medium_class", "amino_acids_supplied", "carbon_source",
                    "oxygen", "growth_temperature", "genotype_class", "perturbed_genes", "inducer_added", "treatment", "exposure")
# what must agree for two control samples of different studies to count as the same condition
CONTROL_FIELDS = ("growth_phase", "base_medium", "medium_class", "carbon_source", "oxygen", "growth_temperature")
REQUIRED = tuple(dict.fromkeys(("GSE", "phase2", "medium_type", "treated", "standard_rnaseq", "is_control", "is_wild_type",
                                "is_reference_genotype", "planktonic_batch", "strain_group", "growth_phase_basis") + CONDITION_FIELDS))


def _true(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.strip().str.lower().isin(("true", "yes", "1")).to_numpy()


def check_columns(meta: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED) - set(meta.columns))
    if missing:
        raise ValueError(f"metadata lacks columns {missing}")


def is_reference(meta: pd.DataFrame) -> np.ndarray:
    """Reference samples: untreated, and of the wild type or of the baseline genotype a study
    compares against. TEST and the cross-validation folds are made of these samples only, and the
    preprocessing statistics come from them; all other samples only add to the training pool."""
    check_columns(meta)
    return _true(meta["is_reference_genotype"]) & (meta["treated"].astype(str).str.strip().str.lower() == "false").to_numpy()


def head_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Columns ``phase``, ``medium`` and ``lab_strain`` (class name or NaN). ``phase`` is set only
    where ``phase2`` is (untreated, standard RNA-seq samples) and the growth phase is one of the two
    clean classes. ``medium`` is `rich` for complex broths and `minimal` for salts with a carbon
    source; every medium in between has no label. ``lab_strain`` is `lab` for the laboratory
    lineages (LAB_GROUPS: K-12, B, C, W) and `non_lab` for every other known strain group (pathogens,
    natural isolates, probiotic strains). It is a call about the lineage: with the genes a strain
    lacks masked, B, C and W strains sit with K-12 in the blocks that are not trained on it."""
    check_columns(meta)
    out = pd.DataFrame(index=meta.index)
    out["phase"] = meta["growth_phase"].map(PHASE_CLASS).where(meta["phase2"].astype(str).str.strip() != "")
    out["medium"] = meta["medium_type"].where(meta["medium_type"].isin(MEDIUM_CLASSES))
    group = meta["strain_group"].astype(str).str.strip()
    out["lab_strain"] = pd.Series(np.where(group.isin(LAB_GROUPS), "lab", "non_lab"), index=meta.index).where(group != "")
    return out.astype("object")


def scored_labels(meta: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The labels a call is scored on. As ``labels``, but a phase label counts only where the
    growth phase is stated in the study's text; a phase inferred from an OD or a culture time is
    used for fitting and left out of every score."""
    out = labels.copy()
    out["phase"] = out["phase"].where(meta["growth_phase_basis"].astype(str).str.strip() == "stated")
    return out


def label_codes(labels: pd.DataFrame) -> dict:
    """head -> int array (0 / 1, -1 = no label)."""
    return {h: labels[h].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int).to_numpy()
            for h, classes in HEADS.items()}


def _untreated_standard(meta: pd.DataFrame) -> np.ndarray:
    return (~_true(meta["treated"])) & (meta["treated"].astype(str).str.strip() != "").to_numpy() & _true(meta["standard_rnaseq"])


def phase_levels(meta: pd.DataFrame) -> np.ndarray:
    """Level of every sample on the ordered phase scale (PHASE_LEVEL; -1 = none). Only untreated,
    standard RNA-seq samples have one, as for the phase classes."""
    check_columns(meta)
    return meta["growth_phase"].map(PHASE_LEVEL).where(_untreated_standard(meta)).fillna(-1).astype(int).to_numpy()


def fit_codes(meta: pd.DataFrame, labels: pd.DataFrame, phase_head: str = "binary") -> dict:
    """head -> the codes the head is fitted on: ``label_codes``, with the phase levels in place of
    the two phase classes for the ordinal phase head."""
    out = label_codes(labels)
    if phase_head == "ordinal":
        out["phase"] = phase_levels(meta)
    return out


def phase_check_group(meta: pd.DataFrame) -> pd.Series:
    """The in-between phase of untreated standard samples ('' for all others)."""
    ok = _untreated_standard(meta)
    return meta["growth_phase"].where(meta["growth_phase"].isin(PHASE_CHECK) & ok, "")


def medium_check_group(meta: pd.DataFrame) -> pd.Series:
    """The in-between medium class of a sample ('' for all others)."""
    return meta["medium_class"].where(meta["medium_class"].isin(MEDIUM_CHECK), "")


def condition_identity(meta: pd.DataFrame) -> dict:
    """Arrays for the alignment terms.

    cond_id     code of the sample's condition (all CONDITION_FIELDS together)
    is_annot    treatment and genotype are both known, so the condition can be compared
    ctrl_cond   code of the control condition (CONTROL_FIELDS)
    is_ctrl     untreated wild-type control in planktonic batch culture with known phase and medium
    """
    check_columns(meta)
    text = meta[list(REQUIRED)].astype(str).apply(lambda s: s.str.strip())
    cond = text[list(CONDITION_FIELDS)].agg("|".join, axis=1)
    ctrl = text[list(CONTROL_FIELDS)].agg("|".join, axis=1)
    is_annot = ((text["treatment"] != "") & (text["genotype_class"] != "")).to_numpy()
    is_ctrl = (_true(text["is_control"]) & _true(text["is_wild_type"]) & (text["treated"].str.lower() == "false").to_numpy()
               & _true(text["planktonic_batch"]) & (text["growth_phase"] != "").to_numpy() & (text["medium_class"] != "").to_numpy())
    return {"cond_id": pd.factorize(cond)[0], "is_annot": is_annot, "ctrl_cond": pd.factorize(ctrl)[0], "is_ctrl": is_ctrl}
