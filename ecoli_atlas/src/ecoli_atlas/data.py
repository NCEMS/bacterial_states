"""Training tables and the tensors made from them.

Inputs are the three tables of the atlas build: the joined count table (features x samples, GSM
columns), the sample metadata (``metadata_v2.tsv``) and the QC table (``sample_qc.tsv``). A sample
is used when the QC table names a model for it in ``in_model``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .config import HEADS, TrainConfig, fit_classes
from .features import COMPOSITION, FeatureSet
from .io import read_counts
from .labels import condition_identity, fit_codes, head_labels, label_codes, phase_levels, scored_labels
from .preprocess import (ZScore, align_counts, composition_covariates, log_expression, low_expression_profile, missing_genes,
                         reference_profile, size_factors, weighted_stats)

# QC covariates: values of the QC table (column `full_<name>`, or `<name>` in a plain table) that
# describe the library and the run, not the biology or the strain.
QC_NUMERIC = ("length_q05_r1", "length_q50_r1", "length_q95_r1", "cycles_r1", "pct_kept", "pct_adapter_trimmed",
              "duplication_rate", "gc_content", "strand_purity", "length_q50_r2", "fragment_q25", "fragment_q50",
              "fragment_q75", "log_dna_fraction", "leakage_b", "intergenic_share", "gene_body_slope",
              "pct_short_features", "log_other_genus")
INSTRUMENTS = ("novaseq", "nextseq", "hiseq", "miseq")
QC_BINARY = ("paired",) + tuple("instrument_" + i for i in INSTRUMENTS) + ("has_qc", "dna_below_limit")
COMPOSITION_NAMES = ("log_mrna_reads",) + tuple(f"log_{k}_to_mrna" for k in COMPOSITION)


def read_table(path) -> pd.DataFrame:
    """A tab-separated table of text, indexed by its first column (the sample id)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.strip()
    return df


def _qc_value(qc: pd.DataFrame, name: str) -> pd.Series:
    for col in ("full_" + name, name):
        if col in qc.columns:
            return qc[col]
    return pd.Series("", index=qc.index)


def qc_covariates(qc: pd.DataFrame | None, ids: list) -> pd.DataFrame:
    """QC_NUMERIC (NaN where a value does not exist) and QC_BINARY columns for the samples ``ids``;
    a sample without a QC row gets NaN everywhere and ``has_qc`` 0."""
    out = pd.DataFrame(np.nan, index=ids, columns=list(QC_NUMERIC + QC_BINARY))
    out[list(QC_BINARY)] = 0.0
    if qc is None:
        return out
    q = qc.reindex(ids)
    have = q.notna().any(axis=1).to_numpy() if len(q.columns) else np.zeros(len(ids), bool)
    q = q.fillna("")
    num = lambda name: pd.to_numeric(_qc_value(q, name), errors="coerce")       # noqa: E731
    for name in QC_NUMERIC:
        if name == "strand_purity":
            out[name] = np.fmax(num("fraction_forward"), num("fraction_reverse"))
        elif name == "log_dna_fraction":
            dna = num("dna_fraction_est")
            out[name] = np.log10(dna.where(dna > 0))
            out["dna_below_limit"] = (dna == 0).astype(float)
        elif name == "log_other_genus":
            out[name] = np.log10(num("pct_top_other_genus_nostable") + 0.01)
        else:
            out[name] = num(name)
    layout = _qc_value(q, "layout").where(_qc_value(q, "layout") != "", q["layout"] if "layout" in q.columns else "")
    out["paired"] = layout.str.upper().isin(("PE", "PAIRED")).astype(float)
    model = _qc_value(q, "instrument_model").where(_qc_value(q, "instrument_model") != "",
                                                    q["instrument_model"] if "instrument_model" in q.columns else "")
    for inst in INSTRUMENTS:
        out["instrument_" + inst] = model.str.lower().str.replace(" ", "").str.contains(inst).astype(float)
    out["has_qc"] = have.astype(float)
    return out


@dataclass
class CovariateScaler:
    """Standardises the numeric covariates with training statistics; a missing value becomes 0 (the
    training mean). Binary columns pass through."""
    names: list
    numeric: list
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, rows: np.ndarray, w: np.ndarray | None = None) -> "CovariateScaler":
        numeric = [c for c in frame.columns if c in COMPOSITION_NAMES + QC_NUMERIC]
        X = frame[numeric].to_numpy(dtype=np.float64)[rows]
        mu, sd = np.zeros(len(numeric)), np.ones(len(numeric))
        for j in range(len(numeric)):
            ok = ~np.isnan(X[:, j])
            if ok.sum() > 1:
                m, s = weighted_stats(X[ok, j][:, None], None if w is None else w[ok])
                mu[j], sd[j] = m[0], s[0] if s[0] > 0 else 1.0
        return cls(list(frame.columns), numeric, mu.astype(np.float32), sd.astype(np.float32))

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        out = frame[self.names].to_numpy(dtype=np.float32).copy()
        idx = [self.names.index(c) for c in self.numeric]
        out[:, idx] = np.nan_to_num((out[:, idx] - self.mu) / self.sd, nan=0.0)
        return out


def covariate_frame(Y: np.ndarray, fam: np.ndarray, fs: FeatureSet, qc: pd.DataFrame | None, ids: list) -> pd.DataFrame:
    comp = pd.DataFrame(composition_covariates(Y, fam, fs.is_mrna), index=ids, columns=list(COMPOSITION_NAMES))
    return comp.join(qc_covariates(qc, ids))


@dataclass
class AtlasTables:
    counts: pd.DataFrame      # features x samples
    meta: pd.DataFrame        # indexed by sample
    qc: pd.DataFrame          # indexed by sample

    @property
    def ids(self) -> list:
        return list(self.counts.columns)


def load_tables(counts_path, metadata_path, qc_path, model: str | None = None) -> AtlasTables:
    """The samples the QC table assigns to a model (``in_model``; ``model`` selects one name),
    with their counts and metadata. Fails if a model sample lacks counts or metadata."""
    qc = read_table(qc_path)
    if "in_model" not in qc.columns:
        raise ValueError(f"{qc_path}: no in_model column (qc_table.py mark-model sets it)")
    chosen = qc.index[(qc["in_model"] == model) if model else (qc["in_model"] != "")]
    if not len(chosen):
        raise ValueError(f"{qc_path}: no sample is marked for " + (f"model {model!r}" if model else "a model"))
    counts = read_counts(counts_path)
    meta = read_table(metadata_path)
    for name, have in (("count table", counts.columns), ("metadata", meta.index)):
        missing = sorted(set(chosen) - set(have))
        if missing:
            raise ValueError(f"{len(missing)} model samples are not in the {name}: {missing[:5]}")
    ids = sorted(chosen)
    return AtlasTables(counts[ids], meta.loc[ids], qc.loc[ids])


def study_fold(splits: pd.DataFrame) -> np.ndarray:
    """The fold of every sample's study (NaN for a study without a fold). The fold is recorded for
    reference samples only; all other samples of the study are held out with it."""
    return splits.groupby("gse")["cv_fold"].transform("max").to_numpy(dtype=float)


def study_weights(gse: np.ndarray, rows: np.ndarray, cap: int | None) -> tuple[np.ndarray, int]:
    """Weight min(1, cap / n) for every sample of a study with n samples among ``rows``; with
    ``cap`` None the cap is the size of the second-largest study. Returns (weights, cap)."""
    sizes = pd.Series(gse[rows]).value_counts()
    if cap is None:
        cap = int(sizes.iloc[1]) if len(sizes) > 1 else int(sizes.iloc[0])
    n = pd.Series(gse).map(sizes).fillna(1).to_numpy(dtype=float)
    return np.minimum(1.0, cap / n), int(cap)


def label_weights(y: np.ndarray, gse: np.ndarray, rows: np.ndarray, cap: int | None = None) -> tuple[np.ndarray, dict]:
    """Weight of every sample's label of one call among ``rows``: min(1, cap / n) for a study with n
    samples of the sample's class, so that no study counts for more than ``cap`` samples of a
    class. With ``cap`` None the cap of a class is the size of the second-largest study in it.
    Samples without a label or outside ``rows`` get 0. Returns (weights, {class code: cap})."""
    w, caps = np.zeros(len(y)), {}
    rows = np.asarray(rows)
    labelled = rows[y[rows] >= 0]
    for c in np.unique(y[labelled]):
        r = labelled[y[labelled] == c]
        sizes = pd.Series(gse[r]).value_counts()
        k = int(cap) if cap is not None else int(sizes.iloc[1] if len(sizes) > 1 else sizes.iloc[0])
        w[r] = np.minimum(1.0, k / pd.Series(gse[r]).map(sizes).to_numpy(dtype=float))
        caps[int(c)] = k
    return w, caps


@dataclass
class TrainingData:
    ids: list
    fs: FeatureSet
    meta: pd.DataFrame
    labels: pd.DataFrame            # phase / medium class names
    splits: pd.DataFrame
    X: torch.Tensor                 # z-scored log expression, samples x features
    Y: torch.Tensor                 # counts
    log_s: torch.Tensor             # log size factor
    C: torch.Tensor                 # covariates
    y: dict                         # head -> int array (0 / 1, -1 = unlabelled): the two classes of every call
    y_eval: dict                    # the labels the calls are scored on (phase: stated only)
    y_fit: dict                     # the codes the heads are fitted on: y, or the phase levels for the ordinal phase head
    fit_classes: dict               # head -> names of the codes in y_fit
    phase_level: np.ndarray         # level on the ordered phase scale (-1 = none), for the within-study order measure
    gse_codes: np.ndarray
    identity: dict                  # cond_id, is_annot, ctrl_cond, is_ctrl
    weights: np.ndarray             # study-cap weight per sample
    study_cap: int
    ref_log: np.ndarray
    zscore: ZScore
    scaler: CovariateScaler
    log_mean: np.ndarray            # per-feature log of the mean normalised count over the reference training samples
    M: torch.Tensor | None = None   # samples x features, True where the gene is called missing from the sample's strain
    low: np.ndarray | None = None   # per-feature low end of the normalised count (for the missing-gene calls); None = no calls
    extra: dict = field(default_factory=dict)

    @property
    def split(self) -> np.ndarray:
        return self.splits["split"].to_numpy()

    @property
    def tr_rows(self) -> np.ndarray:
        return np.where(self.split == 1)[0]

    @property
    def te_rows(self) -> np.ndarray:
        return np.where(self.split == 3)[0]

    @property
    def reference(self) -> np.ndarray:
        return self.splits["reference"].to_numpy(dtype=bool)

    @property
    def fold_rows(self) -> np.ndarray:
        """The training-pool samples that are in a fold (its reference samples)."""
        return np.where((self.split == 1) & self.splits["cv_fold"].notna().to_numpy())[0]

    def fit_rows(self, k: int) -> np.ndarray:
        """The training pool without the studies of fold ``k``: none of their samples, reference or
        not, is fitted."""
        return np.where((self.split == 1) & (study_fold(self.splits) != k))[0]

    def val_rows(self, k: int) -> np.ndarray:
        return np.where((self.split == 1) & (self.splits["cv_fold"].to_numpy() == k))[0]


def build_training(tables: AtlasTables, fs: FeatureSet, cfg: TrainConfig, splits: pd.DataFrame,
                   fit_rows: np.ndarray | None = None) -> TrainingData:
    """Everything the model needs. ``fit_rows`` (default: the training pool) are the samples the
    model is fitted on; the statistics (size-factor reference, z-score, covariate scaling) come from
    the reference samples among them, weighted by the study cap of ``fit_rows``."""
    ids = tables.ids
    splits = splits.loc[ids]
    Y, fam, _ = align_counts(tables.counts, fs)
    gse = tables.meta["GSE"].astype(str).to_numpy()
    fit_rows = np.where(splits["split"].to_numpy() == 1)[0] if fit_rows is None else np.asarray(fit_rows)
    weights, cap = study_weights(gse, fit_rows, cfg.study_cap)
    rows = fit_rows[splits["reference"].to_numpy(dtype=bool)[fit_rows]]
    if not len(rows):
        raise ValueError("no reference sample among the samples to fit on")
    ref_log = reference_profile(Y[rows], fs.is_mrna, weights[rows], cfg.min_gene_presence)
    s = size_factors(Y, ref_log)
    low, miss = None, np.zeros(Y.shape, dtype=bool)
    if cfg.mask_missing:
        if fs.order is None:
            raise ValueError("the feature set has no chromosome order; the missing-gene calls need it")
        low = low_expression_profile(Y[rows], s[rows], fs.order, cfg.mask_min_expected, cfg.mask_low_fraction, cfg.mask_quantile)
        miss = missing_genes(Y, s, low, fs.order, cfg.mask_min_expected, cfg.mask_low_fraction)
        s = size_factors(Y, ref_log, miss)
    logx = log_expression(Y, s)
    zscore = ZScore.fit(logx[rows], weights[rows], cfg.sd_floor, miss[rows])
    cov = covariate_frame(Y, fam, fs, tables.qc, ids)
    scaler = CovariateScaler.fit(cov, rows, weights[rows])
    labels = head_labels(tables.meta)
    w = weights[rows] / weights[rows].sum()
    have = w[:, None] * ~miss[rows]                    # a gene's mean is taken over the samples that have it
    total = have.sum(axis=0)
    log_mean = np.log(np.maximum((have * (Y[rows] / s[rows][:, None])).sum(axis=0) / np.where(total > 0, total, 1.0), 1e-3))
    assert set(label_codes(labels)) == set(HEADS)
    return TrainingData(ids=ids, fs=fs, meta=tables.meta, labels=labels, splits=splits,
                        X=torch.tensor(zscore.apply(logx, miss)), Y=torch.tensor(Y, dtype=torch.float32),
                        log_s=torch.tensor(np.log(s), dtype=torch.float32), C=torch.tensor(scaler.apply(cov)),
                        y=label_codes(labels), y_eval=label_codes(scored_labels(tables.meta, labels)),
                        y_fit=fit_codes(tables.meta, labels, cfg.phase_head), fit_classes={h: fit_classes(h, cfg.phase_head) for h in HEADS},
                        phase_level=phase_levels(tables.meta), gse_codes=pd.factorize(gse)[0], identity=condition_identity(tables.meta),
                        weights=weights, study_cap=cap, ref_log=ref_log, zscore=zscore, scaler=scaler,
                        log_mean=log_mean.astype(np.float32), M=torch.tensor(miss), low=low, extra={"covariates_raw": cov})
