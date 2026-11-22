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
from .labels import condition_identity, fit_codes, head_labels, label_codes, order_levels, order_series_ok, phase_levels, scored_labels
from .preprocess import (ZScore, align_counts, call_groups, composition_covariates, group_calls, log_expression,
                         low_expression_profile, missing_genes, pooled_low_profile, reference_profile, size_factors, weighted_stats)

# QC covariates: values of the QC table (column `full_<name>`, or `<name>` in a plain table) that
# describe the library and the run, not the biology or the strain. The substitution rate (mismatches
# against MG1655) is mostly a property of the sequencing run: it spans a wider range among K-12 samples
# than between strain groups. The share of reads off the MG1655 reference stays out: it follows the strain.
QC_NUMERIC = ("length_q05_r1", "length_q50_r1", "length_q95_r1", "cycles_r1", "pct_kept", "pct_adapter_trimmed",
              "duplication_rate", "gc_content", "strand_purity", "length_q50_r2", "fragment_q25", "fragment_q50",
              "fragment_q75", "log_dna_fraction", "leakage_b", "intergenic_share", "gene_body_slope",
              "pct_short_features", "log_other_genus", "log_substitution_rate", "gc_slope", "gc_curv")
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
        elif name == "log_substitution_rate":
            rate = num("substitution_rate")
            out[name] = np.log10(rate.where(rate > 0))
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


# Covariates that every atlas sample has (QC_PAIRED_REQUIRED: every paired-end one); `embed` refuses a
# sample without those its bundle takes. The adapter-trimmed share may be missing (no adapter found), the
# DNA estimate is missing where it is 0 (dna_below_limit), and the Centrifuge value is missing when the
# workflow ran with `centrifuge: false`: it is then set to the training mean, with a warning.
QC_REQUIRED = ("length_q05_r1", "length_q50_r1", "length_q95_r1", "cycles_r1", "pct_kept", "duplication_rate",
               "gc_content", "strand_purity", "leakage_b", "intergenic_share", "gene_body_slope", "pct_short_features",
               "log_substitution_rate", "gc_slope", "gc_curv")
QC_PAIRED_REQUIRED = ("length_q50_r2", "fragment_q25", "fragment_q50", "fragment_q75")


def check_qc(qc: pd.DataFrame | None, ids: list, names=None) -> list[str]:
    """Refuse (ValueError) a QC table that cannot give the covariates ``names`` (default: all) of the
    samples ``ids``; returns warnings for values that are filled with the training mean."""
    names = set(QC_NUMERIC + QC_BINARY) if names is None else set(names)
    required = tuple(n for n in QC_REQUIRED if n in names)
    paired_required = tuple(n for n in QC_PAIRED_REQUIRED if n in names)
    if qc is None:
        raise ValueError("no QC table: the model takes the samples' QC covariates (use --ingest-results with an inGEST "
                         "run, or --qc with a QC table)")
    lacking = [i for i in ids if i not in qc.index]
    if lacking:
        raise ValueError(f"{len(lacking)} of {len(ids)} samples are not in the QC table: {lacking[:5]}")
    cov = qc_covariates(qc, ids)
    if required and cov[list(required)].isna().all().all():
        raise ValueError("the QC table has none of the QC values the model needs (expected the layout of an inGEST QC "
                         f"record or of sample_qc.tsv, e.g. columns {', '.join(required[:3])})")
    no_layout = [i for i, v in _qc_value(qc.reindex(ids).fillna(""), "layout").items() if not str(v).strip()]
    if no_layout:
        raise ValueError(f"{len(no_layout)} samples have no layout (PE or SE) in the QC table: {no_layout[:5]}")
    problems = []
    for i in ids:
        need = required + (paired_required if cov.at[i, "paired"] == 1 else ())
        miss = [n for n in need if pd.isna(cov.at[i, n])]
        if miss:
            problems.append(f"{i} ({', '.join(miss)})")
    if problems:
        gc_hint = ("; the GC bias (gc_slope, gc_curv) is in QC records made with the reference genome in the inGEST resources "
                   "(annotation/GCF_000005845.2_ASM584v2_genomic.fna): add it and run `ecoli_ingest run -- --forcerun sample_qc`"
                   if any("gc_slope" in p or "gc_curv" in p for p in problems) else "")
        raise ValueError(f"{len(problems)} samples lack QC values that every atlas sample has: " + "; ".join(problems[:5])
                         + (" ..." if len(problems) > 5 else "") + gc_hint)
    no_genus = [i for i in ids if "log_other_genus" in names and pd.isna(cov.at[i, "log_other_genus"])]
    if no_genus:
        return [f"{len(no_genus)} sample(s) have no Centrifuge results (run with centrifuge: false?); their "
                f"other-genus covariate was set to the atlas mean: {', '.join(no_genus[:5])}" + (" ..." if len(no_genus) > 5 else "")]
    return []


DNA_DEFAULT_LENGTH = 150.0    # bp added to a feature's length when a sample has no fragment or read length


def dna_density(qc: pd.DataFrame | None, ids: list, genome_length: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Genomic DNA in every library, from the QC table: (DNA fragments per base and strand, length
    in bp that a fragment adds to a feature's reach).

    density = dna_fraction_est x aligned fragments / (2 x genome length), with the aligned
    fragments taken as reads kept x aligned share, halved for a paired-end library. A feature of
    length L then collects about density x (L + length) DNA reads on its strand, where length is
    the median fragment size (paired-end) or the median read length. The density is 0 where the DNA
    estimate is 0 (below its detection limit) or a value is missing."""
    n = len(ids)
    if qc is None or not genome_length:
        return np.zeros(n), np.full(n, DNA_DEFAULT_LENGTH)
    q = qc.reindex(ids).fillna("")
    num = lambda name: pd.to_numeric(_qc_value(q, name), errors="coerce").to_numpy(dtype=np.float64)       # noqa: E731
    layout = _qc_value(q, "layout").where(_qc_value(q, "layout") != "", q["layout"] if "layout" in q.columns else "")
    paired = layout.str.upper().isin(("PE", "PAIRED")).to_numpy()
    fragments = num("reads_kept") * num("pct_aligned") / 100.0 / np.where(paired, 2.0, 1.0)
    rho = num("dna_fraction_est") * fragments / (2.0 * float(genome_length))
    length = np.where(paired, num("fragment_q50"), num("length_q50_r1"))
    length = np.where(np.isnan(length) | (length <= 0), DNA_DEFAULT_LENGTH, length)
    return np.where(np.isnan(rho) | (rho < 0), 0.0, rho), length


def dna_reads(rho: np.ndarray, length: np.ndarray, feature_lengths: np.ndarray) -> np.ndarray:
    """Expected DNA reads, samples x features."""
    return rho[:, None] * (np.asarray(feature_lengths, dtype=np.float64)[None, :] + length[:, None])


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
    antisense: pd.DataFrame | None = None     # features x samples: reads on the strand a feature is not transcribed from
    time_series: pd.DataFrame | None = None   # indexed by sample: series, x (time_courses.tsv); the model samples among them

    @property
    def ids(self) -> list:
        return list(self.counts.columns)


def read_time_series(path) -> pd.DataFrame:
    """A time-course table (sample, series, x[, x_label]), indexed by sample; one series = one culture line."""
    t = pd.read_csv(path, sep="\t", dtype={"sample": str, "series": str})
    missing = [c for c in ("sample", "series", "x") if c not in t.columns]
    if missing:
        raise ValueError(f"{path}: no column {missing} (expected sample, series, x)")
    if t["sample"].duplicated().any():
        raise ValueError(f"{path}: samples listed twice: {t.loc[t['sample'].duplicated(), 'sample'].tolist()[:5]}")
    t["x"] = pd.to_numeric(t["x"], errors="raise")
    return t.set_index("sample")[["series", "x"]]


def load_tables(counts_path, metadata_path, qc_path, model: str | None = None, antisense_path=None,
                time_series_path=None) -> AtlasTables:
    """The samples the QC table assigns to a model (``in_model``; ``model`` selects one name),
    with their counts and metadata. Fails if a model sample lacks counts or metadata.
    ``antisense_path``: the antisense count table of the same samples (optional; used by the
    missing-gene calls). ``time_series_path``: a time-course table (``read_time_series``); its rows of
    model samples are kept."""
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
    antisense = None
    if antisense_path:
        antisense = read_counts(antisense_path)
        missing = sorted(set(ids) - set(antisense.columns))
        if missing:
            raise ValueError(f"{len(missing)} model samples are not in the antisense count table: {missing[:5]}")
        antisense = antisense[ids]
    series = None
    if time_series_path:
        series = read_time_series(time_series_path)
        series = series.loc[[s for s in series.index if s in set(ids)]]
    return AtlasTables(counts[ids], meta.loc[ids], qc.loc[ids], antisense, series)


def series_codes(ids: list, time_series: pd.DataFrame | None) -> tuple:
    """(series code of every sample, -1 = in none; x of every sample, NaN = none)."""
    code, x = np.full(len(ids), -1), np.full(len(ids), np.nan)
    if time_series is not None and len(time_series):
        pos = {s: i for i, s in enumerate(ids)}
        names = {n: k for k, n in enumerate(pd.unique(time_series["series"]))}
        for s, r in time_series.iterrows():
            if s in pos:
                code[pos[s]], x[pos[s]] = names[r["series"]], r["x"]
    return code, x


def phase_order_pairs(meta: pd.DataFrame, gse: np.ndarray, series: np.ndarray, x: np.ndarray) -> dict:
    """Pairs (i, j) of samples where j is further along the growth curve than i: inside one study
    at different ORDER_LEVELs (source 0), and inside one culture series at different times
    (source 1, samples allowed by ``labels.order_series_ok``). ``group`` = one (source, study)
    each, so that every group can be given the same weight."""
    level, ok = order_levels(meta), order_series_ok(meta)
    i_all, j_all, grp = [], [], []
    groups: dict = {}
    for g in np.unique(gse):
        r = np.where((gse == g) & (level >= 0))[0]
        a, b = np.where(level[r][:, None] < level[r][None, :])
        if len(a):
            groups[(0, g)] = len(groups)
            i_all.append(r[a]); j_all.append(r[b]); grp.append(np.full(len(a), groups[(0, g)]))
    for s in np.unique(series[series >= 0]):
        r = np.where((series == s) & ok)[0]
        a, b = np.where(x[r][:, None] < x[r][None, :])
        if len(a):
            key = (1, gse[r[0]])
            groups.setdefault(key, len(groups))
            i_all.append(r[a]); j_all.append(r[b]); grp.append(np.full(len(a), groups[key]))
    if not i_all:
        return {"i": np.zeros(0, int), "j": np.zeros(0, int), "group": np.zeros(0, int), "source": np.zeros(0, int)}
    grp = np.concatenate(grp)
    source = np.array([k[0] for k in sorted(groups, key=groups.get)])[grp]
    return {"i": np.concatenate(i_all), "j": np.concatenate(j_all), "group": grp, "source": source}


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
    low_pool: np.ndarray | None = None   # the same for counts summed over a study's samples of one strain; None = no pooled calls
    dna_rho: torch.Tensor | None = None  # DNA fragments per base and strand of every sample (``dna_density``); None = no DNA term
    dna_len: torch.Tensor | None = None  # bp a fragment adds to a feature's length, per sample
    series: np.ndarray | None = None     # culture series code of every sample (-1 = none), from the time-course table
    series_x: np.ndarray | None = None   # the sample's time (or OD, or order) in its series
    order_pairs: dict | None = None      # ``phase_order_pairs`` when the phase-order term is on
    extra: dict = field(default_factory=dict)

    def log_dna(self, rt: torch.Tensor) -> torch.Tensor | None:
        """log of the expected DNA reads of the samples ``rt`` (samples x features; -inf where a
        sample has none), or None when the decoder has no DNA term."""
        if self.dna_rho is None:
            return None
        lengths = torch.tensor(np.asarray(self.fs.lengths, dtype=np.float32))      # a copy: the stored lengths are read-only
        return torch.log(self.dna_rho[rt])[:, None] + torch.log(lengths[None, :] + self.dna_len[rt][:, None])

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


def _feature_gc(cfg: TrainConfig, fs: FeatureSet) -> np.ndarray | None:
    """GC content of the model features for the decoder's GC term (None without it)."""
    if not cfg.gc_offset:
        return None
    missing = [c for c in ("gc_slope", "gc_curv") if c not in cfg.covariates]
    if missing:
        raise ValueError(f"the GC term reads the covariates gc_slope and gc_curv; {missing} are not among this model's covariates")
    from .technical import feature_gc_for, load_feature_gc
    return feature_gc_for(fs, load_feature_gc(cfg.feature_gc))


def build_training(tables: AtlasTables, fs: FeatureSet, cfg: TrainConfig, splits: pd.DataFrame,
                   fit_rows: np.ndarray | None = None) -> TrainingData:
    """Everything the model needs. ``fit_rows`` (default: the training pool) are the samples the
    model is fitted on; the statistics (size-factor reference, z-score, covariate scaling) come from
    the reference samples among them, weighted by the study cap of ``fit_rows``."""
    ids = tables.ids
    splits = splits.loc[ids]
    Y, fam, _ = align_counts(tables.counts, fs)
    # genomic DNA: Yin are the counts the encoder inputs and size factors are made from
    rho, dlen = dna_density(tables.qc, ids, fs.genome_length) if (cfg.dna_decoder or cfg.dna_inputs) else (np.zeros(len(ids)), np.zeros(len(ids)))
    Yin = np.maximum(Y - dna_reads(rho, dlen, fs.lengths), 0.0) if cfg.dna_inputs else Y
    gse = tables.meta["GSE"].astype(str).to_numpy()
    fit_rows = np.where(splits["split"].to_numpy() == 1)[0] if fit_rows is None else np.asarray(fit_rows)
    weights, cap = study_weights(gse, fit_rows, cfg.study_cap)
    rows = fit_rows[splits["reference"].to_numpy(dtype=bool)[fit_rows]]
    if not len(rows):
        raise ValueError("no reference sample among the samples to fit on")
    ref_log = reference_profile(Yin[rows], fs.is_mrna, weights[rows], cfg.min_gene_presence)
    s = size_factors(Yin, ref_log)
    low, low_pool, miss = None, None, np.zeros(Y.shape, dtype=bool)
    anti = None if tables.antisense is None else tables.antisense.reindex(fs.ids, fill_value=0).to_numpy(dtype=np.float64).T
    if cfg.mask_missing:
        if fs.order is None:
            raise ValueError("the feature set has no chromosome order; the missing-gene calls need it")
        veto = lambda r: {} if anti is None else {"antisense": anti[r], "lengths": fs.lengths, "veto_reads": cfg.mask_veto_reads,       # noqa: E731
                                                  "veto_fraction": cfg.mask_veto_fraction}
        low = low_expression_profile(Y[rows], s[rows], fs.order, cfg.mask_min_expected, cfg.mask_low_fraction, cfg.mask_quantile, **veto(rows))
        miss = missing_genes(Y, s, low, fs.order, cfg.mask_min_expected, cfg.mask_low_fraction, **veto(slice(None)))
        if cfg.mask_pool:
            # groups and the pooled profile from the reference training samples, the calls for every sample
            call = {"min_expected": cfg.mask_min_expected, "low_fraction": cfg.mask_low_fraction}
            ref_groups = call_groups(miss[rows], gse[rows], cfg.mask_pool_min_calls, cfg.mask_pool_jaccard)
            low_pool = pooled_low_profile(Y[rows], s[rows], ref_groups, fs.order, quantile=cfg.mask_quantile, **call, **veto(rows))
            groups = call_groups(miss, gse, cfg.mask_pool_min_calls, cfg.mask_pool_jaccard)
            miss = miss | group_calls(Y, s, groups, low_pool, fs.order, **call, **veto(slice(None)))
        s = size_factors(Yin, ref_log, miss)
    logx = log_expression(Yin, s)
    zscore = ZScore.fit(logx[rows], weights[rows], cfg.sd_floor, miss[rows])
    cov = covariate_frame(Y, fam, fs, tables.qc, ids)
    unknown = [c for c in cfg.covariates if c not in cov.columns]
    if unknown:
        raise ValueError(f"unknown covariates {unknown}; known: {list(cov.columns)}")
    scaler = CovariateScaler.fit(cov[list(cfg.covariates)], rows, weights[rows])
    labels = head_labels(tables.meta)
    w = weights[rows] / weights[rows].sum()
    have = w[:, None] * ~miss[rows]                    # a gene's mean is taken over the samples that have it
    total = have.sum(axis=0)
    log_mean = np.log(np.maximum((have * (Yin[rows] / s[rows][:, None])).sum(axis=0) / np.where(total > 0, total, 1.0), 1e-3))
    assert set(label_codes(labels)) == set(HEADS)
    series, series_x = series_codes(ids, tables.time_series)
    gse_codes = pd.factorize(gse)[0]
    pairs = phase_order_pairs(tables.meta, gse_codes, series, series_x) if cfg.phase_order_weight > 0 else None
    return TrainingData(ids=ids, fs=fs, meta=tables.meta, labels=labels, splits=splits,
                        X=torch.tensor(zscore.apply(logx, miss)), Y=torch.tensor(Y, dtype=torch.float32),
                        log_s=torch.tensor(np.log(s), dtype=torch.float32), C=torch.tensor(scaler.apply(cov)),
                        y=label_codes(labels), y_eval=label_codes(scored_labels(tables.meta, labels)),
                        y_fit=fit_codes(tables.meta, labels, cfg.phase_head), fit_classes={h: fit_classes(h, cfg.phase_head) for h in HEADS},
                        phase_level=phase_levels(tables.meta), gse_codes=gse_codes, identity=condition_identity(tables.meta),
                        weights=weights, study_cap=cap, ref_log=ref_log, zscore=zscore, scaler=scaler,
                        log_mean=log_mean.astype(np.float32), M=torch.tensor(miss), low=low, low_pool=low_pool,
                        dna_rho=torch.tensor(rho, dtype=torch.float32) if cfg.dna_decoder else None,
                        dna_len=torch.tensor(dlen, dtype=torch.float32) if cfg.dna_decoder else None,
                        series=series, series_x=series_x, order_pairs=pairs,
                        extra={"feature_gc": _feature_gc(cfg, fs), "covariates_raw": cov, "antisense_veto": bool(cfg.mask_missing and anti is not None),
                               "dna_samples": int((rho > 0).sum())})


def prepared(cache: dict | None, key, tables: AtlasTables, fs: FeatureSet, cfg: TrainConfig, splits: pd.DataFrame,
             fit_rows: np.ndarray | None = None) -> TrainingData:
    """``build_training``, kept in ``cache`` under ``key``. Nothing in the training data depends on
    the model seed (missing-gene calls, size factors, statistics, labels), so the seeds of a sweep
    and their fold fits share it. The data is read, never changed, by the fits."""
    if cache is None:
        return build_training(tables, fs, cfg, splits, fit_rows=fit_rows)
    if key not in cache:
        cache[key] = build_training(tables, fs, cfg, splits, fit_rows=fit_rows)
    return cache[key]
