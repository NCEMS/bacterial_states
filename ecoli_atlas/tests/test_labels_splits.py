import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.config import TrainConfig
from ecoli_atlas.data import load_tables, qc_covariates, study_weights
from ecoli_atlas.labels import is_reference, condition_identity, head_labels, label_codes, phase_check_group
from ecoli_atlas.splits import bucket_labels, check_no_leakage, make_splits


@pytest.fixture(scope="module")
def tables(atlas_files):
    return load_tables(atlas_files["counts"], atlas_files["metadata"], atlas_files["qc"])


def test_load_tables_takes_the_model_samples(tables, atlas_files):
    assert len(tables.ids) == 119 and "GSM1100" not in tables.ids          # in_model empty for that one
    with pytest.raises(ValueError, match="no sample is marked"):
        load_tables(atlas_files["counts"], atlas_files["metadata"], atlas_files["qc"], model="other")


def test_phase_labels_leave_out_late_stationary_and_treated(tables):
    lab = head_labels(tables.meta)
    gp = tables.meta["growth_phase"]
    assert lab["phase"][gp == "late_stationary"].isna().all()               # no label ...
    assert (phase_check_group(tables.meta)[gp == "late_stationary"] == "").all()   # ... and not a check group either
    assert lab["phase"][gp == "transition"].isna().all()
    assert set(phase_check_group(tables.meta)[gp == "transition"]) == {"transition"}
    treated = tables.meta["treated"] == "true"
    assert lab["phase"][treated].isna().all() and lab["medium"][treated].notna().all()
    assert set(lab["phase"].dropna()) == {"exponential", "stationary"} and set(lab["medium"].dropna()) == {"rich", "minimal"}
    codes = label_codes(lab)
    assert set(codes["phase"]) == {-1, 0, 1} and (codes["phase"][lab["phase"] == "stationary"] == 1).all()


def test_condition_identity(tables):
    ident = condition_identity(tables.meta)
    m = tables.meta
    a, b = m.index.get_loc("GSM0000"), m.index.get_loc("GSM0002")          # same study, both wild type, same phase
    c = m.index.get_loc("GSM0001")                                         # the deletion strain
    assert ident["cond_id"][a] == ident["cond_id"][b] != ident["cond_id"][c]
    assert ident["is_ctrl"][a] and not ident["is_ctrl"][c] and ident["is_annot"].all()
    other = m.index.get_loc("GSM0400")                                     # another study, same control condition
    assert ident["ctrl_cond"][a] == ident["ctrl_cond"][other]
    assert not ident["is_ctrl"][m.index.get_loc("GSM0701")]                # treated


def test_splits_keep_studies_whole_and_honour_forced_and_withheld(tables):
    cfg = TrainConfig(seed=1, test_frac=0.2, n_candidates=50, withheld_studies=("GSE03",), test_studies=("GSE04",))
    ref = is_reference(tables.meta)
    assert ref.any() and not ref.all() and not ref[tables.meta.index.get_loc("GSM0701")]      # treated
    sp = make_splits(tables.meta["GSE"], head_labels(tables.meta), ref, cfg)
    assert set(sp.loc[sp["gse"] == "GSE03", "split"]) == {0}
    # a TEST study: its reference samples are TEST, its other samples are projected only
    forced = sp[sp["gse"] == "GSE04"]
    assert set(forced.loc[forced["reference"], "split"]) == {3} and set(forced.loc[~forced["reference"], "split"]) == {0}
    assert sp.loc[sp["split"] == 3, "reference"].all()
    # the training pool: whole studies, folds for the reference samples only
    pool = sp[sp["split"] == 1]
    assert not set(pool["gse"]) & set(sp.loc[sp["split"] != 1, "gse"])
    assert set(pool.loc[pool["reference"], "cv_fold"]) == {1.0, 2.0, 3.0} and pool.loc[~pool["reference"], "cv_fold"].isna().all()
    assert (pool[pool["reference"]].groupby("gse")["cv_fold"].nunique() == 1).all()
    assert set(bucket_labels(sp)) == {"fold1", "fold2", "fold3", "test", "train_only", "projected"}
    bad = sp.copy()
    bad.loc[bad.index[bad["gse"] == "GSE04"][0], "split"] = 1
    with pytest.raises(ValueError, match="leakage"):
        check_no_leakage(bad, cfg)


def test_a_study_too_large_for_test_stays_in_training():
    gse = pd.Series(["BIG"] * 60 + [f"S{i}" for i in range(40) for _ in range(2)], index=[f"x{i}" for i in range(140)])
    labels = pd.DataFrame({"phase": ["exponential", "stationary"] * 70, "medium": ["rich", "minimal"] * 70,
                           "lab_strain": ["lab", "non_lab"] * 70}, index=gse.index)
    sp = make_splits(gse, labels, np.ones(len(gse), bool), TrainConfig(seed=3, n_candidates=50, withheld_studies=(), test_studies=()))
    assert set(sp.loc[sp["gse"] == "BIG", "split"]) == {1}


def test_study_weights_cap_the_largest_study():
    gse = np.array(["a"] * 10 + ["b"] * 4 + ["c"] * 2)
    w, cap = study_weights(gse, np.arange(16), None)
    assert cap == 4 and np.allclose(w[:10], 0.4) and np.allclose(w[10:], 1.0)
    assert study_weights(gse, np.arange(16), 5)[0][0] == 0.5


def test_qc_covariates_missing_values_and_flags(tables):
    cov = qc_covariates(tables.qc, ["GSM0000", "GSM0001", "GSM0100", "not_there"])
    assert cov.loc["GSM0000", "paired"] == 1 and cov.loc["GSM0100", "paired"] == 0
    assert np.isnan(cov.loc["GSM0100", "fragment_q50"]) and cov.loc["GSM0000", "fragment_q50"] == 200
    assert cov.loc["GSM0000", "strand_purity"] == 0.97 and cov.loc["GSM0000", "instrument_novaseq"] == 1
    assert cov.loc["GSM0001", "dna_below_limit"] == 1 and np.isnan(cov.loc["GSM0001", "log_dna_fraction"])
    assert np.isclose(cov.loc["GSM0000", "log_dna_fraction"], np.log10(0.004))
    assert cov.loc["not_there", "has_qc"] == 0 and cov.loc["GSM0000", "has_qc"] == 1 and np.isnan(cov.loc["not_there", "gc_content"])
    assert qc_covariates(None, ["x"]).loc["x", "has_qc"] == 0


def test_label_weights_cap_every_study_within_a_class():
    from ecoli_atlas.data import label_weights
    from ecoli_atlas.losses import class_weights
    # class 1: study a has 12 samples, b 3, c 2; class 0: a 2, b 6; two samples without a label, one outside the rows
    gse = np.array(["a"] * 12 + ["b"] * 3 + ["c"] * 2 + ["a"] * 2 + ["b"] * 6 + ["a", "c", "b"])
    y = np.array([1] * 17 + [0] * 8 + [-1, -1, 1])
    rows = np.arange(27)
    w, caps = label_weights(y, gse, rows)
    assert caps == {0: 2, 1: 3}                                   # the second-largest study of each class
    assert np.allclose(w[:12], 3 / 12) and np.allclose(w[12:17], 1.0) and np.allclose(w[17:19], 1.0) and np.allclose(w[19:25], 2 / 6)
    assert (w[25:] == 0).all()                                    # no label, or not among the rows
    assert np.isclose(w[:12].sum(), 3.0)                          # study a counts as 3 samples of class 1
    fixed, caps = label_weights(y, gse, rows, cap=1)
    assert caps == {0: 1, 1: 1} and np.isclose(fixed[(gse == "b") & (y == 0)].sum(), 1.0)
    plain, capped = class_weights(y, 2), class_weights(y, 2, sample_weight=w)
    assert plain.shape == capped.shape == (2,) and not np.allclose(plain, capped)
