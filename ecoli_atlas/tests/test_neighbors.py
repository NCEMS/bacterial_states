"""`compare.knn` exclusions and `ecoli_atlas neighbors` on the synthetic atlas."""
import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.compare import Atlas, knn, layout_from_columns, novelty
from ecoli_atlas.labels import collapse_runs, gsm_accession, oxygen_class, rows_of_gsm
from ecoli_atlas.neighbors import collapse_atlas, select_query, split_buckets


@pytest.fixture
def random_atlas():
    rng = np.random.RandomState(0)
    ids = [f"GSM{i // 2}_SRR{i}" for i in range(30)]           # two runs per GSM
    cols = [f"growth_phase[{i}]" for i in range(4)] + [f"residual[{i}]" for i in range(6)]
    emb = pd.DataFrame(rng.normal(size=(30, 10)), index=pd.Index(ids, name="sample"), columns=cols)
    meta = pd.DataFrame({"gse": [f"GSE{i // 6}" for i in range(30)], "growth_phase": ["exponential", "stationary"] * 15,
                         "OD600": rng.uniform(0.2, 2, 30)}, index=emb.index)
    preds = pd.DataFrame({"growth_phase_pred": ["exponential"] * 30, "growth_phase_confidence": rng.uniform(0.5, 1, 30)}, index=emb.index)
    return Atlas(emb, preds, meta, layout_from_columns(cols), None)


def test_labels_helpers():
    idx = ["GSM1_SRR1", "GSM1_SRR2", "GSM2_SRR3", "plain"]
    acc = gsm_accession(idx)
    assert list(acc.values[:3]) == ["GSM1", "GSM1", "GSM2"] and pd.isna(acc.values[3])
    assert rows_of_gsm(idx) == {"GSM1": [0, 1], "GSM2": [2]}
    df = pd.DataFrame({"x": [1.0, 3.0, 5.0, 7.0], "s": ["a", "b", "c", "d"]}, index=idx)
    c = collapse_runs(df)
    assert list(c.index) == ["GSM1", "GSM2", "plain"] and c.loc["GSM1", "x"] == 2.0 and c.loc["GSM1", "s"] == "a"
    assert collapse_runs(df, how="first").loc["GSM1", "x"] == 1.0
    assert oxygen_class("Aerobic ") == "aerobic" and oxygen_class("anaerobic") == "low"
    assert oxygen_class("aeorbic") is None and oxygen_class(np.nan) is None


def test_knn_exclusions(random_atlas):
    atlas = random_atlas
    q = atlas.embedding.iloc[:5]
    plain = knn(q, atlas, k=5)
    assert (plain[plain["rank"] == 1]["atlas_sample"].values == q.index.values).all()      # default: self first
    assert (plain[plain["rank"] == 1]["distance"] < 1e-9).all()
    no_self = knn(q, atlas, k=5, exclude_self=True)
    assert len(no_self) == 25 and not any(no_self["query"] == no_self["atlas_sample"])
    studies = atlas.metadata["gse"].reindex(q.index)
    no_study = knn(q, atlas, k=5, exclude_studies=studies)
    own = no_study["query"].map(studies.to_dict())
    assert len(no_study) == 25 and not any(no_study["gse"].astype(str) == own.astype(str))
    # restricted to two studies only 12 rows remain -> fewer than k neighbours, no fake rows
    small = Atlas(atlas.embedding.iloc[:12], atlas.predictions.iloc[:12], atlas.metadata.iloc[:12], atlas.layout, None)
    few = knn(small.embedding.iloc[:1], small, k=10, exclude_studies=small.metadata["gse"].iloc[:1])
    assert len(few) == 6 and (few["gse"] == "GSE1").all()
    nov_plain = novelty(q, atlas, k=3)
    nov_masked = novelty(q, atlas, k=3, query_studies=studies)
    assert (nov_masked["novelty_knn_dist"] >= nov_plain["novelty_knn_dist"] - 1e-12).all()
    assert (nov_masked["novelty_knn_dist"] > nov_plain["novelty_knn_dist"]).any()


def test_collapse_select_buckets(random_atlas):
    atlas = random_atlas
    c = collapse_atlas(atlas)
    assert len(c.embedding) == 15 and list(c.embedding.index)[:2] == ["GSM0", "GSM1"]
    np.testing.assert_allclose(c.embedding.loc["GSM0"].values, atlas.embedding.iloc[:2].mean().values)
    assert c.metadata.loc["GSM0", "gse"] == "GSE0" and c.predictions.loc["GSM0", "growth_phase_pred"] == "exponential"
    q, gse, rep = select_query(c, ["GSE1", "GSE9"])
    assert rep == {"found": {"GSE1": 3}, "missing": ["GSE9"]} and len(q) == 3 and (gse == "GSE1").all()
    splits = pd.DataFrame({"gse": atlas.metadata["gse"], "split_base": 1, "cv_fold_base": np.nan,
                           "split": [3] * 6 + [1] * 12 + [0] * 12, "cv_fold": [np.nan] * 6 + [1] * 6 + [np.nan] * 18}, index=atlas.embedding.index)
    b = split_buckets(splits, "GSE4", c.embedding.index)
    assert b.loc["GSM0"] == "test" and b.loc["GSM3"] == "fold1" and b.loc["GSM6"] == "train(aug)"
    assert b.loc["GSM9"] == "projected_only" and b.loc["GSM12"] == "withheld"
    assert split_buckets(None, None, ["a"]).loc["a"] == "no_metadata"


def test_neighbors_cli(trained_models, tmp_path):
    atlas = trained_models / "atlas"
    out = tmp_path / "nb"
    rc = main(["neighbors", "GSE1009", "GSE1012", "--atlas", str(atlas), "--models", str(trained_models), "--out", str(out), "-k", "5", "--umap"])
    assert rc == 0
    summary = pd.read_csv(out / "summary.tsv", sep="\t", index_col=0)
    nb = pd.read_csv(out / "neighbors.tsv", sep="\t")
    study = pd.read_csv(out / "study_summary.tsv", sep="\t", index_col=0)
    assert len(summary) == 24 and len(nb) == 24 * 5 and list(study.index) == ["GSE1009", "GSE1012"]
    # case studies are forced into TEST; their stressed (non-reference) samples are projected only, never TRAIN
    buckets = set(summary["query_split_bucket"])
    assert "test" in buckets and buckets <= {"test", "projected_only"}
    assert not (nb["gse"].astype(str) == nb["query_gse"].astype(str)).any()      # same-study neighbours excluded
    assert not (nb["atlas_sample"] == nb["query"]).any()
    assert {"growth_phase_knn_vote", "growth_phase_agree_meta", "growth_phase_agree_pred", "novelty_percentile",
            "residual_knn_dist", "umap1", "top_studies"} <= set(summary.columns)
    assert {"top_neighbor_studies", "growth_phase_dominant_vote", "mean_novelty_percentile"} <= set(study.columns)
    assert (out / "umap.png").exists() and (out / "umap_atlas.csv").exists()
    man = json.loads((out / "neighbors_manifest.json").read_text())
    assert man["found"] == {"GSE1009": 12, "GSE1012": 12} and man["withheld_study"] == "GSE1013"
    # the withheld study is reported as such; --within-study allows own-study neighbours; --no-collapse-runs keeps rows
    rc = main(["neighbors", "GSE1013", "--atlas", str(atlas), "--models", str(trained_models), "--out", str(tmp_path / "w"), "-k", "3",
               "--within-study", "--no-collapse-runs"])
    assert rc == 0
    s2 = pd.read_csv(tmp_path / "w" / "summary.tsv", sep="\t", index_col=0)
    nb2 = pd.read_csv(tmp_path / "w" / "neighbors.tsv", sep="\t")
    assert set(s2["query_split_bucket"]) == {"withheld"} and len(s2) == 12 and s2.index[0].startswith("GSM") and "_SRR" in s2.index[0]
    assert not (nb2["atlas_sample"] == nb2["query"]).any()                      # self is still excluded
    rc = main(["neighbors", "GSE1013", "--atlas", str(atlas), "--models", str(trained_models), "--out", str(tmp_path / "w2"), "-k", "3",
               "--no-collapse-runs"])
    assert rc == 0
    s3 = pd.read_csv(tmp_path / "w2" / "summary.tsv", sep="\t", index_col=0)
    nb3 = pd.read_csv(tmp_path / "w2" / "neighbors.tsv", sep="\t")
    assert not (nb3["gse"] == "GSE1013").any()                                  # default: own study excluded
    # allowing own-study neighbours can only bring equal or closer ones
    assert (s2["nearest_distance"].values <= s3.reindex(s2.index)["nearest_distance"].values + 1e-12).all()
    # missing studies
    assert main(["neighbors", "GSE999999", "--atlas", str(atlas), "--out", str(tmp_path / "m")]) == 1
    assert main(["neighbors", "GSE1009", "GSE999999", "--atlas", str(atlas), "--out", str(tmp_path / "m2"), "--allow-missing"]) == 0
    assert main(["neighbors", "--atlas", str(atlas), "--out", str(tmp_path / "m3")]) == 2
