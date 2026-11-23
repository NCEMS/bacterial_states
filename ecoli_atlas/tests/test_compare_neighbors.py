"""`ecoli_atlas compare` (new samples against the atlas of a bundle) and `neighbors` (studies of the atlas)."""
import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.compare import Atlas, calls_frame, compare, group_labels, group_order, knn, novelty, query_distances, read_embedding
from ecoli_atlas.compare_plots import group_colors, is_ordered, neighbor_composition
from ecoli_atlas.config import HParams
from ecoli_atlas.io import read_counts
from ecoli_atlas.neighbors import select_query, split_buckets

QUERY = ["GSM0000", "GSM0001", "GSM0005", "GSM0006", "GSM0100", "GSM0101", "GSM0200"]


@pytest.fixture(scope="module")
def atlas(trained, atlas_files):
    return Atlas.load(trained, atlas_files["metadata"])


@pytest.fixture(scope="module")
def embedded(trained, atlas_files, tmp_path_factory):
    """`embed` output of seven atlas samples under new names, with a sample table."""
    d = tmp_path_factory.mktemp("query")
    counts = read_counts(atlas_files["counts"])[QUERY]
    counts.columns = [f"new{i}" for i in range(len(QUERY))]
    counts.to_csv(d / "counts.tsv", sep="\t")
    qc = pd.read_csv(atlas_files["qc"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM").loc[QUERY]
    qc.index = pd.Index(counts.columns, name="GSM")
    qc.to_csv(d / "qc.tsv", sep="\t")
    assert main(["embed", "--counts", str(d / "counts.tsv"), "--qc", str(d / "qc.tsv"), "--models", str(trained), "--out", str(d / "embedded"),
                 "--device", "cpu"]) == 0
    meta = pd.DataFrame({"time_h": [3, 3, 24, 24, 168, 168, 3], "library": ["depleted"] * 6 + ["total"],
                         "stage": ["a_late", "a_late", "z_early", "z_early", "m_mid", "m_mid", "a_late"]},
                        index=pd.Index(counts.columns, name="sample"))
    meta.to_csv(d / "samples.tsv", sep="\t")
    return {"dir": d / "embedded", "meta": d / "samples.tsv", "frame": meta,
            "query": read_embedding(d / "embedded"), "calls": calls_frame(pd.read_csv(d / "embedded" / "predictions.tsv", sep="\t", index_col=0))}


def test_atlas_of_a_bundle(atlas):
    assert atlas.embedding.shape == (119, 27) and list(atlas.layout["block"]) == ["phase", "medium", "lab_strain", "residual"]
    assert {"gse", "phase", "medium", "lab_strain", "strain_group", "growth_phase"} <= set(atlas.metadata.columns)
    assert set(atlas.metadata["phase"].dropna()) == {"exponential", "stationary"} and atlas.metadata["phase"].isna().any()
    assert set(atlas.predictions["phase_pred"]) <= {"exponential", "stationary"} and atlas.predictions["phase_prob_stationary"].between(0, 1).all()
    assert len(atlas.block_columns(("phase", "residual"))) == HParams().block_dims["phase"] + HParams().res_dim
    with pytest.raises(ValueError, match="unknown blocks"):
        atlas.block_columns(("nope",))
    assert atlas.label_source("medium").notna().sum() < 119          # curated labels, missing where a sample has none


def test_knn_exclusions(atlas):
    q = atlas.embedding.iloc[:5]
    plain = knn(q, atlas, k=5)
    assert (plain[plain["rank"] == 1]["atlas_sample"].values == q.index.values).all()      # default: self first
    assert (plain[plain["rank"] == 1]["distance"] < 1e-6).all()
    no_self = knn(q, atlas, k=5, exclude_self=True)
    assert len(no_self) == 25 and not any(no_self["query"] == no_self["atlas_sample"])
    studies = atlas.metadata["gse"].reindex(q.index)
    no_study = knn(q, atlas, k=5, exclude_studies=studies)
    assert len(no_study) == 25 and not any(no_study["gse"].astype(str) == no_study["query"].map(studies.to_dict()).astype(str))
    nov_plain = novelty(q, atlas, k=3)
    nov_masked = novelty(q, atlas, k=3, query_studies=studies)
    assert (nov_masked["novelty_knn_dist"] >= nov_plain["novelty_knn_dist"] - 1e-9).all()
    assert (nov_masked["novelty_knn_dist"] > nov_plain["novelty_knn_dist"]).any()


def test_group_labels_and_order(embedded):
    qmeta = embedded["frame"]
    lab = group_labels(qmeta, "time_h")
    assert list(lab) == ["3", "3", "24", "24", "168", "168", "3"]
    assert group_order(lab, qmeta) == ["3", "24", "168"]                     # numeric, not "168" < "24" < "3"
    by_stage = group_labels(qmeta, "stage")
    assert group_order(by_stage, qmeta) == ["a_late", "m_mid", "z_early"]
    assert group_order(by_stage, qmeta, "time_h") == ["a_late", "z_early", "m_mid"]      # ordered by another column
    assert group_labels(qmeta, ("time_h", "library")).iloc[-1] == "3 | total"
    with pytest.raises(ValueError, match="lacks column"):
        group_labels(qmeta, "nope")
    with pytest.raises(ValueError, match="order-by"):
        group_order(lab, qmeta, "nope")


def test_compare_tables(atlas, embedded, tmp_path):
    query, calls, qmeta = embedded["query"], embedded["calls"], embedded["frame"]
    plain = compare(query, atlas, tmp_path / "plain", k=5, query_predictions=calls)
    # an embedded atlas sample finds itself first, at distance 0, and nothing is new about it
    assert list(plain["nearest_atlas_sample"]) == QUERY and (plain["nearest_distance"] < 1e-4).all()
    assert {"top_study", "phase_knn_vote", "phase_knn_frac", "phase_encoder_pred", "phase_agree", "medium_knn_vote", "lab_strain_knn_vote",
            "residual_knn_dist", "phase_knn_dist", "novelty_knn_dist", "novelty_percentile"} <= set(plain.columns)
    assert not (tmp_path / "plain" / "group_summary.tsv").exists() and not list((tmp_path / "plain").glob("fig_*"))
    nb = pd.read_csv(tmp_path / "plain" / "neighbors.tsv", sep="\t")
    assert len(nb) == 7 * 5 and {"gse", "phase", "strain_group", "atlas_phase_pred"} <= set(nb.columns)
    D = query_distances(query, atlas)
    assert D.shape == (7, 7) and np.allclose(D.values, D.values.T) and np.allclose(np.diag(D.values), 0)
    assert not np.allclose(query_distances(query, atlas, blocks=("phase",)).values, D.values)
    # sample metadata only adds columns
    with_meta = compare(query, atlas, tmp_path / "meta", k=5, query_predictions=calls, metadata=qmeta)
    assert list(with_meta.columns[:3]) == ["time_h", "library", "stage"]
    pd.testing.assert_frame_equal(with_meta[plain.columns], plain)
    assert (tmp_path / "plain" / "neighbors.tsv").read_text() == (tmp_path / "meta" / "neighbors.tsv").read_text()
    # groups
    out = tmp_path / "grp"
    summary = compare(query, atlas, out, k=5, query_predictions=calls, metadata=qmeta, group_by="time_h", order_by="time_h")
    assert list(summary["group"]) == ["3", "3", "24", "24", "168", "168", "3"]
    g = pd.read_csv(out / "group_summary.tsv", sep="\t", index_col=0)
    assert [str(x) for x in g.index] == ["3", "24", "168"] and list(g["n"]) == [3, 2, 2]
    for c in ("phase_prob_stationary", "phase_knn_vote", "phase_knn_frac", "phase_agree", "medium_prob_minimal", "top_study", "top_studies",
              "mean_nearest_distance", "mean_novelty_percentile", "within_group_distance", "distance_to_previous"):
        assert c in g.columns, c
    members = [s for s in query.index if qmeta.loc[s, "time_h"] == 24]
    assert np.isclose(g.loc[24, "phase_prob_stationary"], calls.loc[members, "phase_prob_stationary"].mean(), atol=1e-5)
    assert np.isclose(g.loc[24, "within_group_distance"], D.loc[members[0], members[1]], atol=1e-5)
    assert np.isnan(g["distance_to_previous"].iloc[0]) and g["distance_to_previous"].iloc[1:].notna().all()
    assert {p.stem for p in out.glob("fig_*.png")} == {"fig_query_distances", "fig_head_by_group", "fig_neighbor_labels"}     # no UMAP requested
    with pytest.raises(ValueError, match="metadata lacks rows"):
        compare(query, atlas, tmp_path / "x", k=5, metadata=qmeta.iloc[:-1], group_by="time_h")
    with pytest.raises(ValueError, match="need --metadata"):
        compare(query, atlas, tmp_path / "y", k=5, group_by="time_h")
    with pytest.raises(ValueError, match="do not match"):
        compare(query.iloc[:, :10], atlas, tmp_path / "z", k=5)


def test_plot_helpers(atlas, embedded):
    assert is_ordered(["3", "24", "168"]) and not is_ordered(["a", "3"])
    assert len(set(group_colors(["3", "24", "168"]).values())) == 3
    cats = group_colors([f"c{i}" for i in range(10)])
    assert cats["c8"] == cats["c9"] and cats["c0"] != cats["c1"]                 # ninth and later share "other"
    nb = knn(embedded["query"], atlas, k=5)
    tab = neighbor_composition(nb, nb["gse"], group_labels(embedded["frame"], "time_h"), ["3", "24", "168"])
    assert list(tab.index) == ["3", "24", "168"] and np.allclose(tab.sum(axis=1), 1.0)


def test_compare_command(trained, atlas_files, embedded, tmp_path):
    base = ["compare", "--embeddings", str(embedded["dir"]), "--models", str(trained), "-k", "5"]
    assert main(base + ["--out", str(tmp_path / "plain")]) == 0                    # without any metadata
    plain = pd.read_csv(tmp_path / "plain" / "summary.tsv", sep="\t", index_col=0)
    assert list(plain["nearest_atlas_sample"]) == QUERY and "phase_agree" in plain.columns
    rc = main(base + ["--atlas-metadata", str(atlas_files["metadata"]), "--out", str(tmp_path / "out"), "--metadata", str(embedded["meta"]),
                      "--group-by", "library", "--formats", "png", "--umap"])
    assert rc == 0
    g = pd.read_csv(tmp_path / "out" / "group_summary.tsv", sep="\t", index_col=0)
    assert list(g.index) == ["depleted", "total"] and np.isnan(g.loc["total", "within_group_distance"])
    assert (tmp_path / "out" / "fig_neighbor_labels.png").exists() and not (tmp_path / "out" / "fig_neighbor_labels.pdf").exists()
    assert (tmp_path / "out" / "fig_umap_groups.png").exists() and len(pd.read_csv(tmp_path / "out" / "umap_atlas.tsv", sep="\t")) == 119
    with pytest.raises(SystemExit, match="lacks column"):
        main(base + ["--out", str(tmp_path / "bad"), "--metadata", str(embedded["meta"]), "--group-by", "nope"])


def test_neighbors_command(trained, atlas_files, atlas, tmp_path):
    q, gse, rep = select_query(atlas, ["GSE01", "GSE99"])
    assert rep == {"found": {"GSE01": 10}, "missing": ["GSE99"]} and len(q) == 10 and (gse == "GSE01").all()
    splits = pd.read_csv(trained / "splits.csv", index_col=0)
    b = split_buckets(splits, ["GSE02"], atlas.embedding.index)
    assert set(b) <= {"test", "fold1", "fold2", "fold3", "train_only", "projected", "withheld"} and (b == "test").any()
    assert split_buckets(None, None, ["a"]).loc["a"] == "no_split"
    test_study = splits.loc[splits["split"] == 3, "gse"].iloc[0]
    train_study = splits.loc[splits["split"] == 1, "gse"].iloc[0]
    out = tmp_path / "nb"
    rc = main(["neighbors", test_study, train_study, "--models", str(trained), "--metadata", str(atlas_files["metadata"]),
               "--out", str(out), "-k", "5", "--umap"])
    assert rc == 0
    summary = pd.read_csv(out / "summary.tsv", sep="\t", index_col=0)
    nb = pd.read_csv(out / "neighbors.tsv", sep="\t")
    study = pd.read_csv(out / "study_summary.tsv", sep="\t", index_col=0)
    assert len(summary) == 20 and len(nb) == 20 * 5 and set(study.index) == {test_study, train_study}
    assert set(summary.loc[summary["query_gse"] == test_study, "query_split_bucket"]) <= {"test", "projected"}
    assert not (nb["gse"].astype(str) == nb["query_gse"].astype(str)).any()      # same-study neighbours excluded
    assert {"medium_knn_vote", "medium_meta", "medium_agree_meta", "medium_agree_pred", "novelty_percentile", "residual_knn_dist",
            "umap1", "top_studies", "query_strain_group", "atlas_split_bucket" if False else "nearest_gse"} <= set(summary.columns)
    assert "atlas_split_bucket" in nb.columns
    assert {"top_neighbor_studies", "medium_dominant_vote", "mean_novelty_percentile", "split_buckets"} <= set(study.columns)
    assert (out / "umap.png").exists() and (out / "umap_atlas.tsv").exists()
    man = json.loads((out / "neighbors_manifest.json").read_text())
    assert man["found"] == {test_study: 10, train_study: 10} and man["exclude_same_study"] is True
    # own-study neighbours can only be equal or closer; the sample itself stays excluded
    assert main(["neighbors", test_study, "--models", str(trained), "--out", str(tmp_path / "w"), "-k", "3", "--within-study"]) == 0
    s2 = pd.read_csv(tmp_path / "w" / "summary.tsv", sep="\t", index_col=0)
    nb2 = pd.read_csv(tmp_path / "w" / "neighbors.tsv", sep="\t")
    assert not (nb2["atlas_sample"] == nb2["query"]).any() and (nb2["gse"] == test_study).any()
    assert main(["neighbors", test_study, "--models", str(trained), "--out", str(tmp_path / "w2"), "-k", "3"]) == 0
    s3 = pd.read_csv(tmp_path / "w2" / "summary.tsv", sep="\t", index_col=0)
    assert (s2["nearest_distance"].values <= s3.reindex(s2.index)["nearest_distance"].values + 1e-9).all()
    # missing studies
    assert main(["neighbors", "GSE999999", "--models", str(trained), "--out", str(tmp_path / "m")]) == 1
    assert main(["neighbors", test_study, "GSE999999", "--models", str(trained), "--out", str(tmp_path / "m2"), "--allow-missing"]) == 0
    assert main(["neighbors", "--models", str(trained), "--out", str(tmp_path / "m3")]) == 2
