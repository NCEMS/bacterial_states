"""`ecoli_atlas compare` with sample metadata: groups, ordering, distances between the new samples."""
import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.compare import (Atlas, compare, group_labels, group_order, layout_from_columns, query_distances)
from ecoli_atlas.compare_plots import group_colors, is_ordered, neighbor_composition
from ecoli_atlas.io import read_metadata


@pytest.fixture
def atlas_and_query():
    rng = np.random.RandomState(1)
    ids = [f"GSM{i}_SRR{i}" for i in range(40)]
    cols = [f"growth_phase[{i}]" for i in range(4)] + [f"residual[{i}]" for i in range(6)]
    emb = pd.DataFrame(rng.normal(size=(40, 10)), index=pd.Index(ids, name="sample"), columns=cols)
    meta = pd.DataFrame({"gse": [f"GSE{i // 8}" for i in range(40)], "growth_phase": ["exponential", "stationary"] * 20}, index=emb.index)
    preds = pd.DataFrame({"growth_phase_pred": ["exponential"] * 40}, index=emb.index)
    atlas = Atlas(emb, preds, meta, layout_from_columns(cols), None)
    qids = [f"t{h:03d}h_rep{r}" for h in (3, 24, 168) for r in (1, 2)] + ["t003h_rep3_total"]
    query = pd.DataFrame(rng.normal(size=(len(qids), 10)), index=pd.Index(qids, name="sample"), columns=cols)
    p = rng.uniform(0, 1, len(qids))
    qpred = pd.DataFrame({"growth_phase_pred": np.where(p > 0.5, "stationary", "exponential"),
                          "growth_phase_prob_exponential": 1 - p, "growth_phase_prob_stationary": p}, index=query.index)
    qmeta = pd.DataFrame({"time_h": [3, 3, 24, 24, 168, 168, 3], "library": ["depleted"] * 6 + ["total"],
                          "phase": ["a_late", "a_late", "z_early", "z_early", "m_mid", "m_mid", "a_late"]}, index=query.index)
    return atlas, query, qpred, qmeta


def test_group_labels_and_order(atlas_and_query):
    _, _, _, qmeta = atlas_and_query
    lab = group_labels(qmeta, "time_h")
    assert list(lab) == ["3", "3", "24", "24", "168", "168", "3"]
    assert group_order(lab, qmeta) == ["3", "24", "168"]                     # numeric, not "168" < "24" < "3"
    assert group_order(lab, qmeta, "time_h") == ["3", "24", "168"]
    by_phase = group_labels(qmeta, "phase")
    assert group_order(by_phase, qmeta) == ["a_late", "m_mid", "z_early"]
    assert group_order(by_phase, qmeta, "time_h") == ["a_late", "z_early", "m_mid"]      # ordered by another column
    two = group_labels(qmeta, ("time_h", "library"))
    assert two.iloc[0] == "3 | depleted" and two.iloc[-1] == "3 | total"
    with pytest.raises(ValueError, match="lacks column"):
        group_labels(qmeta, "nope")
    with pytest.raises(ValueError, match="order-by"):
        group_order(lab, qmeta, "nope")


def test_query_distances(atlas_and_query):
    atlas, query, _, _ = atlas_and_query
    D = query_distances(query, atlas)
    assert D.shape == (7, 7) and list(D.index) == list(query.index) == list(D.columns)
    assert np.allclose(D.values, D.values.T) and np.allclose(np.diag(D.values), 0) and (D.values >= 0).all()
    sub = query_distances(query, atlas, blocks=("growth_phase",))
    assert not np.allclose(sub.values, D.values)


def test_compare_default_outputs_unchanged(atlas_and_query, tmp_path):
    atlas, query, qpred, qmeta = atlas_and_query
    plain = compare(query, atlas, tmp_path / "plain", k=5, query_predictions=qpred)
    assert not (tmp_path / "plain" / "group_summary.tsv").exists()
    assert not list((tmp_path / "plain").glob("fig_*"))
    with_meta = compare(query, atlas, tmp_path / "meta", k=5, query_predictions=qpred, metadata=qmeta)
    assert list(with_meta.columns[:3]) == ["time_h", "library", "phase"]
    pd.testing.assert_frame_equal(with_meta[plain.columns], plain)
    assert (tmp_path / "plain" / "neighbors.tsv").read_text() == (tmp_path / "meta" / "neighbors.tsv").read_text()


def test_group_summary(atlas_and_query, tmp_path):
    atlas, query, qpred, qmeta = atlas_and_query
    out = tmp_path / "grp"
    summary = compare(query, atlas, out, k=5, query_predictions=qpred, metadata=qmeta, group_by="time_h", order_by="time_h")
    assert list(summary["group"]) == ["3", "3", "24", "24", "168", "168", "3"]
    g = pd.read_csv(out / "group_summary.tsv", sep="\t", index_col=0)
    assert [str(x) for x in g.index] == ["3", "24", "168"] and list(g["n"]) == [3, 2, 2]
    for c in ("growth_phase_called_exponential", "growth_phase_called_stationary", "growth_phase_prob_stationary",
              "growth_phase_knn_vote", "growth_phase_knn_frac", "growth_phase_agree", "top_study", "top_studies",
              "mean_nearest_distance", "mean_novelty_percentile", "within_group_distance", "distance_to_previous"):
        assert c in g.columns, c
    assert np.allclose(g["growth_phase_called_exponential"] + g["growth_phase_called_stationary"], 1.0)
    members = [s for s in query.index if qmeta.loc[s, "time_h"] == 24]
    assert np.isclose(g.loc[24, "growth_phase_prob_stationary"], qpred.loc[members, "growth_phase_prob_stationary"].mean())
    D = query_distances(query, atlas)
    assert np.isclose(g.loc[24, "within_group_distance"], D.loc[members[0], members[1]])
    assert np.isnan(g["distance_to_previous"].iloc[0]) and g["distance_to_previous"].iloc[1:].notna().all()
    figs = {p.stem for p in out.glob("fig_*.png")}
    assert figs == {"fig_query_distances", "fig_factor_by_group", "fig_neighbor_labels"}       # no UMAP requested


def test_compare_metadata_errors(atlas_and_query, tmp_path):
    atlas, query, qpred, qmeta = atlas_and_query
    with pytest.raises(ValueError, match="metadata lacks rows"):
        compare(query, atlas, tmp_path / "x", k=5, metadata=qmeta.iloc[:-1], group_by="time_h")
    with pytest.raises(ValueError, match="need --metadata"):
        compare(query, atlas, tmp_path / "y", k=5, group_by="time_h")


def test_plot_helpers(atlas_and_query):
    atlas, query, _, qmeta = atlas_and_query
    assert is_ordered(["3", "24", "168"]) and not is_ordered(["a", "3"])
    ramp = group_colors(["3", "24", "168"])
    assert len(set(ramp.values())) == 3
    cats = group_colors([f"c{i}" for i in range(10)])
    assert cats["c8"] == cats["c9"] and cats["c0"] != cats["c1"]                 # ninth and later share "other"
    from ecoli_atlas.compare import knn
    nb = knn(query, atlas, k=5)
    tab = neighbor_composition(nb, nb["gse"], group_labels(qmeta, "time_h"), ["3", "24", "168"])
    assert list(tab.index) == ["3", "24", "168"] and np.allclose(tab.sum(axis=1), 1.0)


def test_cli_compare_groups(atlas_and_query, tmp_path):
    atlas, query, qpred, qmeta = atlas_and_query
    adir = tmp_path / "atlas"
    adir.mkdir()
    atlas.embedding.to_csv(adir / "final_embedding.csv")
    atlas.metadata.to_csv(adir / "atlas_metadata.csv")
    atlas.predictions.to_csv(adir / "sample_predictions.csv")
    qdir = tmp_path / "q"
    qdir.mkdir()
    query.to_csv(qdir / "final_embedding.csv")
    qpred.to_csv(qdir / "sample_predictions.csv")
    qmeta.to_csv(qdir / "samples.tsv", sep="\t")                                 # the layout `prepare` writes
    assert list(read_metadata(qdir / "samples.tsv").columns) == ["time_h", "library", "phase"]
    rc = main(["compare", "--embeddings", str(qdir / "final_embedding.csv"), "--atlas", str(adir), "--out", str(tmp_path / "out"),
               "-k", "5", "--metadata", str(qdir / "samples.tsv"), "--group-by", "library", "--formats", "png"])
    assert rc == 0
    g = pd.read_csv(tmp_path / "out" / "group_summary.tsv", sep="\t", index_col=0)
    assert list(g.index) == ["depleted", "total"] and np.isnan(g.loc["total", "within_group_distance"])
    assert (tmp_path / "out" / "fig_neighbor_labels.png").exists() and not (tmp_path / "out" / "fig_neighbor_labels.pdf").exists()
    with pytest.raises(SystemExit, match="lacks column"):
        main(["compare", "--embeddings", str(qdir / "final_embedding.csv"), "--atlas", str(adir), "--out", str(tmp_path / "bad"),
              "--metadata", str(qdir / "samples.tsv"), "--group-by", "nope"])
