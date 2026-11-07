"""End-to-end smoke test on synthetic data: train -> embed -> compare -> verify-models via the CLI."""
import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main

N_STUDIES, PER_STUDY, N_GENES = 14, 12, 60


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory, feature_table):
    d = tmp_path_factory.mktemp("synthetic")
    rng = np.random.RandomState(1)
    samples, rows = [], []
    for s in range(N_STUDIES):
        gse = f"GSE{1000 + s}"
        lab = s % 3 != 0
        for j in range(PER_STUDY):
            sid = f"GSM{s * 100 + j}_SRR{s * 100 + j}"
            phase = "exponential" if j % 2 == 0 else "late_stationary"
            medium = "complex" if s % 2 == 0 else "defined"
            samples.append(sid)
            rows.append({"sample_id": sid, "gse": gse, "strain": "MG1655" if lab else f"clinical{s}",
                         "strain_background": "K-12" if lab else "clinical isolate", "is_lab_strain": lab,
                         "medium_type": medium, "base_medium": "LB" if medium == "complex" else "M9",
                         "growth_phase": phase, "OD600": 0.5 if phase == "exponential" else 2.0,
                         "carbon_source": np.nan if medium == "complex" else "glucose", "temperature(C)": 37,
                         "oxygen": "aerobic", "culture_vessel": "flask", "culture_volume(mL)": 50,
                         "stress": np.nan if j < 10 else "heat", "stress_type": np.nan, "stress_subtype": np.nan})
    meta = pd.DataFrame(rows).set_index("sample_id")
    # counts: b-number keyed genes x samples, with a signal that depends on phase and medium
    base = rng.gamma(2.0, 200, size=N_GENES)
    X = np.zeros((N_GENES, len(samples)))
    for i, sid in enumerate(samples):
        m = meta.loc[sid]
        eff = np.ones(N_GENES)
        eff[:15] *= 3.0 if m["growth_phase"] == "exponential" else 0.4
        eff[15:30] *= 2.5 if m["medium_type"] == "defined" else 0.5
        eff[30:40] *= 2.0 if m["is_lab_strain"] else 0.6
        X[:, i] = rng.poisson(base * eff * rng.uniform(0.7, 1.3))
    genes = [f"b{i:04d}" for i in range(N_GENES)] + ["b9001", "b9003"]  # rRNA, tRNA
    X = np.vstack([X, rng.poisson(5000, size=(2, len(samples)))])
    counts = pd.DataFrame(X, index=pd.Index(genes, name="Geneid"), columns=samples)
    counts.to_csv(d / "merged_count_table.txt", sep="\t")
    meta.to_csv(d / "metadata_revised.csv")
    feature_table.to_csv(d / "feature_table.txt", sep="\t", index=False)
    return d


def test_train_embed_compare_verify(synthetic, tmp_path):
    models = tmp_path / "models"
    rc = main(["train", "--counts", str(synthetic / "merged_count_table.txt"),
               "--metadata", str(synthetic / "metadata_revised.csv"),
               "--feature-table", str(synthetic / "feature_table.txt"),
               "--out", str(models), "--no-tune", "--factor-epochs", "2", "--top-epochs", "2",
               "--min-class-n", "5", "--withheld-study", "GSE1013", "--case-studies", "GSE1012",
               "--test-frac", "0.15", "--device", "cpu", "--write-atlas"])
    assert rc == 0
    manifest = json.loads((models / "manifest.json").read_text())
    assert manifest["factor_order"] == ["growth_phase", "medium_type", "is_lab_strain"]
    assert manifest["final_embedding_dim"] == sum(manifest["factor_emb_dims"].values()) + manifest["residual_dim"]
    assert manifest["split_config"]["withheld_study"] == "GSE1013"
    splits = pd.read_csv(models / "splits.csv", index_col=0)
    assert set(splits["split"].unique()) <= {0, 1, 3}
    assert (splits.loc[splits["gse"] == "GSE1013", "split"] == 0).all()        # withheld study never trained on
    cs = splits.loc[splits["gse"] == "GSE1012", "split"]                        # case study forced into TEST:
    assert (cs == 3).any() and not (cs == 1).any()                              # reference samples TEST, never TRAIN
    assert not (set(splits.loc[splits["split"] == 1, "gse"]) & set(splits.loc[splits["split"] == 3, "gse"]))
    atlas = tmp_path / "atlas"
    assert (atlas / "final_embedding.csv").exists() and (atlas / "atlas_metadata.csv").exists()
    emb = pd.read_csv(atlas / "final_embedding.csv", index_col=0)
    assert emb.shape == (N_STUDIES * PER_STUDY, manifest["final_embedding_dim"])

    # embed a subset of "new" samples (plain TSV) with the trained bundle
    new = pd.read_csv(synthetic / "merged_count_table.txt", sep="\t", index_col=0).iloc[:, :6]
    new.columns = [f"new{i}" for i in range(6)]
    new.to_csv(tmp_path / "new_counts.tsv", sep="\t")
    out = tmp_path / "embed"
    rc = main(["embed", "--counts", str(tmp_path / "new_counts.tsv"), "--models", str(models), "--out", str(out),
               "--feature-table", str(synthetic / "feature_table.txt"), "--device", "cpu"])
    assert rc == 0
    e = pd.read_csv(out / "final_embedding.csv", index_col=0)
    assert e.shape == (6, manifest["final_embedding_dim"])
    np.testing.assert_allclose(e.values, emb.values[:6], atol=1e-4)  # same samples -> same embedding

    cmp_out = tmp_path / "compare"
    rc = main(["compare", "--embeddings", str(out / "final_embedding.csv"), "--atlas", str(atlas), "--out", str(cmp_out), "-k", "5"])
    assert rc == 0
    summary = pd.read_csv(cmp_out / "summary.tsv", sep="\t", index_col=0)
    nb = pd.read_csv(cmp_out / "neighbors.tsv", sep="\t")
    assert len(summary) == 6 and len(nb) == 6 * 5
    assert (summary["nearest_distance"] < 1e-5).all()  # each new sample is itself in the atlas
    assert {"growth_phase_knn_vote", "medium_type_knn_vote", "top_study", "novelty_percentile", "residual_knn_dist"} <= set(summary.columns)
    assert summary["growth_phase_agree"].dtype == bool
    assert ((summary["novelty_percentile"] >= 0) & (summary["novelty_percentile"] <= 1)).all()

    rc = main(["verify-models", "--old", str(models), "--new", str(models), "--counts", str(synthetic / "merged_count_table.txt"),
               "--feature-table", str(synthetic / "feature_table.txt"), "--metadata", str(synthetic / "metadata_revised.csv"),
               "--report", str(tmp_path / "report.json"), "--device", "cpu"])
    assert rc == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["genes_identical"] and rep["embedding_max_abs_diff"] == 0.0
    assert all(v == 1.0 for v in rep["prediction_agreement"].values())
    assert "growth_phase" in rep["test_metrics"]["new"]


def test_min_gene_coverage_gate(synthetic, tmp_path, bundle):
    models = tmp_path / "m"
    bundle.save(models)
    # count table with only half of the model's genes
    counts = pd.DataFrame(np.ones((30, 3)), index=[f"g{i:03d}" for i in range(30)], columns=["a", "b", "c"])
    counts.index.name = "gene"
    counts.to_csv(tmp_path / "half.tsv", sep="\t")
    rc = main(["embed", "--counts", str(tmp_path / "half.tsv"), "--models", str(models), "--out", str(tmp_path / "e"),
               "--feature-table", str(synthetic / "feature_table.txt"), "--device", "cpu"])
    assert rc == 1
    assert (tmp_path / "e" / "gene_coverage.tsv").exists()
    rc = main(["embed", "--counts", str(tmp_path / "half.tsv"), "--models", str(models), "--out", str(tmp_path / "e2"),
               "--feature-table", str(synthetic / "feature_table.txt"), "--device", "cpu", "--min-gene-coverage", "0.4"])
    assert rc == 0
