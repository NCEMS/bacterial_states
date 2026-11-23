import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main


def test_split_only_then_figures(train_args, atlas_files, tmp_path):
    out = tmp_path / "split"
    assert main(["train", *train_args, "--out", str(out), "--split-only", "--test-frac", "0.2"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["splits.csv"]
    figs = tmp_path / "figs"
    assert main(["figures", "--metadata", str(atlas_files["metadata"]), "--splits", str(out), "--out", str(figs), "--formats", "png"]) == 0
    assert (figs / "fig_split_balance.png").stat().st_size > 10000 and (figs / "fig_corpus_composition.png").stat().st_size > 10000
    table = pd.read_csv(figs / "split_balance.tsv", sep="\t")
    samples = table[(table["unit"] == "samples") & (table["label"] == "phase")]
    splits = pd.read_csv(out / "splits.csv", index_col=0)
    in_split = splits["reference"] & splits["split"].isin([1, 3])            # only reference samples are in a fold or TEST
    assert 0 < in_split.sum() < 119 and samples["n"].sum() == in_split.sum()
    assert set(samples["bucket"]) == {"fold1", "fold2", "fold3", "test"}
    assert (samples.groupby("bucket")["proportion"].sum().round(3) == 1.0).all()
    assert set(table["class"]) == {"exponential", "stationary", "rich", "minimal", "non_lab", "lab", "Unlabeled"}
    studies = table[(table["unit"] == "studies") & (table["label"] == "medium")].drop_duplicates("bucket")
    assert studies["total"].sum() == splits.loc[in_split, "gse"].nunique()
    comp = json.loads((figs / "corpus_composition.json").read_text())
    assert comp["n_samples"] == 119 and set(comp["composition"]["phase"]["wedges"]) == {"exponential", "stationary"}
    assert "none" not in comp["composition"]["stress_family"]["wedges"]
    assert main(["figures", "--metadata", str(atlas_files["metadata"]), "--splits", str(out), "--out", str(figs), "--only", "nope"]) == 2


def test_model_figures(trained, train_args, tmp_path):
    figs = tmp_path / "figs"
    args = ["figures", *train_args, "--models", str(trained), "--out", str(figs), "--formats", "png", "--device", "cpu", "--strict"]
    assert main(args) == 0
    for name in ("split_balance", "corpus_composition", "pca_mg1655_batch", "pca_bw_batch", "pca_strain_all_complex", "batch_genes",
                 "head_test_eval", "head_scores", "head_gene_correlates", "study_leakage", "threshold_curves"):
        assert (figs / f"fig_{name}.png").stat().st_size > 3000, name
    m = json.loads((figs / "figure_metrics.json").read_text())
    assert m["embedding_check"]["max_abs_diff"] < 1e-4                      # the bundle's embedding.tsv is made again
    assert m["test_metrics_vs_manifest"] and all(v["agree"] for v in m["test_metrics_vs_manifest"].values())
    assert set(m["study_leakage"]["rates"]) == {"input", "blocks", "embedding"}
    assert m["head_scores"]["phase"]["auroc"]["train"] > 0.8 and (figs / "head_scores.tsv").exists()
    # phase raises the first ten genes of the synthetic atlas: they lead the correlates of the phase call
    corr = pd.read_csv(figs / "head_gene_correlations.tsv", sep="\t")
    top = corr[corr["head"] == "phase"].sort_values("spearman_rho", ascending=False).head(10)
    assert set(top["gene"]) <= {f"g{i:03d}" for i in range(10)} and "Top 50 Positive" in (figs / "head_gene_correlations.txt").read_text()
    # the bundle has no cv_predictions.tsv: the folds were fitted once and kept beside the figures
    cv = pd.read_csv(figs / "cv_predictions.tsv", sep="\t", index_col=0)
    assert m["cv_predictions"] == str(figs / "cv_predictions.tsv") and set(cv["cv_fold"]) == {1, 2, 3}
    assert 0.02 <= m["threshold_curves"]["medium"]["best_threshold"] <= 0.98
    stamp = (figs / "cv_predictions.tsv").stat().st_mtime_ns
    assert main(args + ["--only", "threshold_curves"]) == 0 and (figs / "cv_predictions.tsv").stat().st_mtime_ns == stamp
    assert main(["figures", "--metadata", train_args[3], "--out", str(figs), "--only", "study_leakage"]) == 2      # no bundle given


def test_case_studies(trained, train_args, atlas_files, tmp_path):
    import argparse

    import numpy as np

    from ecoli_atlas.casestudy.glu_cas import run_glu_cas
    from ecoli_atlas.casestudy.oxygen import run_oxygen
    from ecoli_atlas.casestudy.uti import run_uti
    from ecoli_atlas.cli import _figure_context
    from ecoli_atlas.figures.plotting import apply_style
    out = tmp_path / "case"
    # the synthetic atlas has none of the studies of the case studies
    assert main(["casestudy", "uti", *train_args, "--models", str(trained), "--out", str(out), "--device", "cpu"]) == 1
    assert "error" in json.loads((out / "casestudy_metrics.json").read_text())["uti"]
    ctx, _, _ = _figure_context(argparse.Namespace(models=str(trained), device="cpu", counts=str(atlas_files["counts"]),
                                                   metadata=str(atlas_files["metadata"]), qc=str(atlas_files["qc"]),
                                                   features=str(atlas_files["features"])))
    apply_style()
    test_studies = sorted(set(ctx.gse[ctx.split == 3]) - {"GSE07"})          # GSE07 has treated samples
    train_study = ctx.gse[ctx.split == 1][0]
    meta = ctx.data.meta
    rows = np.where(ctx.gse == test_studies[0])[0]
    # oxygen: half of a TEST study made anaerobic
    meta.loc[meta.index[rows[::2]], "oxygen"] = "anaerobic"
    res = run_oxygen(ctx, out, ("png",), studies=[test_studies[0]])
    assert res["n_samples"] == len(rows) and res["per_study"][test_studies[0]]["n_low"] == len(rows[::2])
    assert 0 <= res["auroc_leave_one_out"] <= 1 and (out / "fig_oxygen_lda_projection.png").exists()
    try:
        run_oxygen(ctx, out, ("png",), studies=[train_study])
        raise AssertionError("a training study must be refused")
    except ValueError:
        pass
    # uti: five isolates with one sample in the host and one in urine each
    meta["strain_name"], meta["in_host"] = "", "false"
    for k in range(5):
        meta.loc[meta.index[rows[2 * k:2 * k + 2]], "strain_name"] = f"HM{k}"
        meta.loc[meta.index[rows[2 * k]], "in_host"] = "true"
    res = run_uti(ctx, out, ("png",), study=test_studies[0], n_perm=20)
    assert res["n_patients"] == 5 and res["conditions"] == {"UTI": 5, "UR": 5} and 0 <= res["correct"] <= 5
    # glu-cas: the two media of a training study, paired by growth phase; the metrics say it was fitted
    tr = np.where(ctx.gse == "GSE00")[0]                                      # exponential and stationary samples
    meta.loc[meta.index[tr[::2]], "amino_acids_supplied"] = "true"
    meta.loc[meta.index[tr[1::2]], "amino_acids_supplied"] = "false"
    res = run_glu_cas(ctx, out, ("png",), study="GSE00", n_perm=20)
    assert res["fitted_by_the_model"] == (ctx.split[tr] == 1).any() and res["lda"]["treatments"] == {"CAS": 5, "GLU": 5}
    assert res["paired"]["n_pairs"] == 2 and (out / "fig_glu_cas_lda_projection.png").exists()
    assert not list(out.glob("*axis_genes*.png")) and not (out / "fig_glu_cas_axis.png").exists() and not (out / "fig_uti_axis.png").exists()
    assert (out / "uti_axis_genes.tsv").exists() and (out / "glu_cas_gene_correlations.tsv").exists()   # the gene tables stay


def test_case_study_pathway_analysis(tmp_path):
    """The pathway analysis of a case-study axis: preranked GSEA of the gene ranking against a gene-set table, positive
    NES for a set at the top of the ranking, one table and figure per kind of gene set."""
    pytest.importorskip("gseapy")
    from ecoli_atlas.casestudy.pathways import pathway_analysis
    genes = [f"g{i:03d}" for i in range(200)]
    rho = pd.Series(np.linspace(1, -1, 200), index=genes)
    sets = pd.DataFrame({"term": ["kegg:top", "imodulon:bottom", "kegg:mixed"], "namespace": ["kegg_pathway", "imodulon_x", "kegg_pathway"],
                         "name": ["top set", "bottom set", "mixed set"],
                         "genes": [",".join(genes[:15]), ",".join(genes[-15:]), ",".join(genes[::13])]})
    sets.to_csv(tmp_path / "modules.tsv", sep="\t", index=False)
    res = pathway_analysis(rho, tmp_path, "oxygen", "low oxygen", "aerobic", ("png",), False, tmp_path / "modules.tsv", None, 200, lambda *a: None)
    t = pd.read_csv(tmp_path / "oxygen_gsea_modules.tsv", sep="\t").set_index("term")
    assert t.loc["kegg:top", "nes"] > 0 > t.loc["imodulon:bottom", "nes"] and t.loc["imodulon:bottom", "namespace"] == "imodulon_x"
    assert t.loc["kegg:top", "database"] == "KEGG pathway" and t.loc["imodulon:bottom", "database"] == "iModulon"
    assert "gsea_modules" in res and "gsea_go" not in res                 # one figure per database
    assert (tmp_path / "fig_oxygen_gsea_kegg_pathway.png").exists() and (tmp_path / "fig_oxygen_gsea_imodulon.png").exists()


def test_batch_tracking_plot(tmp_path):
    """fig_batch_tracking from a per-study table: panels ordered by the between-study spread summed over the strains,
    one y span for every panel, r of each set with the ribosomal proteins; the top sets from the batch GSEA tables."""
    from ecoli_atlas.figures.batch_tracking import plot_batch_tracking, spreads, top_sets
    rng = np.random.default_rng(0)
    rib = rng.uniform(11, 16, 12)
    t = pd.DataFrame({"strain": ["MG1655"] * 8 + ["BW25113"] * 4, "gse": [f"G{i}" for i in range(12)], "n": rng.integers(1, 9, 12),
                      "regulondb:RpoS+": 20 - rib + rng.normal(0, 0.1, 12), "regulondb:Crp+": 15 - 0.5 * rib,
                      "regulondb:FliA+": np.where(np.arange(12) % 3 == 0, 12.0, 5.0), "ribosome": rib})
    res = plot_batch_tracking(t, tmp_path, ("png",), {"regulondb:RpoS+": "RpoS (activated)"})
    assert list(res["sets"]) == list(spreads(t, ["regulondb:RpoS+", "regulondb:Crp+", "regulondb:FliA+"]).index)
    assert list(res["sets"])[0] == "regulondb:FliA+" and res["sets"]["regulondb:RpoS+"]["r_with_ribosome"] < -0.9
    assert res["y_span_log2"] == pytest.approx(8.0) and (tmp_path / "fig_batch_tracking.png").exists()
    assert res["sets"]["regulondb:RpoS+"]["r2"] > 0.95 and res["sets"]["regulondb:FliA+"]["r2"] < 0.3
    assert res["sets"]["regulondb:RpoS+"]["residual_sd"] < 0.2 < res["sets"]["regulondb:FliA+"]["residual_sd"]
    assert top_sets(tmp_path) is None                                        # no GSEA tables
    for s, nes in (("mg1655", [3.0, 2.0, 1.0]), ("bw25113", [1.0, 2.5, 4.0])):
        pd.DataFrame({"term": ["regulondb:A+", "regulondb:B-", "kegg:x"], "database": ["RegulonDB", "RegulonDB", "KEGG pathway"],
                      "nes": nes, "fdr": [0.01, 0.01, 0.01]}).to_csv(tmp_path / f"gsea_batch_modules_{s}.tsv", sep="\t", index=False)
    top = top_sets(tmp_path)
    assert list(top["term"]) == ["regulondb:B-", "regulondb:A+"] and top["nes_sum"].tolist() == [4.5, 4.0]
    t2 = t.rename(columns={"regulondb:RpoS+": "regulondb:B-", "regulondb:Crp+": "regulondb:A+"}).drop(columns="regulondb:FliA+")
    res = plot_batch_tracking(t2, tmp_path, ("png",), nes=top)
    assert list(res["sets"]) == ["regulondb:B-", "regulondb:A+"]                   # in the order of the NES sum
