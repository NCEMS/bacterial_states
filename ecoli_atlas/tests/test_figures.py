"""`ecoli_atlas figures` on the synthetic corpus and its one-epoch bundle."""
import json

import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.figures import FIGURES, figure_outputs
from ecoli_atlas.figures.checks import all_pass, compare_to_manifest


def _figures_args(synthetic, trained_models, out, *extra):
    return ["figures", "--counts", str(synthetic / "merged_count_table.txt"),
            "--metadata", str(synthetic / "metadata_revised.csv"),
            "--feature-table", str(synthetic / "feature_table.txt"),
            "--models", str(trained_models), "--out", str(out), "--device", "cpu", "--threads", "2", *extra]


def test_list(capsys):
    assert main(["figures", "--list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in FIGURES)


def test_full_run_matches_manifest(synthetic, trained_models, tmp_path):
    out = tmp_path / "figs"
    rc = main(_figures_args(synthetic, trained_models, out, "--atlas", str(trained_models / "atlas")))
    assert rc == 0
    for f in figure_outputs():
        assert (out / f).exists(), f
    side = json.loads((out / "figure_metrics.json").read_text())
    assert set(side["figures"]) == set(FIGURES) and not side["failed"]
    # same weights, same split -> TEST metrics identical to the manifest, embedding identical to the atlas.
    # (The study-mixing ordinal check is a property of a trained model, not of a one-epoch synthetic fit.)
    failed = {k: v for k, v in side["checks"].items() if v["pass"] is False and k != "leakage.top_ae_below_input"}
    assert not failed, failed
    manifest = json.loads((trained_models / "manifest.json").read_text())
    for f, m in manifest["test_metrics"].items():
        assert side["metrics"]["test_metrics"][f]["n"] == m["n"]
        assert abs(side["metrics"]["test_metrics"][f]["auroc"] - m["auroc"]) < 1e-9
    assert side["zscore_max_abs_dev"] == 0.0
    assert side["atlas_comparison"]["max_abs_diff"] < 1e-4
    # out-of-fold refits: same rows as the manifest's threshold diagnostics
    for f, d in manifest["threshold_diagnostics"].items():
        assert side["metrics"]["threshold_diagnostics"][f]["n"] == d["n"]
    oof = pd.read_csv(out / "oof_scores.csv")
    assert set(oof["factor"]) == set(manifest["factor_order"])
    txt = (out / "factor_gene_correlations.txt").read_text()
    assert txt.startswith("Gene Correlations for Disentangled Factors\n") and "FACTOR: growth_phase" in txt
    long = pd.read_csv(out / "factor_gene_correlations.tsv", sep="\t")
    assert set(long["factor"]) == set(manifest["factor_order"]) and long["spearman_rho"].abs().max() <= 1.0 + 1e-9
    assert (out / "growth_phase_misclassified.tsv").exists()


def test_only_skip_and_sidecar_merge(synthetic, trained_models, tmp_path):
    out = tmp_path / "figs"
    rc = main(_figures_args(synthetic, trained_models, out, "--only", "corpus_composition,factor_test_eval", "--formats", "png"))
    assert rc == 0
    assert (out / "fig_corpus_composition.png").exists() and not (out / "fig_corpus_composition.pdf").exists()
    assert not (out / "fig_split_balance.png").exists()
    side = json.loads((out / "figure_metrics.json").read_text())
    assert set(side["figures"]) == {"corpus_composition", "factor_test_eval"}
    # a second run into the same directory keeps the earlier figures in the sidecar
    rc = main(_figures_args(synthetic, trained_models, out, "--only", "split_balance", "--formats", "png"))
    assert rc == 0
    side = json.loads((out / "figure_metrics.json").read_text())
    assert set(side["figures"]) == {"corpus_composition", "factor_test_eval", "split_balance"}
    assert "test_metrics" in side["metrics"]
    # --skip and --no-retrain
    rc = main(_figures_args(synthetic, trained_models, tmp_path / "f2", "--no-retrain", "--skip",
                            "factor_umaps,pca_batch,batch_genes,factor_gene_correlates,study_leakage", "--formats", "png"))
    assert rc == 0
    side = json.loads((tmp_path / "f2" / "figure_metrics.json").read_text())
    assert set(side["figures"]) == {"corpus_composition", "factor_test_eval", "split_balance"}
    assert not (tmp_path / "f2" / "oof_scores.csv").exists()


def test_unknown_figure_and_reference_dirs(synthetic, trained_models, tmp_path):
    with pytest.raises(SystemExit):
        main(_figures_args(synthetic, trained_models, tmp_path / "x", "--only", "nope"))
    with pytest.raises(SystemExit):   # never write into the bundle ...
        main(_figures_args(synthetic, trained_models, trained_models))
    with pytest.raises(SystemExit):   # ... or next to it in <models>/../figures (the Colab reference location)
        main(_figures_args(synthetic, trained_models, trained_models.parent / "figures"))


def test_recombat_missing(synthetic, trained_models, tmp_path):
    pytest.importorskip("matplotlib")
    try:
        import reComBat  # noqa: F401
        pytest.skip("reComBat installed")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        main(_figures_args(synthetic, trained_models, tmp_path / "rc", "--only", "study_leakage", "--recombat"))
    rc = main(_figures_args(synthetic, trained_models, tmp_path / "rc2", "--only", "study_leakage", "--formats", "png"))
    assert rc == 0 and (tmp_path / "rc2" / "fig_study_leakage.png").exists()


def test_compare_to_manifest_unit():
    manifest = {"test_metrics": {"growth_phase": {"n": 10, "auroc": 0.9, "balanced_acc": 0.8, "ba_ci": [0.7, 0.9]}},
                "threshold_diagnostics": {"growth_phase": {"n": 50, "auroc": 0.85, "balanced_acc": 0.8}}}
    metrics = {"test_metrics": {"growth_phase": {"n": 10, "auroc": 0.9005, "balanced_acc": 0.805, "ba_ci": [0.71, 0.9]}},
               "threshold_diagnostics": {"growth_phase": {"n": 50, "auroc_refit": 0.80, "balanced_acc_refit": 0.79}},
               "atlas_embedding_max_abs_diff": 1e-5,
               "leakage": {"rates": {"top_ae": {"mean": 0.1}, "input_zcpm": {"mean": 0.3}}}}
    checks = compare_to_manifest(metrics, manifest)
    assert checks["test.growth_phase.n"]["pass"] and checks["test.growth_phase.auroc"]["pass"]
    assert checks["oof.growth_phase.auroc"]["pass"] is False          # 0.80 vs 0.85 > 0.03
    assert checks["leakage.top_ae_below_input"]["pass"] and not all_pass(checks)
    sweep = {"cv_spread": {"growth_phase": {"auroc": {"min": 0.7, "max": 0.9, "n": 8}, "balanced_acc": {"min": 0.6, "max": 0.85, "n": 8}}}}
    checks = compare_to_manifest(metrics, manifest, sweep=sweep)
    assert checks["oof.growth_phase.auroc"]["pass"] and "seed sweep" in checks["oof.growth_phase.auroc"]["tolerance"]
    assert all_pass(checks)
    assert all_pass(compare_to_manifest({}, None))
