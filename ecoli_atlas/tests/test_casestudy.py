"""`ecoli_atlas casestudy` on the synthetic corpus: the oxygen contrast, a UTI-shaped paired design and
GLU/CAS on synthetic inference data. Small n: files and metric keys are asserted, not significance."""
import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.casestudy.axis import center_by_group, lopo_axis, oriented_lda_score, paired_axis_test
from ecoli_atlas.cli import main
from conftest import OXYGEN_STUDIES, uti_titles


def _args(synthetic, trained_models, out, which, *extra):
    return ["casestudy", which, "--counts", str(synthetic / "merged_count_table.txt"),
            "--metadata", str(synthetic / "metadata_revised.csv"),
            "--feature-table", str(synthetic / "feature_table.txt"),
            "--models", str(trained_models), "--out", str(out), "--device", "cpu", "--formats", "png", *extra]


def test_axis_helpers():
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(8), 2)
    y = np.tile([0, 1], 8)
    E = rng.normal(size=(16, 5))
    E[y == 1, 0] += 3.0                      # a clear axis along dimension 0
    Ec = center_by_group(E, groups)
    assert np.allclose(Ec.reshape(8, 2, 5).mean(1), 0)
    oof = lopo_axis(Ec, y, groups)
    assert np.isfinite(oof).all() and (oof[y == 1] > oof[y == 0]).mean() > 0.9
    res = paired_axis_test(Ec, y, groups, n_perm=50, seed=1)
    assert res["n_pairs"] == 8 and res["correct"] >= 7 and 0 <= res["p_perm"] <= 1
    s = oriented_lda_score(E, y)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_oxygen(synthetic, trained_models, tmp_path):
    out = tmp_path / "cs"
    rc = main(_args(synthetic, trained_models, out, "oxygen", "--oxygen-studies", ",".join(OXYGEN_STUDIES)))
    assert rc == 0
    for f in ("fig_oxygen_lda_projection.png", "fig_oxygen_axis_genes.png", "oxygen_axis_genes.tsv", "oxygen_samples.tsv",
              "oxygen_metrics.json", "casestudy_metrics.json"):
        assert (out / f).exists(), f
    m = json.loads((out / "oxygen_metrics.json").read_text())
    assert m["studies"] == list(OXYGEN_STUDIES) and set(m["per_study"]) == set(OXYGEN_STUDIES)
    assert 0 <= m["pooled_auroc"] <= 1 and len(m["toward_low_oxygen"]) == 10
    samples = pd.read_csv(out / "oxygen_samples.tsv", sep="\t")
    assert set(samples["split"]) == {3}       # case-study studies sit in TEST, never TRAIN


def test_uti_shaped_paired_design(synthetic, trained_models, tmp_path):
    from ecoli_atlas.casestudy.uti import run_uti
    from ecoli_atlas.figures.context import build_context
    from ecoli_atlas.figures.plotting import apply_style
    from ecoli_atlas.io import read_counts, read_feature_table, read_metadata
    apply_style()
    ctx = build_context(read_counts(synthetic / "merged_count_table.txt", strip_sample_names=False),
                        read_metadata(synthetic / "metadata_revised.csv"), read_feature_table(synthetic / "feature_table.txt"),
                        trained_models, device="cpu")
    gsms, titles = uti_titles()
    res = run_uti(ctx, tmp_path, formats=("png",), gsms=gsms, titles=titles, n_perm=30)
    assert res["n_patients"] == 6 and res["n_pairs"] == 6 and res["conditions"] == {"UR": 6, "UTI": 6}
    assert {"correct", "p_sign", "p_wilcoxon", "p_perm", "toward_UTI", "toward_UR"} <= set(res)
    assert (tmp_path / "fig_uti_axis.png").exists() and (tmp_path / "uti_axis_genes.tsv").exists()
    with pytest.raises(ValueError):            # an unknown accession list
        run_uti(ctx, tmp_path, formats=("png",), gsms=("GSM999999",), titles=("X_UTI",), n_perm=5)


def test_uti_cli_needs_real_accessions(synthetic, trained_models, tmp_path):
    # the default GSM list is the real GSE128997 study, absent from the synthetic corpus
    with pytest.raises(ValueError):
        main(_args(synthetic, trained_models, tmp_path / "u", "uti"))


def test_glu_cas(synthetic, trained_models, tmp_path):
    counts = pd.read_csv(synthetic / "merged_count_table.txt", sep="\t", index_col=0).iloc[:, :12]
    runs = [f"SRR{i:03d}" for i in range(12)]
    counts.columns = [f"res/{r}/vg/{r}_sort.bam" for r in runs]      # inGEST-style BAM path columns
    counts.to_csv(tmp_path / "counts.tsv", sep="\t")
    meta = pd.DataFrame({"Run": runs, "BioProject": "PRJNA1",
                         "treatment": [f"{'GLU' if i % 2 == 0 else 'CAS'} condition. timepoint T{i // 2}" for i in range(12)]}).set_index("Run")
    meta.to_csv(tmp_path / "meta.csv")
    out = tmp_path / "gc"
    rc = main(["casestudy", "glu-cas", "--counts", str(tmp_path / "counts.tsv"), "--case-metadata", str(tmp_path / "meta.csv"),
               "--feature-table", str(synthetic / "feature_table.txt"), "--models", str(trained_models), "--out", str(out),
               "--device", "cpu", "--formats", "png"])
    assert rc == 0
    for f in ("fig_glu_cas_axis.png", "fig_glu_cas_lda_projection.png", "fig_glu_cas_axis_genes.png",
              "glu_cas_gene_correlations.csv", "glu_cas_lda_scores.csv", "glu_cas_metrics.json"):
        assert (out / f).exists(), f
    m = json.loads((out / "glu_cas_metrics.json").read_text())
    assert m["paired"]["n_pairs"] == 6 and m["lda"]["n_samples"] == 12 and m["lda"]["treatments"] == {"GLU": 6, "CAS": 6}
    corr = pd.read_csv(out / "glu_cas_gene_correlations.csv")
    assert list(corr.columns) == ["Gene_Symbol", "Spearman_Rho"] and corr["Spearman_Rho"].is_monotonic_decreasing
    scores = pd.read_csv(out / "glu_cas_lda_scores.csv")
    assert list(scores["Sample_ID"]) == runs
    # missing inputs are reported, not crashed on
    assert main(["casestudy", "glu-cas", "--models", str(trained_models), "--out", str(tmp_path / "x")]) == 2
