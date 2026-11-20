import json

import pandas as pd

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
