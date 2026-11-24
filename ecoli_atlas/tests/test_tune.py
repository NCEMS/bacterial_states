"""`train --hparams`, the out-of-fold count fit and `tune` (Optuna) on the synthetic atlas."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ecoli_atlas.cli import main


@pytest.fixture(scope="module")
def split_file(train_args, tmp_path_factory):
    out = tmp_path_factory.mktemp("split")
    assert main(["train", *train_args, "--out", str(out), "--split-only", "--test-frac", "0.2"]) == 0
    return out / "splits.csv"


def test_hparams_file_sets_the_model_and_refuses_unknown_keys(train_args, split_file, tmp_path):
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"res_dim": 5, "enc_width": 64, "block_width": 32, "epochs": 3, "block_epochs": 2, "kl_w": 0.002}))
    out = tmp_path / "b"
    assert main(["train", *train_args, "--out", str(out), "--splits", str(split_file), "--hparams", str(hp), "--device", "cpu"]) == 0
    man = json.loads((out / "manifest.json").read_text())
    cfg = man["config"]
    assert cfg["hparams"]["res_dim"] == 5 and cfg["hparams"]["enc_width"] == 64 and cfg["hparams"]["kl_w"] == 0.002
    assert cfg["epochs"] == 3 and cfg["block_epochs"] == 2 and man["embedding_dim"] == 3 + 5
    hist = pd.read_csv(out / "history.tsv", sep="\t")
    assert (hist["stage"] == "residual").sum() == 3 and (hist["stage"] == "phase").sum() == 2
    # --epochs overrides the file
    assert main(["train", *train_args, "--out", str(tmp_path / "c"), "--splits", str(split_file), "--hparams", str(hp),
                 "--epochs", "2", "--device", "cpu"]) == 0
    assert json.loads((tmp_path / "c" / "manifest.json").read_text())["config"]["epochs"] == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"res_dims": 5}))
    with pytest.raises(SystemExit, match="unknown keys res_dims"):
        main(["train", *train_args, "--out", str(tmp_path / "x"), "--splits", str(split_file), "--hparams", str(bad), "--device", "cpu"])


def test_cv_reports_the_out_of_fold_count_fit(train_args, split_file, tmp_path):
    out = tmp_path / "sweep"
    assert main(["train", *train_args, "--out", str(out), "--splits", str(split_file), "--epochs", "3", "--seeds", "1,2", "--cv",
                 "--device", "cpu"]) == 0
    c = json.loads((out / "manifest.json").read_text())["cv_counts"]
    splits = pd.read_csv(split_file, index_col=0)
    held = splits[(splits["split"] == 1) & splits["gse"].isin(splits.loc[splits["cv_fold"].notna(), "gse"])]
    assert c["n"] == len(held) == sum(f["n"] for f in c["folds"].values()) and c["nll_per_feature"] > 0
    assert c["nll_per_feature"] == pytest.approx(sum(f["n"] * f["nll_per_feature"] for f in c["folds"].values()) / c["n"])
    assert pd.read_csv(out / "seed_sweep.tsv", sep="\t")["cv_nll"].notna().all()


def test_tune_runs_resumes_and_writes_the_front(train_args, split_file, tmp_path):
    pytest.importorskip("optuna")
    from ecoli_atlas.tune import SPACE
    out = tmp_path / "study"
    args = ["tune", *train_args, "--splits", str(split_file), "--out", str(out), "--max-epochs", "2", "--device", "cpu",
            "--sampler-seed", "0"]
    assert main([*args, "--trials", "2"]) == 0
    trials = pd.read_csv(out / "trials.tsv", sep="\t")
    assert len(trials) == 2 and (trials["state"] == "COMPLETE").all()
    assert trials.loc[0, "baseline"] and trials.loc[0, "p_res_dim"] == 24 and trials.loc[0, "p_epochs"] == 150
    assert trials[["cv_nll", "cv_mean_auroc", "cv_phase_auroc"]].notna().all().all()
    pareto = pd.read_csv(out / "pareto.tsv", sep="\t")
    assert len(pareto) >= 1 and set(pareto["trial"]) == set(trials.loc[trials["pareto"], "trial"])
    best = sorted(out.glob("best/trial_*.json"))
    assert len(best) == len(pareto) and len(list(out.glob("hparams/trial_*.json"))) == 2
    hp = json.loads(best[0].read_text())
    assert set(SPACE) <= set(hp) and "block_dims" in hp
    # resumed: --max-total-trials stops at the total
    assert main([*args, "--trials", "5", "--max-total-trials", "3"]) == 0
    assert len(pd.read_csv(out / "trials.tsv", sep="\t")) == 3
    # a point of the front trains as a bundle
    assert main(["train", *train_args, "--out", str(tmp_path / "b"), "--splits", str(split_file), "--hparams", str(best[0]),
                 "--epochs", "2", "--device", "cpu"]) == 0
    # another fixed configuration in the same directory is refused
    with pytest.raises(SystemExit, match="different fixed_config"):
        main([*args, "--trials", "1", "--dna-penalty", "0"])
