import json

import numpy as np
import pandas as pd
import pytest
import torch

from ecoli_atlas.artifacts import MODEL_FILE, PREPROCESSING_FILE, ModelBundle
from ecoli_atlas.cli import main
from ecoli_atlas.config import HParams, TrainConfig
from ecoli_atlas.data import build_training, load_tables
from ecoli_atlas.embed import embed_counts
from ecoli_atlas.features import FeatureSet, read_feature_table
from ecoli_atlas.io import read_counts
from ecoli_atlas.labels import is_reference, head_labels
from ecoli_atlas.losses import kl_normal, nb_nll
from ecoli_atlas.splits import make_splits
from ecoli_atlas.train import encode, fit, study_blocked_batches


def test_nb_nll_matches_scipy_and_is_lowest_at_the_true_mean():
    from scipy.stats import nbinom
    y = torch.tensor([0.0, 3.0, 250.0, 2.0e7])
    mean, theta = torch.tensor([2.0, 3.0, 200.0, 1.9e7]), torch.tensor(4.0)
    ours = nb_nll(y, torch.log(mean), torch.log(theta)).numpy()
    ref = -nbinom.logpmf(y.numpy(), theta.item(), theta.item() / (theta.item() + mean.numpy()))
    assert np.allclose(ours, ref, rtol=1e-6)
    grid = torch.log(torch.tensor([50.0, 100.0, 200.0, 400.0]))
    assert int(torch.argmin(nb_nll(torch.tensor(200.0), grid, torch.log(theta)))) == 2
    assert torch.allclose(kl_normal(torch.zeros(2, 3), torch.zeros(2, 3)), torch.zeros(2))


def test_batches_hold_whole_studies_and_cap_large_ones():
    gse = np.array([0] * 30 + [1] * 5 + [2] * 5 + [3] * 5)
    batches = study_blocked_batches(np.arange(45), gse, batch_size=12, seed=0, cap=10)
    rows = np.concatenate(batches)
    assert len(rows) == 25 and (gse[rows] == 0).sum() == 10 and len(set(rows)) == 25
    for b in batches:                                   # a study is never divided between batches
        for g in set(gse[b]):
            assert (gse[b] == g).sum() == (gse[rows] == g).sum()
    again = np.concatenate(study_blocked_batches(np.arange(45), gse, 12, seed=1, cap=10))
    assert set(again[gse[again] == 0]) != set(rows[gse[rows] == 0])      # a fresh draw of the large study


def test_training_lowers_the_loss_and_uses_training_statistics_only(atlas_files):
    tables = load_tables(atlas_files["counts"], atlas_files["metadata"], atlas_files["qc"])
    fs = FeatureSet.from_table(read_feature_table(atlas_files["features"]))
    cfg = TrainConfig(seed=5, test_frac=0.2, n_candidates=50, epochs=30, batch=40, hparams=HParams(enc_width=64, dec_width=64, block_width=32, res_dim=8))
    splits = make_splits(tables.meta["GSE"], head_labels(tables.meta), is_reference(tables.meta), cfg)
    data = build_training(tables, fs, cfg, splits)
    assert data.X.shape == (119, fs.n) and data.C.shape[0] == 119 and not torch.isnan(data.C).any()
    ref_tr = data.tr_rows[data.reference[data.tr_rows]]
    assert abs(float(data.X[ref_tr].mean())) < 0.05                       # z-scored on the reference samples of the training pool
    model, hist = fit(data, cfg, "cpu")
    stage = lambda name: [h for h in hist if h["stage"] == name]                                    # noqa: E731
    assert len(stage("phase")) == cfg.block_epochs and stage("phase")[-1]["ce"] < stage("phase")[0]["ce"]
    assert len(stage("residual")) == 30 and stage("residual")[-1]["counts"] < stage("residual")[0]["counts"]
    # a phase label inferred from an OD or a time is fitted on but not scored
    inferred = (tables.meta["growth_phase_basis"] == "inferred").to_numpy()
    assert (data.y["phase"][inferred] >= 0).any() and (data.y_eval["phase"][inferred] == -1).all()
    assert np.array_equal(data.y["phase"][~inferred], data.y_eval["phase"][~inferred]) and np.array_equal(data.y["medium"], data.y_eval["medium"])
    # holding out a fold leaves none of its studies' samples in the fit, reference or not
    for k in (1, 2, 3):
        held = set(data.splits["gse"].to_numpy()[data.val_rows(k)])
        fitted = set(data.splits["gse"].to_numpy()[data.fit_rows(k)])
        assert held and fitted and not held & fitted
    assert sum(len(data.fit_rows(k)) for k in (1, 2, 3)) < 3 * len(data.tr_rows)
    # the embedding is [phase | medium | lab strain | residual]; the blocks are standardised on the training pool
    z = encode(model, data.X, data.C, "cpu")["z"]
    assert z.shape[1] == 6 + 12 + 4 + 8 and np.allclose(z[data.tr_rows, :22].mean(0), 0, atol=1e-3)
    # the residual stage leaves the block encoders as they were
    before = {k: v.clone() for k, v in model.blocks.state_dict().items()}
    blocks_only, _ = fit(data, cfg, "cpu", epochs=0)                       # same seed, no residual stage
    assert all(torch.equal(before[k], v) for k, v in blocks_only.blocks.state_dict().items())
    # changing a TEST sample's counts changes no training statistic
    tables.counts.iloc[:, data.te_rows[0]] *= 7
    again = build_training(tables, fs, cfg, splits)
    assert np.array_equal(np.nan_to_num(again.ref_log), np.nan_to_num(data.ref_log)) and np.array_equal(again.zscore.mu, data.zscore.mu)
    assert np.array_equal(again.scaler.mu, data.scaler.mu)


def test_bundle_files_and_evaluation(trained):
    for name in (MODEL_FILE, PREPROCESSING_FILE):
        obj = torch.load(trained / name, weights_only=True)           # tensors only
        assert all(isinstance(v, torch.Tensor) for v in obj.values())
    manifest = json.loads((trained / "manifest.json").read_text())
    assert manifest["schema_version"] == 2 and manifest["heads"] == {"phase": ["exponential", "stationary"], "medium": ["rich", "minimal"], "lab_strain": ["non_lab", "lab"]}
    assert [b["block"] for b in manifest["embedding_blocks"]] == ["phase", "medium", "lab_strain", "residual"] and manifest["embedding_blocks"][-1]["end"] == 46
    n = manifest["n_samples"]
    assert n["training_pool"] + n["test"] + n["projected_only"] == 119 and 0 < n["training_reference"] < n["training_pool"]
    report = json.loads((trained / "evaluation.json").read_text())
    assert set(report["heads"]["test"]) <= {"phase", "medium", "lab_strain"} and "medium" in report["heads"]["test"] and report["heads"]["training_pool"]["medium"]["auroc"] > 0.9
    assert "late_stationary" not in report["phase_check"] and "transition" in report["phase_check"]
    assert set(report["medium_check"]) == {"minimal_plus_hydrolysate"} and report["medium_check"]["minimal_plus_hydrolysate"]["n_studies"] == 1
    pred = pd.read_csv(trained / "predictions.tsv", sep="\t", index_col=0)
    emb = pd.read_csv(trained / "embedding.tsv", sep="\t", index_col=0)
    assert len(pred) == len(emb) == 119 and emb.shape[1] == 46 and pred["p_stationary"].between(0, 1).all()
    splits = pd.read_csv(trained / "splits.csv", index_col=0)
    assert not set(splits.loc[splits["split"] == 1, "gse"]) & set(splits.loc[splits["split"] == 3, "gse"])
    calls = pd.read_csv(trained / "missing_genes.tsv", sep="\t")              # one row per sample and gene called missing
    assert list(calls.columns) == ["sample", "feature"] and calls.groupby("sample").size().reindex(pred.index, fill_value=0).equals(pred["n_missing_genes"])


def test_embed_reproduces_the_training_embedding_and_works_without_qc(trained, atlas_files, tmp_path):
    bundle = ModelBundle.load(trained)
    counts = read_counts(atlas_files["counts"])
    qc = pd.read_csv(atlas_files["qc"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM")
    res = embed_counts(bundle, counts[["GSM0000", "GSM0105"]], qc)
    ref = pd.read_csv(trained / "embedding.tsv", sep="\t", index_col=0).loc[["GSM0000", "GSM0105"]]
    assert np.allclose(res["embedding"].to_numpy(), ref.to_numpy(), atol=2e-4)
    assert res["coverage"].fraction == 1.0 and list(res["predictions"].columns) == ["p_stationary", "phase", "p_minimal", "medium", "p_lab", "lab_strain"]
    assert main(["embed", "--counts", str(atlas_files["counts"]), "--models", str(trained), "--out", str(tmp_path / "e"), "--device", "cpu"]) == 0
    noqc = pd.read_csv(tmp_path / "e" / "samples.tsv", sep="\t", index_col=0)
    assert len(noqc) == 120 and (noqc["has_qc"] == 0).all() and (noqc["size_factor"] > 0).all()
    few = counts.iloc[:20]
    with pytest.raises(ValueError, match="model features"):
        embed_counts(bundle, few)


def test_evaluate_command_repeats_the_training_report(trained, train_args, tmp_path):
    assert main(["evaluate", *train_args, "--models", str(trained), "--out", str(tmp_path / "ev"), "--device", "cpu"]) == 0
    a = json.loads((trained / "evaluation.json").read_text())["heads"]["test"]
    b = json.loads((tmp_path / "ev" / "evaluation.json").read_text())["heads"]["test"]
    assert a["medium"]["auroc"] == pytest.approx(b["medium"]["auroc"], abs=1e-6)


def test_seed_sweep_with_cv_copies_one_seed(train_args, tmp_path):
    out = tmp_path / "sweep"
    assert main(["train", *train_args, "--out", str(out), "--epochs", "4", "--seeds", "1,2", "--cv", "--device", "cpu", "--test-frac", "0.2"]) == 0
    sweep = pd.read_csv(out / "seed_sweep.tsv", sep="\t")
    assert set(sweep["seed"]) == {1, 2} and sweep.groupby("seed")["selected"].first().sum() == 1 and sweep["cv_auroc"].notna().all()
    chosen = int(sweep.loc[sweep["selected"], "seed"].iloc[0])
    assert json.loads((out / "manifest.json").read_text())["seed"] == chosen and (out / f"seed_{chosen}" / "model.pt").exists()
    assert (out / "missing_genes.tsv").exists() and not list(out.glob("seed_*/missing_genes.tsv"))       # written once for the sweep
    # the out-of-fold scores behind the cv columns: every sample of the fold studies, scored by the model fitted without its study
    from sklearn.metrics import roc_auc_score
    from ecoli_atlas.metrics import within_study_order
    splits = pd.read_csv(out / "splits.csv", index_col=0)
    for s in (1, 2):
        cv = pd.read_csv(out / f"seed_{s}" / "cv_predictions.tsv", sep="\t", index_col=0)
        pool = splits[splits["gse"].isin(splits.loc[splits["cv_fold"].notna(), "gse"]) & (splits["split"] == 1)]
        assert set(cv.index) == set(pool.index) and cv[["p_stationary", "p_minimal", "p_lab", "phase_logit"]].notna().all().all()
        assert (cv.groupby("gse")["cv_fold"].nunique() == 1).all()
        ev = json.loads((out / f"seed_{s}" / "evaluation.json").read_text())
        r = cv[cv["reference"] & cv["medium_scored"]]
        assert roc_auc_score(r["medium_label"] == "minimal", r["p_minimal"]) == pytest.approx(ev["cv_metrics"]["medium"]["auroc"], abs=1e-4)
        order = within_study_order(cv["phase_logit"].to_numpy(), cv["phase_level"].to_numpy(), pd.factorize(cv["gse"])[0], np.arange(len(cv)))
        assert order["share_in_order"] == pytest.approx(ev["cv_phase_order"]["share_in_order"], abs=1e-3)
    assert (out / "cv_predictions.tsv").exists()


@pytest.mark.parametrize("fit_name", ["one_pass", "unstructured"])
def test_the_one_loss_fits_train_save_and_embed(fit_name, atlas_files, tmp_path):
    tables = load_tables(atlas_files["counts"], atlas_files["metadata"], atlas_files["qc"])
    fs = FeatureSet.from_table(read_feature_table(atlas_files["features"]))
    hp = HParams(enc_width=64, dec_width=64, block_width=32, res_dim=8)
    cfg = TrainConfig(seed=5, fit=fit_name, test_frac=0.2, n_candidates=50, epochs=30, batch=40, hparams=hp)
    splits = make_splits(tables.meta["GSE"], head_labels(tables.meta), is_reference(tables.meta), cfg)
    data = build_training(tables, fs, cfg, splits)
    model, hist = fit(data, cfg, "cpu")
    assert {h["stage"] for h in hist} == {fit_name} and len(hist) == 30
    for term in ("counts", "phase", "medium", "lab_strain"):
        assert hist[-1][term] < hist[0][term]
    z = encode(model, data.X, data.C, "cpu")["z"]
    assert z.shape[1] == hp.z_dim == 30
    if fit_name == "one_pass":            # the blocks moved with the rest: not those of a two-stage fit from the same seed
        two, _ = fit(data, TrainConfig(seed=5, test_frac=0.2, n_candidates=50, epochs=0, batch=40, hparams=hp), "cpu")
        assert not all(torch.equal(v, two.blocks.state_dict()[k]) for k, v in model.blocks.state_dict().items())
        assert float(model.block_mu.abs().sum()) == 0.0 and float((model.block_sd - 1).abs().sum()) == 0.0
    from ecoli_atlas.artifacts import save_bundle
    manifest = save_bundle(tmp_path / "b", model, data, cfg, "cpu", 5)
    assert manifest["fit"] == fit_name
    assert [b["block"] for b in manifest["embedding_blocks"]] == (["z"] if fit_name == "unstructured" else ["phase", "medium", "lab_strain", "residual"])
    bundle = ModelBundle.load(tmp_path / "b")
    counts = read_counts(atlas_files["counts"])
    qc = pd.read_csv(atlas_files["qc"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM")
    res = embed_counts(bundle, counts[data.ids[:5]], qc)
    assert np.allclose(res["embedding"].to_numpy(), z[:5], atol=2e-4)


def test_an_unknown_fit_is_refused():
    with pytest.raises(ValueError, match="fit must be one of"):
        TrainConfig(fit="joint")


def test_ordinal_head_and_loss():
    from ecoli_atlas.models import OrdinalHead, call_logits
    from ecoli_atlas.train import weighted_ordinal_loss
    head = OrdinalHead(3)
    cuts = head.cuts()
    assert cuts.shape == (2,) and float(cuts[0].detach()) == 0.0 and float(cuts[1].detach()) > 0.0
    cum = head(torch.randn(7, 3))
    assert cum.shape == (7, 2) and bool((cum[:, 0] > cum[:, 1]).all())           # P(level > 0) > P(level > 1)
    two = call_logits(head, cum)
    assert torch.allclose(torch.softmax(two, 1)[:, 1], torch.sigmoid(cum[:, 1]))    # the call is P(last level)
    lab, w, cw = torch.tensor([0, 1, 2, -1]), torch.ones(4), torch.ones(3)
    right = torch.tensor([[-9.0, -9.0], [9.0, -9.0], [9.0, 9.0], [0.0, 0.0]])
    wrong = right.flip(0)
    assert float(weighted_ordinal_loss(right, lab, w, cw)) < 1e-3 < 1.0 < float(weighted_ordinal_loss(wrong, lab, w, cw))
    assert float(weighted_ordinal_loss(right, torch.full((4,), -1), w, cw)) == 0.0


def test_within_study_order():
    from ecoli_atlas.metrics import within_study_order
    level = np.array([0, 1, 2, 0, 2, -1, 0, 0])
    gse = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    score = np.array([0.1, 0.5, 0.9, 0.8, 0.2, 5.0, 0.0, 1.0])
    out = within_study_order(score, level, gse, np.arange(8))
    assert out["n_studies"] == 2 and out["n_pairs"] == 4                 # study 2 has one level only
    assert out["share_in_order_mean_over_studies"] == pytest.approx(0.5) and out["share_in_order"] == pytest.approx(0.75)


@pytest.mark.parametrize("fit_name", ["two_stage", "unstructured"])
def test_the_ordinal_phase_head_trains_saves_and_embeds(fit_name, atlas_files, tmp_path):
    tables = load_tables(atlas_files["counts"], atlas_files["metadata"], atlas_files["qc"])
    fs = FeatureSet.from_table(read_feature_table(atlas_files["features"]))
    hp = HParams(enc_width=64, dec_width=64, block_width=32, res_dim=8)
    for seed in range(1, 30):             # the one study with transition samples must be in the training pool
        cfg = TrainConfig(seed=seed, fit=fit_name, phase_head="ordinal", test_frac=0.2, n_candidates=50, epochs=20, batch=40, hparams=hp)
        splits = make_splits(tables.meta["GSE"], head_labels(tables.meta), is_reference(tables.meta), cfg)
        if (splits.loc[splits["gse"] == "GSE06", "split"] == 1).all():
            break
    data = build_training(tables, fs, cfg, splits)
    assert data.fit_classes["phase"] == ("exponential", "transition", "stationary") and data.fit_classes["medium"] == ("rich", "minimal")
    in6 = (tables.meta["GSE"] == "GSE06").to_numpy()
    assert set(data.y_fit["phase"][in6]) == {1, 2} and (data.y["phase"][in6] == -1).all()     # transition and late stationary: levels, no class
    assert np.array_equal(data.y_fit["medium"], data.y["medium"])
    model, hist = fit(data, cfg, "cpu")
    phase_rows = [h for h in hist if h["stage"] == "phase"] if fit_name == "two_stage" else hist
    key = "ce" if fit_name == "two_stage" else "phase"
    assert phase_rows[-1][key] < phase_rows[0][key]
    enc = encode(model, data.X, data.C, "cpu")
    assert ((enc["probs"]["phase"] >= 0) & (enc["probs"]["phase"] <= 1)).all()
    assert np.allclose(1 / (1 + np.exp(-enc["scores"]["phase"])), enc["probs"]["phase"], atol=1e-5)
    from ecoli_atlas.artifacts import save_bundle
    from ecoli_atlas.evaluate import evaluate
    report = evaluate(model, data, cfg, "cpu")["report"]
    assert report["phase_order"]["training_pool"]["n_studies"] >= 1
    manifest = save_bundle(tmp_path / "b", model, data, cfg, "cpu", cfg.seed)
    assert manifest["phase_head"] == "ordinal" and set(manifest["label_caps"]["phase"]) == {"exponential", "transition", "stationary"}
    assert set(manifest["n_labelled_training"]["phase"]) == {"exponential", "transition", "stationary"}
    bundle = ModelBundle.load(tmp_path / "b")
    res = embed_counts(bundle, read_counts(atlas_files["counts"])[data.ids[:5]],
                       pd.read_csv(atlas_files["qc"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM"))
    assert np.allclose(res["embedding"].to_numpy(), enc["z"][:5], atol=2e-4)
    assert np.allclose(res["predictions"]["p_stationary"].to_numpy(), enc["probs"]["phase"][:5], atol=2e-4)


def test_an_unknown_phase_head_is_refused():
    with pytest.raises(ValueError, match="phase_head must be one of"):
        TrainConfig(phase_head="three_class")


def test_drivers_attributions_add_up_and_find_the_phase_genes(trained, train_args, atlas_files, tmp_path):
    from ecoli_atlas.drivers import combine_seeds, default_contrasts, expression_units, spearman_columns, unit_name
    assert default_contrasts(("a", "b")) == [("a", "b")] and default_contrasts(("a", "b", "c")) == [("a", "c"), ("a", "b"), ("b", "c")]
    rng = np.random.RandomState(0)
    v = rng.rand(30)
    rho = spearman_columns(np.column_stack([v ** 3, -v, np.ones(30)]), v)
    assert rho == pytest.approx([1.0, -1.0, 0.0])
    assert unit_name(["pspA", "pspB", "pspC"]) == "pspABC" and unit_name(["iscR", "hscA", "fdx"]) == "iscR-fdx (3)" and unit_name(["dps"]) == "dps"
    # units: neighbours on one strand that move together within studies; a study effect alone does not join genes
    feat = pd.DataFrame({"Chr": "c", "Start": [1, 200, 400, 600, 5000], "End": [150, 350, 550, 750, 5200], "Strand": ["+", "+", "+", "-", "+"]},
                        index=list("abcde"))
    gse = np.repeat(np.arange(5), 8)
    common, study = rng.randn(40), np.repeat(rng.randn(5) * 5, 8)
    logx = np.column_stack([common + 0.1 * rng.randn(40), common + 0.1 * rng.randn(40), study + rng.randn(40), common, common])
    assert expression_units(feat, list("abcde"), logx, gse).tolist() == [0, 0, 2, 3, 4]
    a = pd.DataFrame({"k": list("xyz"), "rank": [1, 2, 3], "attribution": [3.0, 1.0, -1.0]})
    b = pd.DataFrame({"k": list("zyx"), "rank": [1, 2, 3], "attribution": [2.0, 1.0, 1.0]})
    both = combine_seeds([a, b], "k", top=1).set_index("k")
    assert both.loc["x", "attribution"] == 2.0 and both.loc["z", "seeds_same_sign"] == 1 and both["seeds_in_top"].tolist() == [1, 0, 1]
    out = tmp_path / "drivers"
    assert main(["drivers", *train_args, "--models", str(trained), "--out", str(out), "--device", "cpu", "--formats", "png", "--top", "5"]) == 0
    table = pd.read_csv(out / "drivers_phase_exponential_to_stationary.tsv", sep="\t")
    summary = json.loads((out / "drivers_phase.json").read_text())["contrasts"][0]
    assert summary["sum_of_attributions"] == pytest.approx(summary["score_gap_high_minus_baseline"], rel=0.05)   # completeness
    assert table["share"].sum() == pytest.approx(1.0, abs=1e-3) and table["rank"].tolist() == list(range(1, len(table) + 1))
    feat = pd.read_csv(atlas_files["features"], sep="\t", keep_default_na=False)
    phase_genes = set(feat["Feature_ID"][:10])                        # the fixture raises the first ten genes in stationary phase
    top = table.head(10)
    hit = top["feature"].isin(phase_genes)
    assert hit.sum() >= 7 and (top.loc[hit, "attribution"] > 0).all() and (top.loc[hit, "direction"] == "up").all()
    units = pd.read_csv(out / "units_phase_exponential_to_stationary.tsv", sep="\t")
    assert units["n_genes"].sum() == len(table) and units["attribution"].sum() == pytest.approx(table["attribution"].sum(), rel=1e-3)
    assert (out / "fig_drivers_phase_exponential_to_stationary.png").stat().st_size > 5000
    assert (out / "fig_units_phase_exponential_to_stationary.png").stat().st_size > 5000
    assert main(["drivers", *train_args, "--models", str(trained), "--out", str(out), "--device", "cpu", "--contrast", "rich:stationary"]) == 2


def test_drivers_average_over_the_seeds_of_a_sweep(train_args, tmp_path):
    sweep, out = tmp_path / "sweep", tmp_path / "drivers"
    assert main(["train", *train_args, "--out", str(sweep), "--epochs", "4", "--seeds", "1,2,3", "--device", "cpu", "--test-frac", "0.2"]) == 0
    assert main(["drivers", *train_args, "--models", str(sweep), "--out", str(out), "--device", "cpu", "--formats", "png", "--top", "5"]) == 0
    table = pd.read_csv(out / "drivers_phase_exponential_to_stationary.tsv", sep="\t")
    assert (table["n_seeds"] == 3).all() and table["seeds_same_sign"].between(0, 3).all() and table["seeds_in_top"].max() == 3
    summary = json.loads((out / "drivers_phase.json").read_text())
    assert len(summary["models"]) == 3 and -1 <= summary["contrasts"][0]["rank_correlation_between_seeds"] <= 1
    assert main(["drivers", *train_args, "--models", str(sweep), "--out", str(out / "one"), "--device", "cpu", "--formats", "png", "--one-model"]) == 0
    assert "n_seeds" not in pd.read_csv(out / "one" / "drivers_phase_exponential_to_stationary.tsv", sep="\t").columns


def test_genes_missing_from_a_strain_are_called_masked_and_not_fitted(atlas_files, tmp_path):
    tables = load_tables(atlas_files["counts"], atlas_files["metadata"], atlas_files["qc"])
    fs = FeatureSet.from_table(read_feature_table(atlas_files["features"]))
    island = list(range(40, 56))
    upec = (tables.meta["strain_group"] == "UPEC").to_numpy()
    tables.counts.iloc[island, np.flatnonzero(upec)] = 0              # the UPEC studies lack a 16-gene island
    cfg = TrainConfig(seed=5, test_frac=0.2, n_candidates=50, epochs=3, block_epochs=3, batch=40,
                      hparams=HParams(enc_width=32, dec_width=32, block_width=16, res_dim=4))
    splits = make_splits(tables.meta["GSE"], head_labels(tables.meta), is_reference(tables.meta), cfg)
    data = build_training(tables, fs, cfg, splits)
    M = data.M.numpy()
    assert M[upec][:, island].all() and M[~upec].sum() <= 0.001 * M[~upec].size and M[upec].sum() <= upec.sum() * (len(island) + 1)
    assert (data.X.numpy()[M] == 0).all() and (data.low[island] > 0).all()
    # the island's statistics come from the samples that have it
    plain = build_training(tables, fs, TrainConfig(**{**cfg.to_dict(), "hparams": cfg.hparams, "mask_missing": False}), splits)
    assert plain.low is None and not plain.M.any() and (data.zscore.mu[island] > plain.zscore.mu[island]).all()
    assert (plain.X.numpy()[upec][:, island] < -1).all()
    # training samples are also seen with the gaps of other samples; the counts in the gaps do not enter the loss
    from ecoli_atlas.train import model_input
    rt = torch.as_tensor(np.flatnonzero(~upec)[:30])
    x = model_input(data, rt, data.tr_rows, cfg, np.random.RandomState(0))
    gaps = (x[:, island] == 0).all(dim=1).numpy()
    assert 0 < gaps.sum() < len(rt) and torch.equal(x[~torch.as_tensor(gaps)], data.X[rt][~torch.as_tensor(gaps)])
    model, hist = fit(data, cfg, "cpu")
    changed = data.Y.clone()
    changed[data.M] = 1e6
    from ecoli_atlas.evaluate import count_fit
    before = count_fit(model, data, data.tr_rows, "cpu")
    data.Y = changed
    assert count_fit(model, data, data.tr_rows, "cpu") == before
    # a saved bundle makes the same calls for new samples
    from ecoli_atlas.artifacts import save_bundle
    save_bundle(tmp_path / "b", model, data, cfg, "cpu", 5)
    bundle = ModelBundle.load(tmp_path / "b")
    ids = [i for i, u in zip(data.ids, upec) if u][:3] + [i for i, u in zip(data.ids, upec) if not u][:3]
    res = embed_counts(bundle, tables.counts[ids])
    assert list(res["samples"]["n_missing_genes"]) == [int(M[data.ids.index(i)].sum()) for i in ids]
    assert set(res["missing"]["sample"]) == set(ids[:3]) and set(res["missing"]["feature"]) >= {fs.ids[g] for g in island}


def test_antisense_table_vetoes_calls_in_training_and_embed(atlas_files, tmp_path):
    counts = read_counts(atlas_files["counts"])
    meta = pd.read_csv(atlas_files["metadata"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM")
    upec = list(meta.index[meta["strain_group"] == "UPEC"])
    silent = list(meta.index[meta["GSE"] == "GSE00"])
    counts.loc[counts.index[40:56], upec] = 0                        # missing from the UPEC studies
    counts.loc[counts.index[60:70], silent] = 0                      # silent in one K-12 study
    anti = pd.DataFrame(np.random.RandomState(1).poisson(12, size=counts.shape), index=counts.index, columns=counts.columns)
    anti.loc[anti.index[40:56], upec] = 0
    counts.to_csv(tmp_path / "counts.tsv", sep="\t")
    anti.to_csv(tmp_path / "antisense.tsv", sep="\t")
    args = ["--counts", str(tmp_path / "counts.tsv"), "--metadata", str(atlas_files["metadata"]), "--qc", str(atlas_files["qc"]),
            "--features", str(atlas_files["features"]), "--epochs", "2", "--device", "cpu", "--test-frac", "0.2"]
    assert main(["train", *args, "--out", str(tmp_path / "plain")]) == 0
    assert main(["train", *args, "--out", str(tmp_path / "veto"), "--antisense", str(tmp_path / "antisense.tsv")]) == 0
    plain = pd.read_csv(tmp_path / "plain" / "predictions.tsv", sep="\t", index_col=0)["n_missing_genes"]
    veto = pd.read_csv(tmp_path / "veto" / "predictions.tsv", sep="\t", index_col=0)["n_missing_genes"]
    in_model = [i for i in silent if i in plain.index]
    assert (plain[in_model] >= 10).all() and (veto[in_model] <= 1).all()            # the silent genes are present
    assert (veto[[i for i in upec if i in veto.index]] >= 16).all()                  # the deleted ones stay called
    assert json.loads((tmp_path / "veto" / "manifest.json").read_text())["missing_genes"]["antisense_veto"] is True
    assert json.loads((tmp_path / "plain" / "manifest.json").read_text())["missing_genes"]["antisense_veto"] is False
    # embed: with the antisense table the bundle's own calls come back, without it the silent genes are called
    ids = in_model[:2] + upec[:2]
    counts[ids].to_csv(tmp_path / "q.tsv", sep="\t")
    assert main(["embed", "--counts", str(tmp_path / "q.tsv"), "--models", str(tmp_path / "veto"), "--out", str(tmp_path / "e1"),
                 "--antisense", str(tmp_path / "antisense.tsv"), "--device", "cpu"]) == 0
    assert main(["embed", "--counts", str(tmp_path / "q.tsv"), "--models", str(tmp_path / "veto"), "--out", str(tmp_path / "e2"), "--device", "cpu"]) == 0
    n1 = pd.read_csv(tmp_path / "e1" / "samples.tsv", sep="\t", index_col=0)["n_missing_genes"]
    n2 = pd.read_csv(tmp_path / "e2" / "samples.tsv", sep="\t", index_col=0)["n_missing_genes"]
    assert list(n1) == list(veto[ids]) and (n2[ids[:2]] >= 10).all() and list(n2[ids[2:]]) == list(n1[ids[2:]])


def test_pooled_calls_in_training_and_embed(atlas_files, tmp_path):
    counts = read_counts(atlas_files["counts"])
    meta = pd.read_csv(atlas_files["metadata"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM")
    upec = list(meta.index[meta["strain_group"] == "UPEC"])
    rng = np.random.RandomState(2)
    weak = counts.index[60:80]
    counts.loc[weak] = rng.poisson(0.3, size=(len(weak), counts.shape[1]))         # weakly expressed genes
    counts.loc[counts.index[30:40], upec] = 0                                     # the UPEC studies lack a well expressed island ...
    counts.loc[counts.index[62:78], upec] = 0                                     # ... and one of weakly expressed genes
    counts.to_csv(tmp_path / "counts.tsv", sep="\t")
    args = ["--counts", str(tmp_path / "counts.tsv"), "--metadata", str(atlas_files["metadata"]), "--qc", str(atlas_files["qc"]),
            "--features", str(atlas_files["features"]), "--epochs", "2", "--device", "cpu", "--test-frac", "0.2"]
    assert main(["train", *args, "--out", str(tmp_path / "pool")]) == 0
    assert main(["train", *args, "--out", str(tmp_path / "single"), "--no-pool"]) == 0
    pool = pd.read_csv(tmp_path / "pool" / "predictions.tsv", sep="\t", index_col=0)["n_missing_genes"]
    single = pd.read_csv(tmp_path / "single" / "predictions.tsv", sep="\t", index_col=0)["n_missing_genes"]
    u = [i for i in upec if i in pool.index]
    assert (single[u] >= 10).all() and (single[u] < 20).all() and (pool[u] >= 24).all()      # the weak island needs the summed counts
    others = [i for i in pool.index if i not in upec]
    assert pool[others].median() == 0
    saved = pd.read_csv(tmp_path / "pool" / "missing_genes.tsv", sep="\t")
    assert saved.groupby("sample").size().reindex(pool.index, fill_value=0).equals(pool) and set(saved["sample"]) >= set(u)
    man = json.loads((tmp_path / "pool" / "manifest.json").read_text())
    assert man["missing_genes"]["pooled"] is True and json.loads((tmp_path / "single" / "manifest.json").read_text())["missing_genes"]["pooled"] is False
    assert "missing_low_pool" in torch.load(tmp_path / "pool" / PREPROCESSING_FILE, weights_only=True)
    # embed: the samples of one study given together get the training calls; --no-pool and a single sample call alone
    ids = [i for i in u if meta.loc[i, "GSE"] == meta.loc[u[0], "GSE"]]
    counts[ids].to_csv(tmp_path / "q.tsv", sep="\t")
    base = ["embed", "--counts", str(tmp_path / "q.tsv"), "--models", str(tmp_path / "pool"), "--device", "cpu"]
    assert main(base + ["--out", str(tmp_path / "e1")]) == 0 and main(base + ["--out", str(tmp_path / "e2"), "--no-pool"]) == 0
    n1 = pd.read_csv(tmp_path / "e1" / "samples.tsv", sep="\t", index_col=0)["n_missing_genes"]
    n2 = pd.read_csv(tmp_path / "e2" / "samples.tsv", sep="\t", index_col=0)["n_missing_genes"]
    assert list(n1) == list(pool[ids]) and (n2 <= n1).all() and (n2 < n1).any()
    # a table of studies splits the pooling
    pd.DataFrame({"sample": ids, "GSE": [f"own{i}" for i in range(len(ids))]}).to_csv(tmp_path / "studies.tsv", sep="\t", index=False)
    assert main(base + ["--out", str(tmp_path / "e3"), "--studies", str(tmp_path / "studies.tsv")]) == 0
    assert list(pd.read_csv(tmp_path / "e3" / "samples.tsv", sep="\t", index_col=0)["n_missing_genes"]) == list(n2)


def _with_dna(atlas_files, tmp_path, rho=0.06):
    """The synthetic atlas with genomic DNA in two studies: counts with DNA reads added, and a QC table whose DNA estimate says so."""
    from ecoli_atlas.data import dna_reads
    counts = read_counts(atlas_files["counts"])
    feat = read_feature_table(atlas_files["features"])
    qc = pd.read_csv(atlas_files["qc"], sep="\t", dtype=str, keep_default_na=False).set_index("GSM")
    genome = int(feat["Length"].sum())
    qc["full_reads_kept"], qc["full_pct_aligned"], qc["full_dna_fraction_est"] = "1000000", "100", "0"
    dirty = list(qc.index[qc["GSE"].isin(["GSE04", "GSE05"])])
    paired = (qc.loc[dirty, "full_layout"] == "PE").to_numpy()
    fragments = np.where(paired, 500000.0, 1000000.0)
    qc.loc[dirty, "full_dna_fraction_est"] = [f"{x:.6g}" for x in rho * 2 * genome / fragments]
    length = np.where(paired, 200.0, 75.0)                       # fragment_q50 / length_q50_r1 of the fixture
    D = dna_reads(np.full(len(dirty), rho), length, feat.loc[counts.index, "Length"].to_numpy())
    noisy = counts.copy()
    cols = [c for c in dirty if c in counts.columns]
    noisy[cols] = counts[cols] + np.random.RandomState(7).poisson(D[[dirty.index(c) for c in cols]]).T
    noisy.to_csv(tmp_path / "counts_dna.tsv", sep="\t")
    qc.to_csv(tmp_path / "qc_dna.tsv", sep="\t")
    return counts, noisy, tmp_path / "counts_dna.tsv", tmp_path / "qc_dna.tsv", cols


def test_dna_density_from_the_qc_table(atlas_files, tmp_path):
    from ecoli_atlas.data import dna_density, dna_reads, read_table
    _, _, _, qc_path, dirty = _with_dna(atlas_files, tmp_path)
    qc = read_table(qc_path)
    genome = 100000
    rho, length = dna_density(qc, ["GSM0400", "GSM0500", "GSM0000", "not_there"], genome)
    f = float(qc.loc["GSM0400", "full_dna_fraction_est"])
    assert np.isclose(rho[0], f * 500000 / (2 * genome)) and length[0] == 200          # paired-end: fragments = reads / 2, fragment size
    assert np.isclose(rho[1], float(qc.loc["GSM0500", "full_dna_fraction_est"]) * 1000000 / (2 * genome)) and length[1] == 75
    assert rho[2] == 0 and rho[3] == 0                                                  # estimate 0, and no QC row
    assert dna_density(None, ["a"], genome)[0][0] == 0 and dna_density(qc, ["GSM0400"], None)[0][0] == 0
    assert np.allclose(dna_reads(np.array([0.5]), np.array([100.0]), np.array([900, 1900])), [[500.0, 1000.0]])


def test_dna_term_in_training_and_embed(atlas_files, tmp_path):
    clean, noisy, counts_path, qc_path, dirty = _with_dna(atlas_files, tmp_path)
    fs = FeatureSet.from_table(read_feature_table(atlas_files["features"]))
    hp = HParams(enc_width=32, dec_width=32, block_width=16, res_dim=4)
    assert not TrainConfig().dna_decoder and not TrainConfig().dna_inputs and TrainConfig().max_dna_fraction == 0.10      # off unless asked for
    cfg = TrainConfig(seed=5, test_frac=0.2, n_candidates=50, epochs=40, block_epochs=3, batch=40, hparams=hp, dna_decoder=True, dna_inputs=True)
    off = TrainConfig(**{**cfg.to_dict(), "hparams": hp, "dna_decoder": False, "dna_inputs": False})
    t_dna = load_tables(counts_path, atlas_files["metadata"], qc_path)
    t_clean = load_tables(atlas_files["counts"], atlas_files["metadata"], qc_path)
    splits = make_splits(t_dna.meta["GSE"], head_labels(t_dna.meta), is_reference(t_dna.meta), cfg)
    truth = build_training(t_clean, fs, off, splits)             # the same libraries without DNA
    corrected, raw = build_training(t_dna, fs, cfg, splits), build_training(t_dna, fs, off, splits)
    r = [corrected.ids.index(i) for i in dirty]
    err = lambda d: float((d.X[r] - truth.X[r]).abs().mean())                           # noqa: E731
    assert err(corrected) < 0.4 * err(raw)                       # the inputs with the DNA taken off are close to the DNA-free ones
    others = [i for i in range(len(corrected.ids)) if i not in r]
    assert float((corrected.X[others] - truth.X[others]).abs().mean()) < 0.05
    assert corrected.dna_rho is not None and int((corrected.dna_rho > 0).sum()) == len(dirty) and raw.dna_rho is None
    # the decoder: the learned factor stays near 1, and the counts of the DNA samples are fitted better with the term
    from ecoli_atlas.evaluate import count_fit
    m_dna, _ = fit(corrected, cfg, "cpu")
    assert 0.6 < float(torch.exp(m_dna.log_dna_scale)) < 1.6
    only_inputs = TrainConfig(**{**cfg.to_dict(), "hparams": hp, "dna_decoder": False})
    d2 = build_training(t_dna, fs, only_inputs, splits)
    m2, _ = fit(d2, only_inputs, "cpu")
    assert not hasattr(m2, "log_dna_scale")
    rr = np.array(r)
    assert count_fit(m_dna, corrected, rr, "cpu")["nll_per_feature"] < count_fit(m2, d2, rr, "cpu")["nll_per_feature"]
    # bundle: the manifest says what was done, embed repeats it, and a bundle from before the term loads with it off
    from ecoli_atlas.artifacts import save_bundle
    save_bundle(tmp_path / "b", m_dna, corrected, cfg, "cpu", 5)
    man = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert man["dna"]["decoder"] and man["dna"]["inputs"] and 0.6 < man["dna"]["scale"] < 1.6 and man["dna"]["samples_with_dna"] == len(dirty)
    bundle = ModelBundle.load(tmp_path / "b")
    qc = pd.read_csv(qc_path, sep="\t", dtype=str, keep_default_na=False).set_index("GSM")
    ids = dirty[:3] + [corrected.ids[i] for i in others[:3]]
    res = embed_counts(bundle, noisy[ids], qc)
    z = encode(m_dna, corrected.X, corrected.C, "cpu")["z"]
    assert np.allclose(res["embedding"].to_numpy(), z[[corrected.ids.index(i) for i in ids]], atol=1e-4)
    assert (res["samples"]["dna_reads_removed"][dirty[:3]] > 0).all() and (res["samples"]["dna_reads_removed"][ids[3:]] == 0).all()
    # embed reports the DNA estimate and flags a sample above the bundle's limit; the command warns and still embeds
    assert np.allclose(res["samples"]["dna_fraction_est"], [float(qc.loc[i, "full_dna_fraction_est"]) for i in ids]) and not res["samples"]["high_dna"].any()
    qc_high = qc.copy()
    qc_high.loc[ids[0], "full_dna_fraction_est"] = "0.25"
    flagged = embed_counts(bundle, noisy[ids], qc_high)["samples"]
    assert list(flagged["high_dna"]) == [1, 0, 0, 0, 0, 0] and embed_counts(bundle, noisy[ids])["samples"]["high_dna"].sum() == 0
    noisy[ids].to_csv(tmp_path / "q.tsv", sep="\t")
    qc_high.to_csv(tmp_path / "qc_high.tsv", sep="\t")
    assert main(["embed", "--counts", str(tmp_path / "q.tsv"), "--qc", str(tmp_path / "qc_high.tsv"), "--models", str(tmp_path / "b"), "--out", str(tmp_path / "e"),
                 "--device", "cpu"]) == 0
    assert pd.read_csv(tmp_path / "e" / "samples.tsv", sep="\t", index_col=0)["high_dna"].sum() == 1
    assert not np.allclose(embed_counts(bundle, noisy[ids]).get("embedding").to_numpy()[:3], res["embedding"].to_numpy()[:3], atol=1e-3)   # no QC table: no correction
    m_off, _ = fit(raw, off, "cpu", epochs=2)
    save_bundle(tmp_path / "old", m_off, raw, off, "cpu", 5)
    man = json.loads((tmp_path / "old" / "manifest.json").read_text())
    for key in ("dna_decoder", "dna_inputs"):
        del man["config"][key]
    del man["dna"]
    (tmp_path / "old" / "manifest.json").write_text(json.dumps(man))
    old = ModelBundle.load(tmp_path / "old")
    assert not old.config.dna_decoder and not old.config.dna_inputs and not hasattr(old.model, "log_dna_scale")
    z_off = encode(m_off, raw.X, raw.C, "cpu")["z"]
    assert np.allclose(embed_counts(old, noisy[ids], qc)["embedding"].to_numpy(), z_off[[raw.ids.index(i) for i in ids]], atol=1e-4)
