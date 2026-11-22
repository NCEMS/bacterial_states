import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.axes import AxisSpec, axis_scores, block_columns, category_frame, pair_order, study_folds, summarize
from ecoli_atlas.cli import main

CATS = ["a", "b", "c", "d", "e"]


def synthetic(n_studies=12, per=20, dim=8, seed=0):
    """An embedding in which one direction carries a latent position (the category rank plus noise)
    and every study has its own offset; study s lacks category (s mod 5)."""
    rng = np.random.RandomState(seed)
    direction = rng.normal(size=dim)
    rows, Z = [], []
    for s in range(n_studies):
        offset = rng.normal(scale=0.6, size=dim)
        for j in range(per):
            r = j % 5
            if r == s % 5:
                continue
            Z.append((r + rng.normal(scale=0.25)) * direction + offset + rng.normal(scale=0.3, size=dim))
            rows.append({"sample": f"S{s:02d}_{j:02d}", "gse": f"G{s:02d}", "split": 1 if s < 9 else 3, "category": CATS[r]})
    frame = pd.DataFrame(rows).set_index("sample")
    return np.array(Z), frame


def test_axis_spec_parsing():
    spec = AxisSpec.parse("phase:growth_phase:x,y,z", holdout=["z", "other"])
    assert (spec.name, spec.column, spec.categories, spec.holdout, spec.fitted) == ("phase", "growth_phase", ["x", "y", "z"], ("z",), ["x", "y"])
    for bad in ("phase:growth_phase", "p:c:x", "p:c:x,x"):
        with pytest.raises(ValueError):
            AxisSpec.parse(bad)
    with pytest.raises(ValueError, match="fewer than two"):
        AxisSpec.parse("p:c:x,y", holdout=["y"])
    # grouped categories: label=value1+value2; a value in two categories is refused
    g = AxisSpec.parse("carbon:carbon_source:sugar=glucose+other_sugar,non_sugar=glycerol+organic_acid,acetate")
    assert g.categories == ["sugar", "non_sugar", "acetate"] and g.values_of("non_sugar") == ("glycerol", "organic_acid")
    assert g.values_of("acetate") == ("acetate",)
    with pytest.raises(ValueError, match="two categories"):
        AxisSpec.parse("c:col:a=x+y,b=y+z")
    meta = pd.DataFrame({"carbon_source": ["glucose", "glycerol", "other_sugar", "mixed", "organic_acid"],
                         "medium_class": ["defined_minimal"] * 4 + ["complex"]}, index=list("abcde"))
    splits = pd.DataFrame({"gse": ["G1", "G2", "G3", "G4", "G5"], "split": 1}, index=list("abcde"))
    frame = category_frame(meta, splits, g, {"medium_class": ["defined_minimal"]})
    assert frame["category"].tolist() == ["sugar", "non_sugar", "sugar", "", ""]
    assert block_columns({"embedding_blocks": [{"block": "phase", "start": 0, "end": 2}, {"block": "residual", "start": 2, "end": 5}]}, "residual", 5) == [2, 3, 4]
    with pytest.raises(ValueError, match="no embedding block"):
        block_columns({"embedding_blocks": []}, "phase", 5)


def test_study_folds_keep_studies_together():
    study = np.repeat([f"g{i}" for i in range(7)], [30, 5, 5, 5, 5, 5, 5])
    fold = study_folds(study, 3, seed=1)
    assert set(fold) == {0, 1, 2} and all(len(set(fold[study == g])) == 1 for g in set(study))


def test_scores_follow_the_order_and_are_out_of_fold():
    Z, frame = synthetic()
    spec = AxisSpec("x", "category", CATS)
    res = axis_scores(Z, frame, spec, n_folds=4)
    s = res["scores"]
    assert s["ordinal"].notna().all() and s["anchor"].notna().all() and set(s["role"]) == {"fitted"}
    med = s.groupby("category")["ordinal"].median().reindex(CATS)
    assert med.is_monotonic_increasing and abs(med["a"]) < 0.25 and abs(med["e"] - 1) < 0.25
    summary = summarize(s, spec)
    for method in ("ordinal", "anchor"):
        block = summary["methods"][method]
        assert block["all"]["spearman"] > 0.9 and block["all"]["within_study"]["share_in_order_mean_over_studies"] > 0.9
        assert block["not_in_embedding_fit"]["n"] == int((frame["split"] == 3).sum())
    assert list(res["thresholds"]) == ["a|b", "b|c", "c|d", "d|e"] and np.all(np.diff(list(res["thresholds"].values())) > 0)
    # out of fold: a study's scores do not depend on its own categories (reverse them for one study)
    g3 = (frame["gse"] == "G03").to_numpy()
    flipped = frame.copy()
    flipped.loc[g3, "category"] = flipped.loc[g3, "category"].map(dict(zip(CATS, CATS[::-1])))
    again = axis_scores(Z, flipped, spec, n_folds=4)["scores"]
    assert np.allclose(again.loc[g3, "ordinal"], s.loc[g3, "ordinal"], atol=1e-6) and np.allclose(again.loc[g3, "anchor"], s.loc[g3, "anchor"], atol=1e-6)


def test_held_out_categories_extrapolate_and_interpolate():
    Z, frame = synthetic()
    res = axis_scores(Z, frame, AxisSpec("x", "category", CATS, holdout=("e",)), n_folds=4)
    s = res["scores"]
    assert set(s.loc[s["category"] == "e", "role"]) == {"holdout"} and "d|e" not in res["thresholds"]
    assert s.loc[s["category"] == "e", "ordinal"].median() > 1.15            # beyond the last fitted category (d = 1)
    check = summarize(s, AxisSpec("x", "category", CATS, holdout=("e",)))["methods"]["ordinal"]["holdout"]["e"]
    assert check["share_above_d"] > 0.9
    mid = axis_scores(Z, frame, AxisSpec("x", "category", CATS, holdout=("c",)), n_folds=4)["scores"]
    med = mid.groupby("category")["ordinal"].median()
    assert med["b"] < med["c"] < med["d"]


def test_an_end_category_of_one_study_is_refused():
    Z, frame = synthetic()
    frame.loc[(frame["category"] == "a") & (frame["gse"] != "G01"), "category"] = ""
    with pytest.raises(ValueError, match="one study only"):
        axis_scores(Z, frame, AxisSpec("x", "category", CATS))


def test_pair_order_counts_every_study_once():
    s = pd.DataFrame({"gse": ["g1"] * 4 + ["g2"] * 2, "rank": [0, 0, 1, 1, 0, 1], "category": list("aabbab"), "v": [0.0, 0.1, 1.0, 1.1, 2.0, 1.0]})
    out = pair_order(s, "v")
    assert out["n_pairs"] == 5 and out["n_studies"] == 2 and out["share_in_order"] == 0.8 and out["share_in_order_mean_over_studies"] == 0.5


def test_cli_writes_tables_and_figures(tmp_path):
    Z, frame = synthetic()
    models = tmp_path / "bundle"
    models.mkdir()
    emb = pd.DataFrame(Z, index=frame.index, columns=[f"z{i + 1}" for i in range(Z.shape[1])])
    emb.index.name = "sample"
    emb.to_csv(models / "embedding.tsv", sep="\t")
    pd.DataFrame({"gse": frame["gse"], "reference": True, "split": frame["split"], "cv_fold": 1.0}).to_csv(models / "splits.csv")
    (models / "manifest.json").write_text(json.dumps({"embedding_blocks": [{"block": "phase", "start": 0, "end": 3}, {"block": "residual", "start": 3, "end": 8}]}))
    meta = pd.DataFrame({"GSM": frame.index, "GSE": frame["gse"].to_numpy(), "letter": frame["category"].to_numpy(),
                         "size": np.where(frame["category"].isin(["a", "b"]), "small", "large"), "treated": "false"})
    meta.to_csv(tmp_path / "meta.tsv", sep="\t", index=False)
    series = pd.DataFrame({"sample": frame.index[frame["gse"] == "G00"], "series": "G00 line", "x": np.arange((frame["gse"] == "G00").sum()) + 1.0})
    series.to_csv(tmp_path / "series.tsv", sep="\t", index=False)
    out = tmp_path / "out"
    args = ["axes", "--models", str(models), "--metadata", str(tmp_path / "meta.tsv"), "--out", str(out), "--formats", "png", "--where", "treated=false"]
    assert main(args + ["--axis", "letter:letter:a,b,c,d,e", "--axis", "size:size:small,large", "--holdout", "e", "--series", str(tmp_path / "series.tsv")]) == 0
    names = {p.name for p in out.iterdir()}
    assert {"axis_letter_scores.tsv", "axis_letter_summary.json", "fig_axis_letter_distributions.png", "fig_axis_letter_series.png",
            "axis_size_scores.tsv", "fig_axes_letter_by_size.png"} <= names
    scores = pd.read_csv(out / "axis_letter_scores.tsv", sep="\t", index_col=0)
    assert len(scores) == len(frame) and set(scores["role"].dropna()) == {"fitted", "holdout"}
    summary = json.loads((out / "axis_letter_summary.json").read_text())
    assert summary["holdout"] == ["e"] and summary["where"] == {"treated": ["false"]} and len(summary["weights"]["ordinal"]) == 8
    assert main(args + ["--axis", "letter:letter:a,b,c,d,e", "--blocks", "phase"]) == 0
    assert len(json.loads((out / "axis_letter_summary.json").read_text())["weights"]["ordinal"]) == 3
    assert main(args + ["--axis", "letter:nope:a,b"]) == 1 and main(args + ["--axis", "letter:letter:a,b", "--blocks", "nope"]) == 1
