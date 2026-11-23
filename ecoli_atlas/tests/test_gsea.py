"""Gene-set enrichment (`ecoli_atlas.gsea`), `figures` gsea panel helpers, `drivers --gsea` and `drivers --axis`."""
import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.gsea import coverage, prerank, ranking_by_symbol, read_gene_sets


def test_the_package_go_sets_load_with_their_release():
    sets, terms, header = read_gene_sets()
    assert len(sets) > 3000 and any("data-version" in h for h in header)
    flagellum = terms.index[terms["name"] == "bacterial-type flagellum"]
    assert len(flagellum) == 1 and "flgb" in sets[flagellum[0]] and terms.loc[flagellum[0], "namespace"] == "cellular_component"
    only_bp, terms_bp, _ = read_gene_sets(namespaces=("biological_process",))
    assert 0 < len(only_bp) < len(sets) and set(terms_bp["namespace"]) == {"biological_process"}


def test_ranking_by_symbol_keeps_one_score_per_symbol():
    r = ranking_by_symbol([1.0, -3.0, 2.0, 5.0, np.nan], ["thrA", "THRA", "", "lacZ", "dnaK"])
    assert list(r.index) == ["lacz", "thra"] and r["thra"] == -3.0          # the larger |score| of the two thrA rows
    assert coverage(r, {"x": ["lacz"]}) == 0.5


def test_prerank_is_two_sided(tmp_path):
    pytest.importorskip("gseapy")
    genes = [f"g{i:03d}" for i in range(200)]
    ranking = pd.Series(np.linspace(2, -2, 200), index=genes)
    sets = {"TOP": genes[:15], "BOTTOM": genes[-15:], "SPREAD": genes[::13], "TINY": genes[:3]}
    terms = pd.DataFrame({"namespace": "biological_process", "name": ["top", "bottom", "spread", "tiny"]}, index=list(sets))
    res = prerank(ranking, sets, terms, permutations=200, min_size=5).set_index("term")
    assert "TINY" not in res.index                                        # below min_size
    assert res.loc["TOP", "nes"] > 1.5 and res.loc["BOTTOM", "nes"] < -1.5 and abs(res.loc["SPREAD", "nes"]) < 1.5
    assert res.loc["TOP", "fdr"] < 0.05 and res.loc["BOTTOM", "fdr"] < 0.05 and res.loc["TOP", "name"] == "top"
    assert res["pval"].min() >= 1 / 200


def test_drivers_of_an_axis_with_gsea(trained, train_args, atlas_files, tmp_path):
    pytest.importorskip("gseapy")
    axes_out = tmp_path / "axes"
    assert main(["axes", "--models", str(trained), "--metadata", str(atlas_files["metadata"]), "--out", str(axes_out), "--formats", "png",
                 "--axis", "phase:growth_phase:mid_exponential,stationary"]) == 0
    feat = pd.read_csv(atlas_files["features"], sep="\t", keep_default_na=False)
    sym = feat.loc[feat["Type"] == "CDS", "Symbol"].tolist()
    sets = pd.DataFrame({"term": ["T:PHASE", "T:OTHER"], "namespace": "biological_process", "name": ["phase genes", "other genes"],
                         "genes": [",".join(sym[:10]), ",".join(sym[20:40])]})           # the fixture raises the first ten in stationary
    sets.to_csv(tmp_path / "sets.tsv", sep="\t", index=False)
    out = tmp_path / "drivers"
    args = ["drivers", *train_args, "--models", str(trained), "--out", str(out), "--device", "cpu", "--formats", "png", "--top", "5",
            "--axis", str(axes_out / "axis_phase_summary.json"), "--gsea", "--gene-sets", str(tmp_path / "sets.tsv"),
            "--gsea-permutations", "200", "--study-boot", "20"]
    assert main(args) == 0
    name = "axis_phase_mid_exponential_to_stationary"
    table = pd.read_csv(out / f"drivers_{name}.tsv", sep="\t")
    report = json.loads((out / "drivers_axis_phase.json").read_text())
    summary = report["contrasts"][0]
    assert report["axis"]["name"] == "phase" and report["classes"] == ["mid_exponential", "stationary"]
    assert summary["sum_of_attributions"] == pytest.approx(summary["score_gap_high_minus_baseline"], rel=0.05)   # completeness
    assert table.head(10)["feature"].isin(set(feat["Feature_ID"][:10])).sum() >= 6
    gsea = pd.read_csv(out / f"gsea_{name}.tsv", sep="\t").set_index("term")
    assert gsea.loc["T:PHASE", "nes"] > 0 and summary["gsea"]["terms_tested"] == 2
    assert (out / f"fig_gsea_{name}.png").stat().st_size > 5000
    # an axis summary that is not one, or a category the axis does not have, is refused
    assert main(args[:-9] + ["--axis", str(tmp_path / "sets.tsv")]) == 2
    assert main(args + ["--contrast", "mid_exponential:nope"]) == 2
