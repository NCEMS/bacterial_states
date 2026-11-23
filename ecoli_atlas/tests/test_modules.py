"""Curated gene modules: the module table built from the downloaded sources, and `drivers --modules`."""
import json

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.modules import build_module_table, membership, read_module_table, score_modules, write_module_table


def _sources(d):
    p = d / "precise1k"
    p.mkdir(parents=True)
    pd.DataFrame({"regulator": ["ArcA", "ArcA", "ArcA", "ArcA", "RpoS", "RpoS"], "gene_name": ["cyoA", "sdhC", "hyaA", "thrC", "osmY", "dps"],
                  "gene_id": ["b0432", "b0721", "b0972", "b0004", "b4376", "b0812"], "effect": ["-", "-", "+", "?", "+", "+"],
                  "evidence": [1, 1, 0, 1, 1, 1],
                  "source": ["RegulonDB_TF"] * 4 + ["RegulonDB_sigma"] * 2}).to_csv(p / "TRN.csv", index=False)
    pd.DataFrame({"Fur-1": [3.0, -0.1, 2.5, 0.05], "Empty": [0.1, 0.2, 0.3, 0.4]}, index=["b0001", "b0002", "b0003", "b0004"]).to_csv(p / "M.csv")
    pd.DataFrame({"imodulon_size": [2, 0], "system_category": ["Metabolism", "Unknown"]}, index=["Fur-1", "Empty"]).to_csv(p / "imodulon_table.csv")
    k = d / "kegg"
    k.mkdir()
    (k / "list_pathway_eco.txt").write_text("eco00010\tGlycolysis - Escherichia coli K-12 MG1655\neco01100\tMetabolic pathways - Escherichia coli K-12 MG1655\n")
    (k / "link_pathway_eco.txt").write_text("eco:b0001\tpath:eco00010\neco:b0002\tpath:eco00010\neco:b0001\tpath:eco01100\n")
    (k / "list_module.txt").write_text("M00001\tGlycolysis core\n")
    (k / "link_module_eco.txt").write_text("eco:b0003\tmd:eco_M00001\neco:b9999\tmd:eco_M00001\n")
    (k / "brite_pathways.keg").write_text("+C\tMap number\n!\nAMetabolism\nB  Carbohydrate metabolism\nC    00010  Glycolysis\n")
    (k / "brite_modules.keg").write_text("+D\tModule\n!\nAPathway modules\nB  Carbohydrate metabolism\nC    Central\n"
                                         "D      M00001  Glycolysis core [PATH:map00010]\n")


def test_build_module_table_from_the_sources(tmp_path):
    _sources(tmp_path / "src")
    feat = pd.DataFrame({"Symbol": ["thrL", "thrA", "thrB", "thrC", "cyoA", "sdhC", "hyaA", "osmY", "dps"],
                         "Locus_Tag": ["b0001", "b0002", "b0003", "b0004", "b0432", "b0721", "b0972", "b4376", "b0812"]})
    t, stats = build_module_table(tmp_path / "src", feat)
    rows = t.set_index("term")
    # regulons only by direction: no whole-regulon set, the unknown-effect link (thrC) in none
    assert "regulondb:ArcA" not in rows.index and "regulondb:RpoS" not in rows.index
    assert rows.loc["regulondb:ArcA-", "genes"] == "cyoA,sdhC" and rows.loc["regulondb:ArcA+", "genes"] == "hyaA"
    assert rows.loc["regulondb:ArcA+", "name"] == "ArcA (activated)" and rows.loc["regulondb:ArcA-", "name"] == "ArcA (repressed)"
    assert rows.loc["regulondb:RpoS+", "genes"] == "osmY,dps" and rows.loc["regulondb:RpoS+", "namespace"] == "regulondb_sigma_regulon_activated"
    assert not t.loc[t["term"].str.startswith("regulondb"), "genes"].str.contains("thrC").any()
    assert rows.loc["regulondb:ArcA+", "category"] == "" == rows.loc["regulondb:ArcA-", "category"]     # both directions
    assert rows.loc["regulondb:RpoS+", "category"] == "sigma factor"
    assert rows.loc["kegg:eco00010", "category"] == "Carbohydrate metabolism" == rows.loc["kegg:eco_M00001", "category"]
    assert rows.loc["imodulon:Fur-1", "genes"] == "thrL,thrB" and "imodulon:Empty" not in rows.index
    assert rows.loc["kegg:eco00010", "name"] == "Glycolysis" and "kegg:eco01100" not in rows.index      # overview map dropped
    assert rows.loc["kegg:eco_M00001", "genes"] == "thrB" and stats["kegg"]["members_matched"] < stats["kegg"]["members_listed"]
    write_module_table(t, tmp_path / "m.tsv", ["test header"])
    back = read_module_table(tmp_path / "m.tsv")
    assert len(back) == len(t) and (tmp_path / "m.tsv").read_text().startswith("# test header")
    kept, G = membership(back, feat["Symbol"], feat["Locus_Tag"], min_size=2, max_size=3)
    assert set(kept["term"]) == {"regulondb:ArcA-", "regulondb:RpoS+", "imodulon:Fur-1", "kegg:eco00010"}
    assert G.shape == (len(feat), len(kept)) and G.sum() == sum(kept["genes"].str.count(",") + 1)


def test_score_modules_finds_the_module_that_carries_the_score():
    rng = np.random.default_rng(0)
    F, J = 60, 6
    attr = rng.normal(0, 0.01, (2, J, F))
    attr[:, :, :5] += 0.2                                          # genes 0-4 carry the score
    unit = np.arange(F)
    mods = pd.DataFrame({"term": ["hit", "noise"], "namespace": ["test_a", "test_a"], "name": ["hit", "noise"], "genes": ["", ""]})
    G = np.zeros((F, 2), np.float32)
    G[:5, 0] = 1
    G[20:25, 1] = 1
    res = score_modules(attr, unit, mods, G, [f"g{i}" for i in range(F)], log2fc=np.ones(F), n_null=500).set_index("term")
    assert res.loc["hit", "z_net"] > 5 and abs(res.loc["noise", "z_net"]) < 3 and res.loc["hit", "boot_in_top"] == 1
    assert res.loc["hit", "seeds_same_sign"] == 2 and res.loc["hit", "source"] == "test" and res.loc["hit", "p_net"] <= 2 / 501


def test_drivers_with_modules(trained, train_args, atlas_files, tmp_path):
    feat = pd.read_csv(atlas_files["features"], sep="\t", keep_default_na=False)
    sym = feat.loc[feat["Type"] == "CDS", "Symbol"].tolist()
    pd.DataFrame({"term": ["t:phase", "t:other"], "namespace": ["test_set", "test_set"], "name": ["phase genes", "other genes"],
                  "genes": [",".join(sym[:10]), ",".join(sym[20:40])]}).to_csv(tmp_path / "m.tsv", sep="\t", index=False)
    out = tmp_path / "drivers"
    args = ["drivers", *train_args, "--models", str(trained), "--out", str(out), "--device", "cpu", "--formats", "png",
            "--modules", str(tmp_path / "m.tsv"), "--module-null", "300", "--study-boot", "20"]
    assert main(args) == 0
    res = pd.read_csv(out / "modules_phase_exponential_to_stationary.tsv", sep="\t").set_index("term")
    assert res.loc["t:phase", "z_net"] > 3 and abs(res.loc["t:phase", "z_net"]) > abs(res.loc["t:other", "z_net"])
    assert (out / "fig_modules_phase_exponential_to_stationary.png").stat().st_size > 3000
    summary = json.loads((out / "drivers_phase.json").read_text())["contrasts"][0]["modules"]
    assert summary["namespaces"]["test"]["modules"] == 2
    assert main(args[:-6] + ["--modules", str(tmp_path / "m.tsv"), "--module-min", "50"]) == 2      # no module that large


def test_download_modules_dry_run(tmp_path):
    assert main(["download", "modules", "--dest", str(tmp_path / "mods"), "--dry-run"]) == 0
    assert not (tmp_path / "mods").exists()
