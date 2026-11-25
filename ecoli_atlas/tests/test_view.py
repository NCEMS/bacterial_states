"""`ecoli_atlas view`: the interactive 3-D page of the atlas."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.compare import read_embedding
from ecoli_atlas.io import read_counts
from ecoli_atlas.viewer.build import build_payload, encode_field, title_group
from ecoli_atlas.viewer.layouts import build_layouts

QUERY = ["GSM0000", "GSM0105", "GSM0300"]


def payload_of(html: str) -> dict:
    m = re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


@pytest.fixture(scope="module")
def text_tables(atlas_files, tmp_path_factory):
    d = tmp_path_factory.mktemp("text")
    meta = pd.read_csv(atlas_files["metadata"], sep="\t", dtype=str, keep_default_na=False)
    studies = pd.DataFrame({"GSE": sorted(meta["GSE"].unique())})
    studies["title"] = "Study " + studies["GSE"]
    studies["summary"] = "A <b>summary</b> with a closing </script> tag"
    studies["pubmed_ids"] = "123"
    studies.to_csv(d / "studies.tsv", sep="\t", index=False)
    text = pd.DataFrame({"GSM": meta["GSM"], "GSE": meta["GSE"], "title": [f"cond{int(g[-2:]) % 3} rep{int(g[-2:]) % 2 + 1}" for g in meta["GSM"]],
                         "runs": "SRR1"})
    text.to_csv(d / "sample_text.tsv", sep="\t", index=False)
    return {"studies": d / "studies.tsv", "sample_text": d / "sample_text.tsv"}


@pytest.fixture(scope="module")
def query_dir(trained, atlas_files, tmp_path_factory):
    d = tmp_path_factory.mktemp("viewquery")
    counts = read_counts(atlas_files["counts"])[QUERY]
    counts.columns = [f"mine{i}" for i in range(len(QUERY))]
    counts.to_csv(d / "counts.tsv", sep="\t")
    qc = pd.read_csv(atlas_files["qc"], sep="\t", index_col=0, keep_default_na=False).loc[QUERY]     # embed needs the samples' QC
    qc.index = list(counts.columns)
    qc.to_csv(d / "qc.tsv", sep="\t")
    assert main(["embed", "--counts", str(d / "counts.tsv"), "--qc", str(d / "qc.tsv"), "--models", str(trained),
                 "--out", str(d / "embedded"), "--device", "cpu"]) == 0
    return d / "embedded"


def test_title_group():
    assert title_group("WT 3h rep2") == "WT 3h"
    assert title_group("planktonic_11_33h_r1") == "planktonic_11_33h"
    assert title_group("Glucose time course, 3 hour time point, biological replicate 1") == "Glucose time course, 3 hour time point"
    assert title_group("dksA_B") == "dksA"


def test_encode_field():
    f = encode_field("strain_group", pd.Series(["K-12", "", None, np.nan, "B", "K-12"]))
    assert f["kind"] == "cat" and f["levels"] == ["B", "K-12"] and f["values"] == [1, -1, -1, -1, 0, 1]
    n = encode_field("p_lab", pd.Series([0.5, None, "0.25"]))
    assert n["kind"] == "num" and n["values"] == [0.5, None, 0.25]


def test_layouts_of_flat_and_full_blocks(trained):
    emb = read_embedding(trained)
    manifest = json.loads((trained / "manifest.json").read_text())
    preds = pd.read_csv(trained / "predictions.tsv", sep="\t", index_col=0).reindex(emb.index)
    lay = {L["name"]: L for L in build_layouts(emb, manifest["embedding_blocks"], preds, emb.iloc[:2], preds.iloc[:2], umap=False, log=lambda m: None)}
    assert {"pca_all", "pca_residual", "block_phase", "block_medium", "block_lab_strain", "axes"} <= set(lay)
    for L in lay.values():
        K = L["atlas"].shape[1]
        assert L["atlas"].shape == (len(emb), K) and L["overlay"].shape == (2, K) and len(L["axes"]) == K
        np.testing.assert_allclose(L["overlay"], L["atlas"][:2], atol=1e-6)      # atlas samples projected onto themselves
    # ten components to choose from in the PCAs; the block layouts fix x to the block axis
    assert lay["pca_all"]["atlas"].shape[1] == 10 and lay["pca_all"]["choose"] == [list(range(10))] * 3
    assert lay["pca_residual"]["atlas"].shape[1] == 10
    b = lay["block_phase"]
    assert b["atlas"].shape[1] == 11 and b["choose"][0] == [0] and b["choose"][1] == list(range(1, 11)) and b["default"] == [0, 1, 2]
    np.testing.assert_allclose(b["atlas"][:, 1:], lay["pca_residual"]["atlas"])
    assert lay["axes"]["atlas"].shape[1] == 3 and lay["axes"].get("choose") is None
    few = {L["name"]: L for L in build_layouts(emb, manifest["embedding_blocks"], preds, umap=False, n_pcs=4, log=lambda m: None)}
    assert few["pca_all"]["atlas"].shape[1] == 4 and few["block_medium"]["atlas"].shape[1] == 5
    # the phase axis rises with p(stationary)
    assert np.corrcoef(lay["axes"]["atlas"][:, 0], preds["p_stationary"])[0, 1] > 0
    # a block that is not flat is drawn as its own 3-D PCA
    lay2 = {L["name"]: L for L in build_layouts(emb, manifest["embedding_blocks"], preds, umap=False, flat_share=1.01, log=lambda m: None)}
    assert lay2["block_medium"]["label"].startswith("PCA")


def test_view_page(trained, atlas_files, text_tables, query_dir, tmp_path):
    presets = tmp_path / "presets.json"
    presets.write_text(json.dumps([{"id": "a", "title": "two studies", "gse": ["GSE00", "GSE99"], "layout": "block_phase", "color_by": "condition"},
                                   {"id": "b", "title": "absent", "gse": ["GSE99"], "layout": "pca_all"},
                                   {"id": "c", "title": "unknown layout", "gse": ["GSE03"], "layout": "umap_all"}]))
    out = tmp_path / "atlas.html"
    assert main(["view", "--models", str(trained), "--metadata", str(atlas_files["metadata"]), "--studies", str(text_tables["studies"]),
                 "--sample-text", str(text_tables["sample_text"]), "--embeddings", f"{query_dir}=mine", "--presets", str(presets),
                 "--counts", str(atlas_files["counts"]), "--qc", str(atlas_files["qc"]),
                 "--no-umap", "--plotly", "cdn", "-k", "5", "--out", str(out)]) == 0
    html = out.read_text()
    assert html.count("</script>") == html.count("<script")          # the study text cannot close the data block
    assert "cdnjs.cloudflare.com/ajax/libs/plotly.js/" in html and "/*JS*/" not in html and "__DATA__" not in html
    p = payload_of(html)
    ids = list(read_embedding(trained).index)
    assert p["ids"] == ids and p["meta"]["n_atlas"] == len(ids)
    fields = {f["name"]: f for f in p["fields"]}
    assert {"split", "gse", "phase_call", "p_stationary", "growth_phase", "growth_phase_basis", "condition", "title_group"} <= set(fields)
    assert all(len(f["values"]) == len(ids) for f in p["fields"])
    for L in p["layouts"]:
        own = L.get("own", L["ncol"])
        assert len(L["atlas"]) == own * len(ids) and len(L["overlay"]) == own * len(QUERY)
        assert len(L["axes"]) == L["ncol"] and len(L["default"]) == 3
    pl = {L["name"]: L for L in p["layouts"]}
    # the block layouts store their own axis and take the residual PCs from the residual PCA
    for h in ("phase", "medium", "lab_strain"):
        if pl[f"block_{h}"]["ncol"] == 11:
            assert pl[f"block_{h}"]["base"] == "pca_residual" and pl[f"block_{h}"]["own"] == 1
    assert "base" not in pl["pca_all"] and "base" not in pl["pca_residual"]
    # conditions: GSE00 varies in phase and genotype, GSE01 only in genotype
    cond = fields["condition"]
    by = {D: {cond["levels"][cond["values"][i]] for i, s in enumerate(ids) if s.startswith(f"GSM{D[3:]}")} for D in ("GSE00", "GSE01")}
    assert p["studies"]["GSE00"]["varying"] == ["growth_phase", "genotype_class", "perturbed_genes"] and len(by["GSE00"]) == 4
    assert p["studies"]["GSE01"]["varying"] == ["genotype_class", "perturbed_genes"] and len(by["GSE01"]) == 2
    assert p["studies"]["GSE00"]["title"] == "Study GSE00" and p["studies"]["GSE00"]["n"] == 10
    assert p["sample_text"]["title"][0] == "cond0 rep1"
    # overlay: atlas samples under new names find themselves first
    ov = p["overlay"]
    assert ov["sample"] == ["mine0", "mine1", "mine2"] and ov["set"] == ["mine"] * 3
    assert all(len(n) == 5 for n in ov["neighbors"])
    assert [ids[n[0][0]] for n in ov["neighbors"]] == QUERY
    assert set(ov["calls"]) == {"phase", "medium", "lab_strain"}
    # QC covariates: composition from the counts, QC from the QC table, as the model computes them
    qcf = p["meta"]["qc_fields"]
    assert {"cov:log_mrna_reads", "cov:log_tRNA_to_mrna", "cov:duplication_rate", "cov:log_substitution_rate", "cov:instrument",
            "cov:paired", "cov:has_qc"} <= set(qcf) and not any(n.startswith("cov:instrument_") for n in qcf)
    assert fields["cov:duplication_rate"]["kind"] == "num" and fields["cov:duplication_rate"]["values"][0] == 0.3
    assert fields["cov:instrument"]["levels"] == ["novaseq"] and fields["cov:paired"]["levels"] == ["no", "yes"]
    sub = fields["cov:log_substitution_rate"]["values"]
    assert sub[ids.index("GSM0502")] is None and sub[ids.index("GSM0000")] is not None      # the fixture gives GSM0502 no rate
    assert fields["cov:length_q50_r2"]["kind"] == "num"                                     # numeric although SE samples lack it
    # the overlay's composition (from embed's covariates.tsv) equals that of the same atlas samples
    oc = ov["covariates"]
    for j, gsm in enumerate(QUERY):
        assert oc["cov:log_mrna_reads"][j] == pytest.approx(fields["cov:log_mrna_reads"]["values"][ids.index(gsm)], abs=1e-3)
    assert oc["cov:has_qc"] == ["yes"] * 3                                                  # embedded with their QC rows
    assert oc["cov:duplication_rate"] == [fields["cov:duplication_rate"]["values"][ids.index(g)] for g in QUERY]
    # presets: absent studies dropped, unknown layout replaced
    pr = {q["id"]: q for q in p["presets"]}
    assert set(pr) == {"a", "c"} and pr["a"]["gse"] == ["GSE00"] and pr["c"]["layout"] == "pca_all"


def test_default_presets_and_minimal_inputs(trained, tmp_path):
    p = build_payload(trained, umap=False, log=lambda m: None)
    assert p["presets"] == []                     # the shipped examples name real GSEs, none in the synthetic atlas
    assert "overlay" not in p and p["sample_text"] == {} and p["meta"]["label_fields"] == [] and p["meta"]["qc_fields"] == []
    assert main(["view", "--models", str(trained), "--no-umap", "--plotly", "cdn", "--out", str(tmp_path / "x.html")]) == 0


def test_umap_layouts(trained):
    pytest.importorskip("umap")
    emb = read_embedding(trained)
    manifest = json.loads((trained / "manifest.json").read_text())
    preds = pd.read_csv(trained / "predictions.tsv", sep="\t", index_col=0).reindex(emb.index)
    names = [L["name"] for L in build_layouts(emb, manifest["embedding_blocks"], preds, emb.iloc[:3], preds.iloc[:3], umap=True, log=lambda m: None)]
    assert "umap_all" in names and "umap_residual" in names


def test_regulon_values():
    from ecoli_atlas.viewer import regulons as rg
    ids = ["CDS_b0001_aaaA", "CDS_b0002_bbbB", "CDS_b0003_cccC", "ncRNA_RF00001", "CDS_b0004_dddD"]
    assert rg.feature_symbols(ids) == (["aaaA", "bbbB", "cccC", "RF00001", "dddD"], ["b0001", "b0002", "b0003", "", "b0004"])
    table = pd.DataFrame({"term": ["regulondb:X+", "regulondb:X-", "imodulon:1", "regulondb:Y+"], "name": ["X (activated)", "X (repressed)", "im", "Y (activated)"],
                          "category": ["", "", "", "activator only"], "genes": ["aaaa,BBBB,zzzZ", "b0003", "aaaA", "nope"]})
    sets = rg.regulon_sets(table, ids)
    assert [s["term"] for s in sets] == ["regulondb:X+", "regulondb:X-", "regulondb:Y+"]       # RegulonDB only, every size
    assert sets[0]["members"] == [0, 1] and sets[0]["absent"] == ["zzzZ"] and sets[1]["members"] == [2] and sets[2]["members"] == []
    with pytest.raises(ValueError, match="no RegulonDB sets"):
        rg.regulon_sets(table[table["term"] == "imodulon:1"], ids)
    # values: mean of the standardised genes the sample has, minus the sample's median over all genes
    E = np.array([[1.0, 2.0, 5.0, 0.0, 3.0], [3.0, 2.0, 1.0, 0.0, 3.0], [2.0, np.nan, 3.0, 0.0, 9.0], [np.nan, np.nan, 3.0, 0.0, 9.0]])
    mu, sd = rg.gene_stats(E)
    assert np.isnan(sd[3]) and mu[1] == 2.0                                                    # a gene that does not vary has no SD
    V = rg.set_values(E, mu, sd, sets)
    Z = (E - mu) / sd
    assert V[0, 0] == pytest.approx(np.nanmean(Z[0, :2]) - np.nanmedian(Z[0]))
    assert V[2, 0] == pytest.approx(Z[2, 0] - np.nanmedian(Z[2]))                              # the gene the sample lacks is left out
    assert np.isnan(V[3, 0]) and np.isnan(V[:, 2]).all()                                       # none of the set's genes; a set without model genes
    # one byte per value: 0 = none, the ends of the scale are the limit
    v = np.array([np.nan, -2.0, -0.5, 0.0, 0.25, 0.5, 7.0])
    back = rg.decode(rg.encode(v, 0.5), 0.5)
    assert np.isnan(back[0]) and back[1] == -0.5 and back[-1] == 0.5 and np.allclose(back[2:6], v[2:6], atol=0.5 / 127)
    assert rg.limits(np.array([[0.1, 3.0], [-0.2, -4.0], [np.nan, 1.0]])).tolist() == [0.5, pytest.approx(3.99, abs=0.02)]


def test_missing_matrix(tmp_path):
    from ecoli_atlas.viewer.regulons import missing_matrix
    pd.DataFrame({"sample": ["s2", "s2", "zz"], "feature": ["f1", "unknown", "f2"]}).to_csv(tmp_path / "missing_genes.tsv", sep="\t", index=False)
    M = missing_matrix(tmp_path / "missing_genes.tsv", ["s1", "s2"], ["f1", "f2"])
    assert M.tolist() == [[False, False], [True, False]]                                       # other samples and features are ignored
    assert not missing_matrix(tmp_path / "absent.tsv", ["s1"], ["f1"]).any() and not missing_matrix(None, ["s1"], ["f1"]).any()


def test_view_regulons_series_and_sheet(trained, atlas_files, query_dir, tmp_path):
    from ecoli_atlas.viewer.regulons import decode
    ids = list(read_embedding(trained).index)
    modules = tmp_path / "modules.tsv"
    modules.write_text("# a comment line\nterm\tnamespace\tname\tcategory\tn_genes\tgenes\n"
                       "regulondb:Pha+\tregulondb_tf_regulon_activated\tPha (activated)\tactivator only\t10\t" + ",".join(f"g{i:03d}" for i in range(10)) + "\n"
                       "regulondb:Med+\tregulondb_tf_regulon_activated\tMed (activated)\t\t3\tg010,b0011,G012\n"
                       "kegg:eco1\tkegg_pathway\tA pathway\t\t2\tg020,g021\n")
    series = tmp_path / "time_courses.tsv"
    pd.DataFrame({"sample": ["GSM0002", "GSM0000", "GSM0001", "GSM0100", "absent"], "series": ["a", "a", "a", "b", "b"], "x": [3, 1, 2, 1, 2],
                  "x_label": "hours"}).to_csv(series, sep="\t", index=False)
    sheet = tmp_path / "samples.csv"
    sheet.write_text("sample_id,condition,replicate,time_h\nmine2,late,1,24\nmine0,early,1,3\nmine1,early,1,6\n")
    counts = read_counts(atlas_files["counts"])[QUERY]
    counts.columns = ["mine0", "mine1", "mine2"]
    counts.to_csv(tmp_path / "mine_counts.tsv", sep="\t")
    common = dict(metadata=atlas_files["metadata"], counts=atlas_files["counts"], qc=atlas_files["qc"], umap=False, log=lambda m: None)
    p = build_payload(trained, modules=modules, time_series=series, embeddings=[f"{query_dir}=mine"], overlay_metadata=[str(sheet)],
                      overlay_series="replicate:time_h", overlay_counts=[f"mine={tmp_path / 'mine_counts.tsv'}"], **common)
    json.dumps(p, allow_nan=False)
    R = p["regulons"]
    assert [s["term"] for s in R["sets"]] == ["regulondb:Pha+", "regulondb:Med+"] and len(R["sets"][0]["genes"]) == 10
    assert R["sets"][1]["genes"] == ["g010", "g011", "g012"]                                   # by symbol, locus tag, any case
    v = decode(R["sets"][0]["values"], R["sets"][0]["limit"])
    assert len(v) == len(ids)
    phase = pd.read_csv(atlas_files["metadata"], sep="\t", index_col=0)["growth_phase"].reindex(ids)
    assert np.nanmean(v[(phase == "stationary").to_numpy()]) > np.nanmean(v[(phase == "mid_exponential").to_numpy()]) + 0.5   # the fixture's phase genes
    # your samples are atlas samples under new names: the same values
    vo = decode(R["overlay"][0], R["sets"][0]["limit"])
    np.testing.assert_allclose(vo, v[[ids.index(g) for g in QUERY]], atol=0.05)
    # time courses of the atlas in the order of x; a series needs two atlas samples
    assert p["series"] == [{"name": "a", "idx": [ids.index("GSM0000"), ids.index("GSM0001"), ids.index("GSM0002")], "x": [1.0, 2.0, 3.0], "x_label": "hours"}]
    # the sample sheet: columns of your samples, in the page's sample order; a series through them by time
    ov = p["overlay"]
    f = {x["name"]: x for x in ov["fields"]}
    assert set(f) == {"condition", "replicate", "time_h"} and [f["condition"]["levels"][k] for k in f["condition"]["values"]] == ["early", "early", "late"]
    assert ov["series"] == [{"name": "replicate 1", "idx": [0, 1, 2], "x": ["3", "6", "24"], "x_label": "time_h"}]
    assert "seed_agreement" not in ov and p["meta"]["n_seeds"] == 0
    # all of it is optional, and the options that need another one say so
    q = build_payload(trained, embeddings=[str(query_dir)], **common)
    assert "regulons" not in q and "series" not in q and "fields" not in q["overlay"] and "series" not in q["overlay"]
    r = build_payload(trained, modules=modules, embeddings=[str(query_dir)], **common)
    assert np.isnan(decode(r["regulons"]["overlay"][0], 1.0)).all()                            # no counts of your samples: no values
    with pytest.raises(ValueError, match="--counts"):
        build_payload(trained, modules=modules, umap=False, log=lambda m: None)
    with pytest.raises(ValueError, match="--embeddings"):
        build_payload(trained, overlay_metadata=[str(sheet)], umap=False, log=lambda m: None)
    with pytest.raises(ValueError, match="no column 'day'"):
        build_payload(trained, embeddings=[str(query_dir)], overlay_metadata=[str(sheet)], overlay_series="replicate:day", **common)
    out = tmp_path / "full.html"
    assert main(["view", "--models", str(trained), "--counts", str(atlas_files["counts"]), "--modules", str(modules), "--time-series", str(series),
                 "--embeddings", f"{query_dir}=mine", "--overlay-metadata", str(sheet), "--overlay-series", "replicate:time_h",
                 "--overlay-counts", str(tmp_path / "mine_counts.tsv"), "--metadata", str(atlas_files["metadata"]),
                 "--no-umap", "--plotly", "cdn", "--out", str(out)]) == 0
    assert payload_of(out.read_text())["regulons"]["sets"][0]["name"] == "Pha (activated)"


def test_seed_agreement(tmp_path):
    from ecoli_atlas.viewer.build import seed_agreement
    raw = pd.DataFrame({"p_stationary": [0.9, 0.2, np.nan, 0.6], "p_minimal": [0.1, 0.1, 0.1, 0.1]}, index=["a", "b", "c", "d"])
    assert seed_agreement(tmp_path, raw) == ({}, 0)                                            # a bundle without seeds
    for n, ps in enumerate([[0.9, 0.2, 0.5, 0.6], [0.8, 0.7, 0.5, 0.4], [0.7, 0.1, 0.5, 0.3]], start=1):
        (tmp_path / f"seed_{n}").mkdir()
        pd.DataFrame({"p_stationary": ps, "p_minimal": 0.2}, index=["a", "b", "c", "d"]).to_csv(tmp_path / f"seed_{n}" / "predictions.tsv", sep="\t")
    agree, n = seed_agreement(tmp_path, raw)
    assert n == 3 and set(agree) == {"phase", "medium"}                                        # no p_lab in these tables
    assert agree["phase"].tolist()[:2] == [3.0, 2.0] and np.isnan(agree["phase"]["c"]) and agree["phase"]["d"] == 1.0
    assert agree["medium"].tolist() == [3.0] * 4


def test_shipped_presets():
    from importlib import resources
    presets = json.loads(resources.files("ecoli_atlas.viewer").joinpath("presets.json").read_text(encoding="utf-8"))
    assert len({p["id"] for p in presets}) == len(presets) >= 11
    assert all({"id", "title", "gse", "layout", "color_by", "caption"} <= set(p) and p["gse"] for p in presets)
    assert any(p["gse"] == ["GSE147611"] for p in presets)
