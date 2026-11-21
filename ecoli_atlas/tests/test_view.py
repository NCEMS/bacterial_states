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
    assert main(["embed", "--counts", str(d / "counts.tsv"), "--models", str(trained), "--out", str(d / "embedded"), "--device", "cpu"]) == 0
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
        assert L["atlas"].shape == (len(emb), 3) and L["overlay"].shape == (2, 3) and len(L["axes"]) == 3
        np.testing.assert_allclose(L["overlay"], L["atlas"][:2], atol=1e-6)      # atlas samples projected onto themselves
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
        assert len(L["atlas"]) == 3 * len(ids) and len(L["overlay"]) == 3 * len(QUERY)
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
    assert sub[ids.index("GSM0002")] is None and sub[ids.index("GSM0000")] is not None      # no rate given for GSM..02
    assert fields["cov:length_q50_r2"]["kind"] == "num"                                     # numeric although SE samples lack it
    # the overlay's composition (from embed's covariates.tsv) equals that of the same atlas samples
    oc = ov["covariates"]
    for j, gsm in enumerate(QUERY):
        assert oc["cov:log_mrna_reads"][j] == pytest.approx(fields["cov:log_mrna_reads"]["values"][ids.index(gsm)], abs=1e-3)
    assert oc["cov:duplication_rate"] == [None] * 3 and oc["cov:has_qc"] == ["no"] * 3      # embedded without --qc
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
