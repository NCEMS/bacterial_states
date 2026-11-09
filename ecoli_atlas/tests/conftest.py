"""Synthetic fixtures: a tiny feature table, count tables in the three inGEST flavours, and a
randomly initialised model bundle with the real file layout."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ecoli_atlas.artifacts import FactorArtifact, ModelBundle
from ecoli_atlas.models import FactorEncoder, ResidualAE

N_GENES = 60
N_SAMPLES = 40


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(0)


@pytest.fixture(scope="session")
def feature_table():
    rows = []
    for i in range(N_GENES):
        rows.append(dict(feature="gene", **{"class": "protein_coding"}, symbol=f"g{i:03d}", locus_tag=f"b{i:04d}"))
    # two rRNA and one tRNA, one symbol shared by two loci (ambiguous), one locus without symbol
    rows += [dict(feature="gene", **{"class": "rRNA"}, symbol="rrsA", locus_tag="b9001"),
             dict(feature="gene", **{"class": "rRNA"}, symbol="rrsB", locus_tag="b9002"),
             dict(feature="gene", **{"class": "tRNA"}, symbol="ileV", locus_tag="b9003"),
             dict(feature="gene", **{"class": "protein_coding"}, symbol="dupX", locus_tag="b9101"),
             dict(feature="gene", **{"class": "protein_coding"}, symbol="dupX", locus_tag="b9102"),
             dict(feature="gene", **{"class": "protein_coding"}, symbol=np.nan, locus_tag="b9201")]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def model_genes():
    return [f"g{i:03d}" for i in range(N_GENES)]


@pytest.fixture(scope="session")
def counts_symbols(rng, model_genes):
    """genes x samples, symbol-keyed, plus a gene the model does not know."""
    data = rng.negative_binomial(5, 0.05, size=(N_GENES + 1, N_SAMPLES)).astype(float)
    df = pd.DataFrame(data, index=model_genes + ["novelGene"], columns=[f"S{j}" for j in range(N_SAMPLES)])
    df.index.name = "gene"
    return df


@pytest.fixture(scope="session")
def bundle(model_genes, rng, tmp_path_factory):
    torch.manual_seed(0)
    factors = {}
    layout, cursor = [], 0
    specs = {"growth_phase": (3, ["exponential", "stationary"], 0.3, "stationary"),
             "medium_type": (4, ["complex", "defined"], 0.9, "defined"),
             "is_lab_strain": (2, ["False", "True"], 0.5, "True")}
    for f, (dim, classes, thr, pos) in specs.items():
        m = FactorEncoder(N_GENES, dim, 2, width=16, depth=2, dropout=0.1)
        factors[f] = FactorArtifact(f, m, classes, thr, pos)
        layout.append({"block": f, "kind": "factor", "start": cursor, "end": cursor + dim})
        cursor += dim
    res_dim = 5
    layout.append({"block": "residual", "kind": "residual", "start": cursor, "end": cursor + res_dim})
    residual = ResidualAE(N_GENES, cursor, res_dim, width=16, depth=1, dropout=0.0)
    prep = {
        "encoder_norm": "zscore_cpm",
        "genes": list(model_genes),
        "zscore_mu": rng.uniform(100, 1000, N_GENES).astype(np.float32),
        "zscore_sd": rng.uniform(10, 100, N_GENES).astype(np.float32),
        "tb_mu": rng.normal(size=cursor).astype(np.float32),
        "tb_sd": rng.uniform(0.5, 2, cursor).astype(np.float32),
        "factor_order": list(specs),
        "factor_emb_dims": {f: specs[f][0] for f in specs},
        "res_dim": res_dim,
        "top_emb_dim": cursor + res_dim,
        "block_layout": layout,
        "factor_classes": {f: specs[f][1] for f in specs},
        "factor_class_dtype": {f: "str" for f in specs},
        "factor_thresholds": {f: specs[f][2] for f in specs},
        "positive_class_index": {f: specs[f][1].index(specs[f][3]) for f in specs},
        "seed": 0,
    }
    b = ModelBundle(factors, residual, prep, manifest={"seed": 0, "factor_order": list(specs)})
    return b.to("cpu")


N_STUDIES, PER_STUDY, N_GENES = 14, 12, 60
OXYGEN_STUDIES = ("GSE1009", "GSE1012")
UTI_STUDY = "GSE1012"


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory, feature_table):
    """A 14-study corpus with phase / medium / lab-strain signal. Two studies (GSE1009, GSE1012) vary
    oxygen (odd samples low oxygen) so they can serve as oxygen case studies; study GSE1012 doubles as a
    paired "patient" design through ``uti_titles``."""
    d = tmp_path_factory.mktemp("synthetic")
    rng = np.random.RandomState(1)
    samples, rows = [], []
    for s in range(N_STUDIES):
        gse = f"GSE{1000 + s}"
        lab = s % 3 != 0
        for j in range(PER_STUDY):
            sid = f"GSM{s * 100 + j}_SRR{s * 100 + j}"
            phase = "exponential" if j % 2 == 0 else "late_stationary"
            medium = "complex" if s % 2 == 0 else "defined"
            samples.append(sid)
            rows.append({"sample_id": sid, "gse": gse, "strain": "MG1655" if lab else f"clinical{s}",
                         "strain_background": "K-12" if lab else "clinical isolate", "is_lab_strain": lab,
                         "medium_type": medium, "base_medium": "LB" if medium == "complex" else "M9",
                         "growth_phase": phase, "OD600": 0.5 if phase == "exponential" else 2.0,
                         "carbon_source": np.nan if medium == "complex" else "glucose", "temperature(C)": 37,
                         "oxygen": ("anaerobic" if (gse in OXYGEN_STUDIES and j % 2 == 1) else "aerobic"), "culture_vessel": "flask", "culture_volume(mL)": 50,
                         "stress": np.nan if j < 10 else "heat", "stress_type": np.nan, "stress_subtype": np.nan})
    meta = pd.DataFrame(rows).set_index("sample_id")
    # counts: b-number keyed genes x samples, with a signal that depends on phase and medium
    base = rng.gamma(2.0, 200, size=N_GENES)
    X = np.zeros((N_GENES, len(samples)))
    for i, sid in enumerate(samples):
        m = meta.loc[sid]
        eff = np.ones(N_GENES)
        eff[:15] *= 3.0 if m["growth_phase"] == "exponential" else 0.4
        eff[15:30] *= 2.5 if m["medium_type"] == "defined" else 0.5
        eff[30:40] *= 2.0 if m["is_lab_strain"] else 0.6
        eff[40:48] *= 0.3 if m["oxygen"] == "anaerobic" else 1.0
        X[:, i] = rng.poisson(base * eff * rng.uniform(0.7, 1.3))
    genes = [f"b{i:04d}" for i in range(N_GENES)] + ["b9001", "b9003"]  # rRNA, tRNA
    X = np.vstack([X, rng.poisson(5000, size=(2, len(samples)))])
    counts = pd.DataFrame(X, index=pd.Index(genes, name="Geneid"), columns=samples)
    counts.to_csv(d / "merged_count_table.txt", sep="\t")
    meta.to_csv(d / "metadata_revised.csv")
    feature_table.to_csv(d / "feature_table.txt", sep="\t", index=False)
    return d




def uti_titles():
    """GSMs of UTI_STUDY paired as <patient>_<UTI|UR> for the paired case-study test (6 patients)."""
    s = 12  # GSE1012 -> study index 12
    gsms = [f"GSM{s * 100 + j}" for j in range(PER_STUDY)]
    titles = [f"P{j // 2:02d}_{'UTI' if j % 2 == 0 else 'UR'}" for j in range(PER_STUDY)]
    return gsms, titles


@pytest.fixture(scope="session")
def trained_models(synthetic, tmp_path_factory):
    """A one-epoch bundle (+ atlas) trained on the synthetic corpus with the oxygen studies forced into TEST."""
    from ecoli_atlas.cli import main
    out = tmp_path_factory.mktemp("trained") / "models"
    rc = main(["train", "--counts", str(synthetic / "merged_count_table.txt"),
               "--metadata", str(synthetic / "metadata_revised.csv"),
               "--feature-table", str(synthetic / "feature_table.txt"),
               "--out", str(out), "--no-tune", "--factor-epochs", "1", "--top-epochs", "1",
               "--min-class-n", "5", "--withheld-study", "GSE1013", "--case-studies", ",".join(OXYGEN_STUDIES),
               "--test-frac", "0.15", "--device", "cpu", "--write-atlas", "--threads", "2"])
    assert rc == 0
    return out
