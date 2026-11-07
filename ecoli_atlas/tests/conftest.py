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
