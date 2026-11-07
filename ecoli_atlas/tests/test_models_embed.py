import numpy as np
import torch

from ecoli_atlas.artifacts import ModelBundle
from ecoli_atlas.embed import embed_counts
from ecoli_atlas.models import FactorEncoder, ResidualAE
from ecoli_atlas.preprocess import GeneMap


def test_model_shapes_and_arch_roundtrip():
    m = FactorEncoder(50, 6, 2, width=32, depth=3, dropout=0.2)
    z, logits = m(torch.zeros(4, 50))
    assert z.shape == (4, 6) and logits.shape == (4, 2)
    a = m.arch()
    assert a == dict(n_genes=50, width=32, emb_dim=6, depth=3, dropout=0.2, n_out=2)
    m2 = FactorEncoder.from_arch(a)
    m2.load_state_dict(m.state_dict())

    r = ResidualAE(50, 22, 24, width=128, depth=2, dropout=0.1)
    res, z, xhat = r(torch.zeros(4, 50), torch.zeros(4, 22))
    assert res.shape == (4, 24) and z.shape == (4, 46) and xhat.shape == (4, 50)
    assert r.arch() == dict(d_raw=50, width=128, d_res=24, depth=2, dropout=0.1, d_block=22)
    ResidualAE.from_arch(r.arch()).load_state_dict(r.state_dict())


def test_bundle_save_load_roundtrip(tmp_path, bundle, counts_symbols, feature_table):
    out = tmp_path / "models"
    bundle.save(out)
    for f in bundle.factor_order:
        assert (out / f"{f}_encoder.pt").exists()
    assert (out / "residual_ae.pt").exists() and (out / "preprocessing.pt").exists() and (out / "manifest.json").exists()
    # the saved bundle must load with weights_only=True (no pickled numpy objects)
    torch.load(out / "preprocessing.pt", map_location="cpu", weights_only=True)
    loaded = ModelBundle.load(out)
    assert loaded.genes == bundle.genes and loaded.factor_order == bundle.factor_order
    assert loaded.factors["growth_phase"].threshold == 0.3
    assert loaded.embedding_columns()[:2] == ["growth_phase[0]", "growth_phase[1]"]
    assert loaded.embedding_columns()[-1] == "residual[4]"

    gm = GeneMap.from_feature_table(feature_table)
    a = embed_counts(counts_symbols, bundle, gm)
    b = embed_counts(counts_symbols, loaded, gm)
    np.testing.assert_allclose(a.embedding.values, b.embedding.values, atol=1e-6)


def test_embed_outputs(bundle, counts_symbols, feature_table):
    gm = GeneMap.from_feature_table(feature_table)
    res = embed_counts(counts_symbols, bundle, gm)
    n = counts_symbols.shape[1]
    assert res.embedding.shape == (n, bundle.embedding_dim)
    assert list(res.embedding.index) == list(counts_symbols.columns)
    assert res.coverage.n_present == len(bundle.genes) and res.coverage.fraction == 1.0
    cols = res.predictions.columns
    assert "growth_phase_pred" in cols and "growth_phase_prob_stationary" in cols and "is_lab_strain_confidence" in cols
    assert set(res.predictions["medium_type_pred"]) <= {"complex", "defined"}
    # confidence equals the max probability
    p = res.predictions[["growth_phase_prob_exponential", "growth_phase_prob_stationary"]].values
    np.testing.assert_allclose(res.predictions["growth_phase_confidence"].values, p.max(1))
    assert list(res.block_layout.columns) == ["block", "kind", "start", "end", "width"]
    assert res.block_layout["width"].sum() == bundle.embedding_dim


def test_threshold_vs_argmax(bundle, counts_symbols, feature_table):
    gm = GeneMap.from_feature_table(feature_table)
    arg = embed_counts(counts_symbols, bundle, gm)
    thr = embed_counts(counts_symbols, bundle, gm, apply_thresholds=("growth_phase",))
    p_stat = arg.predictions["growth_phase_prob_stationary"].values
    expected = np.where(p_stat >= 0.3, "stationary", "exponential")
    assert list(thr.predictions["growth_phase_pred"]) == list(expected)
    assert list(arg.predictions["growth_phase_pred"]) == list(np.where(p_stat >= 0.5, "stationary", "exponential"))
    # other factors unaffected
    assert list(thr.predictions["medium_type_pred"]) == list(arg.predictions["medium_type_pred"])


def test_missing_genes_reported(bundle, counts_symbols, feature_table):
    gm = GeneMap.from_feature_table(feature_table)
    sub = counts_symbols.drop(index=["g000", "g010", "g020"])
    res = embed_counts(sub, bundle, gm)
    assert res.coverage.missing == ["g000", "g010", "g020"]
    assert abs(res.coverage.fraction - 57 / 60) < 1e-9
