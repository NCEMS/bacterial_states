import numpy as np
import pandas as pd
import pytest

from ecoli_atlas.features import FeatureSet, read_feature_table
from ecoli_atlas.io import read_counts, sample_id_from_column
from ecoli_atlas.prepare import prepare
from ecoli_atlas.preprocess import (ZScore, align_counts, call_groups, group_calls, pooled_low_profile, composition_covariates, log_expression, low_expression_profile, missing_genes,
                                    reference_profile,
                                    size_factors)


def test_sample_id_from_column():
    assert sample_id_from_column("experiment3_results/KO_1/vg/KO_1_sort.bam") == "KO_1"
    assert sample_id_from_column("results/GSM1/vg/GSM1.bam") == "GSM1"
    assert sample_id_from_column("GSM123") == "GSM123"


def test_read_counts_ingest_layouts(tmp_path):
    ids = ["CDS_b0001_thrL", "rRNA_b0201_rrsH"]
    fc = tmp_path / "all_samples_raw_counts.txt"
    fc.write_text('# Program:featureCounts v2.1.1\nGeneid\tChr\tStart\tEnd\tStrand\tLength\tresults/A/vg/A_sort.bam\tresults/B/vg/B_sort.bam\n'
                  + "".join(f"{g}\tNC\t1\t9\t+\t9\t{i}\t{i + 1}\n" for i, g in enumerate(ids)))
    ext = tmp_path / "all_samples_counts_extended.tsv"
    ext.write_text("Feature_ID\tSymbol\tLocus_Tag\tType\tSource\tRfam\tDbxref\tProduct\tChr\tStart\tEnd\tStrand\tLength\tresults/A/vg/A_sort.bam\n"
                   + "".join(f"{g}\ts\tb\tCDS\tNCBI\t\t\tp\tNC\t1\t9\t+\t9\t{i}\n" for i, g in enumerate(ids)))
    a, b = read_counts(fc), read_counts(ext)
    assert list(a.columns) == ["A", "B"] and list(b.columns) == ["A"] and list(a.index) == ids == list(b.index)
    assert a.loc["rRNA_b0201_rrsH", "B"] == 2
    dup = tmp_path / "dup.tsv"
    dup.write_text("gene\tx/A_sort.bam\ty/A_sort.bam\ng\t1\t2\n")
    with pytest.raises(ValueError, match="duplicate sample names"):
        read_counts(dup)


def test_feature_groups(feature_table):
    fs = FeatureSet.from_table(read_feature_table(feature_table))
    assert fs.n == 83 and int(fs.is_mrna.sum()) == 81            # 80 CDS + 1 pseudo, then 2 other RNAs
    assert fs.ids[-2:] == ["ncRNA_b9100_ssrA", "ncRNA_RF00001"] and not any(i.startswith(("rRNA", "tRNA")) for i in fs.ids)
    assert fs.families == {"16S": ["rRNA_b9201_rrsA"], "23S": ["rRNA_b9202_rrlA"], "5S": ["rRNA_b9203_rrfA"], "tRNA": ["tRNA_b9301_ileV"]}
    back = FeatureSet.from_dict(fs.to_dict())
    assert back.ids == fs.ids and (back.is_mrna == fs.is_mrna).all() and (back.lengths == fs.lengths).all()
    bad = read_feature_table(feature_table)
    bad.loc[bad.index[0], "Type"] = "misc"
    with pytest.raises(ValueError, match="does not know"):
        FeatureSet.from_table(bad)


def test_size_factors_recover_depth():
    rng = np.random.RandomState(1)
    base = rng.gamma(2.0, 300.0, size=200)
    depth = np.array([0.5, 1.0, 2.0, 4.0])
    Y = rng.poisson(base[None, :] * depth[:, None]).astype(float)
    is_mrna = np.array([True] * 190 + [False] * 10)
    Y[:, 0] = 0                                    # never seen: not a reference gene
    Y[0, 1] = 0                                    # a zero in one sample only: skipped for that sample
    ref = reference_profile(Y, is_mrna, min_presence=0.5)
    assert np.isnan(ref[0]) and np.isnan(ref[190:]).all() and not np.isnan(ref[1])
    s = size_factors(Y, ref)
    assert np.allclose(s / s[1], depth, rtol=0.05)
    logx = log_expression(Y, s)
    assert np.abs(logx[0, 5:190] - logx[3, 5:190]).mean() < 0.3      # depth is removed
    with pytest.raises(ValueError, match="no size factor"):
        size_factors(np.zeros((1, 200)), ref)


def test_zscore_weights_and_floor():
    X = np.array([[0.0, 1.0, 5.0], [2.0, 1.0, 5.2], [4.0, 1.0, 4.8]])
    z = ZScore.fit(X, sd_floor=0.5)
    assert np.allclose(z.mu, [2.0, 1.0, 5.0]) and z.sd[1] == 1.0 and z.sd[2] == 0.5     # constant -> 1, small -> floor
    zw = ZScore.fit(X, w=np.array([1.0, 1.0, 0.0]))
    assert np.allclose(zw.mu, [1.0, 1.0, 5.1]) and np.allclose(zw.apply(X)[:, 0], [-1.0, 1.0, 3.0])


def test_align_and_composition(feature_table):
    fs = FeatureSet.from_table(read_feature_table(feature_table))
    counts = pd.DataFrame({"A": 10.0, "B": 20.0}, index=fs.ids[1:] + ["rRNA_b9201_rrsA", "tRNA_b9301_ileV", "unknown_x"])
    Y, fam, cov = align_counts(counts, fs)
    assert Y.shape == (2, fs.n) and (Y[:, 0] == 0).all() and cov.missing == [fs.ids[0]] and cov.n_present == fs.n - 1
    assert fam.tolist() == [[10.0, 0.0, 0.0, 10.0], [20.0, 0.0, 0.0, 20.0]]
    comp = composition_covariates(Y, fam, fs.is_mrna)
    assert comp.shape == (2, 5) and np.isclose(comp[0, 0], np.log10(800.5)) and np.isclose(comp[0, 1], np.log10(10.5 / 800.5))
    assert comp[0, 2] < -3                                             # no 23S reads


def test_prepare_from_an_ingest_count_table(tmp_path):
    res = tmp_path / "x_results"
    res.mkdir()
    (res / "all_samples_raw_counts.txt").write_text(
        "# Program:featureCounts\nGeneid\tChr\tStart\tEnd\tStrand\tLength\tx_results/S1/vg/S1_sort.bam\n"
        "CDS_b0001_thrL\tNC\t1\t9\t+\t9\t7\n")
    (tmp_path / "meta.tsv").write_text("sample\tcondition\nS1\twt\n")
    counts, samples = prepare(None, res / "all_samples_raw_counts.txt", tmp_path / "meta.tsv", tmp_path / "out")
    assert counts.loc["CDS_b0001_thrL", "S1"] == 7 and samples.loc["S1", "condition"] == "wt"
    assert (tmp_path / "out" / "counts.tsv").exists() and samples.loc["S1", "source_column"].endswith("S1_sort.bam")


def test_missing_genes_are_called_in_runs_and_left_out_of_the_statistics():
    rng = np.random.RandomState(3)
    n, g = 40, 60
    base = np.r_[np.full(30, 400.0), np.full(30, 3.0)]            # 30 well expressed genes, then 30 weak ones
    depth = rng.uniform(0.5, 2.0, n)
    Y = rng.poisson(base[None, :] * depth[:, None]).astype(float)
    Y[:8, 5] = 0                                                   # a single well expressed gene
    Y[:8, 40:52] = 0                                               # an island of weak genes
    Y[20, :] = rng.poisson(base * 0.002)                           # a very shallow sample
    order = np.arange(g)
    s = np.r_[depth[:20], 0.002, depth[21:]]
    low = low_expression_profile(Y, s, order)
    assert (low[:30] > 200).all() and (low[30:] > 0).all()        # the samples without the gene do not pull the low end to 0
    M = missing_genes(Y, s, low, order)
    assert M[:8, 5].all() and M[:8, 40:52].all()
    assert M[8:].sum() == 0 and not M[20].any()                    # nothing in the other samples, nor in the shallow one
    assert M[:8].sum() == 8 * 13
    # a weak gene alone is not called; a different chromosome order breaks the island
    Y2 = Y.copy()
    Y2[30, 45] = 0
    assert not missing_genes(Y2, s, low, order)[30].any()
    # missing genes do not enter the size factor or the gene statistics, and are set to the mean
    ref = np.log(base)
    Y3 = Y.copy()
    Y3[:8, 5] = 1                                                  # a stray read on a well expressed gene: still called, and not used
    M3 = missing_genes(Y3, s, low, order)
    assert M3[:8, 5].all() and M3[:8, 40:52].all()
    assert np.allclose(size_factors(Y3, ref, M3)[8:], size_factors(Y3, ref)[8:])
    logx = log_expression(Y, s)
    z = ZScore.fit(logx, missing=M)
    assert abs(z.mu[5] - logx[8:, 5].mean()) < 1e-5 and z.mu[5] > ZScore.fit(logx).mu[5]
    X = z.apply(logx, M)
    assert (X[M] == 0).all() and abs(X[~M[:, 5], 5].mean()) < 1e-5


def test_antisense_reads_veto_a_missing_gene_call():
    rng = np.random.RandomState(4)
    n, g = 30, 40
    base = np.full(g, 300.0)
    depth = rng.uniform(0.5, 2.0, n)
    Y = rng.poisson(base[None, :] * depth[:, None]).astype(float)
    lengths = np.full(g, 1000)
    A = rng.poisson(40.0 * depth[:, None] * np.ones((n, g))).astype(float)      # about 40 antisense reads per kb
    Y[:6, 10:14] = 0
    A[:6, 10:14] = 0                                               # a deleted island: no read on either strand
    Y[6:12, 20:24] = 0                                             # a silenced island: no sense reads, antisense as elsewhere
    Y[12, 30] = 0
    A[12, 30] = 1                                                  # one stray antisense read does not veto
    order = np.arange(g)
    low = low_expression_profile(Y, depth, order)
    plain = missing_genes(Y, depth, low, order)
    assert plain[:6, 10:14].all() and plain[6:12, 20:24].all() and plain[12, 30]
    veto = missing_genes(Y, depth, low, order, antisense=A, lengths=lengths)
    assert veto[:6, 10:14].all() and not veto[6:12, 20:24].any() and veto[12, 30]
    assert not (veto & ~plain).any()                               # antisense reads only remove calls
    assert np.array_equal(low_expression_profile(Y, depth, order, antisense=A, lengths=lengths) > 0, low > 0)
    with pytest.raises(ValueError, match="gene lengths"):
        missing_genes(Y, depth, low, order, antisense=A)


def test_calls_pooled_over_the_samples_of_a_study_that_lack_the_same_genes():
    rng = np.random.RandomState(5)
    g = 120
    base = np.r_[np.full(60, 300.0), np.full(60, 0.4)]            # 60 well expressed genes, then 60 very weak ones
    study = np.repeat([f"S{i}" for i in range(12)], 6)
    n = len(study)
    depth = rng.uniform(0.6, 1.6, n)
    Y = rng.poisson(base[None, :] * depth[:, None]).astype(float)
    strain_b = np.zeros(n, bool)
    strain_b[:6] = True                                            # study S0: all samples of a strain without two islands
    strain_b[6:9] = True                                           # study S1: half of its samples
    Y[np.ix_(strain_b, np.r_[10:40, 70:100])] = 0
    order = np.arange(g)
    low = low_expression_profile(Y, depth, order)
    M = missing_genes(Y, depth, low, order)
    assert M[strain_b][:, 10:40].all() and not M[strain_b][:, 70:100].all()      # the weak island gives a single sample too little
    groups = call_groups(M, study)
    assert len(set(groups[:6])) == 1 and len(set(groups[6:9])) == 1 and len(set(groups[9:12])) == 1
    assert groups[6] != groups[9] and groups[0] != groups[6]      # two strains in S1 are two groups; groups never span studies
    assert len(set(groups)) == 13
    low_pool = pooled_low_profile(Y, depth, groups, order)
    assert (low_pool[60:] > 0).mean() > (low[60:] > 0).mean()
    G = group_calls(Y, depth, groups, low_pool, order)
    assert G[strain_b][:, 10:40].all() and G[strain_b][:, 70:100].all()         # summed over the group the weak island is called
    assert G[~strain_b].mean() < 0.01
    alone = group_calls(Y[:1], depth[:1], np.zeros(1, int), low_pool, order)
    assert not alone.any()                                         # a sample alone in its group gets no group calls
