import numpy as np
import pandas as pd
import pytest

from ecoli_atlas import io, prepare
from ecoli_atlas.preprocess import GeneMap, ZScore, align_to_genes, clean_counts, cpm, cpm_array


def _write_featurecounts(path, counts_df):
    cols = ["Geneid", "Chr", "Start", "End", "Strand", "Length"] + \
           [f"exp_results/{s}/vg/{s}_sort.bam" for s in counts_df.columns]
    with open(path, "w") as fh:
        fh.write('# Program:featureCounts v2.0.6; Command:"featureCounts" "-a" "x.gff"\n')
        fh.write("\t".join(cols) + "\n")
        for g, row in counts_df.iterrows():
            fh.write("\t".join([g, "NC_1", "1", "100", "+", "100"] + [str(int(v)) for v in row]) + "\n")


def test_sample_id_from_column():
    assert io.sample_id_from_column("experiment3_results/KO_1/vg/KO_1_sort.bam") == "KO_1"
    assert io.sample_id_from_column("Paper1st_results/SRR29642643/vg/SRR29642643_sort.bam") == "SRR29642643"
    assert io.sample_id_from_column("GSM2679698_SRR5742252") == "GSM2679698_SRR5742252"
    assert io.sample_id_from_column("x/y.bam") == "y"


def test_read_counts_three_formats(tmp_path, counts_symbols):
    fc = tmp_path / "all_samples_raw_counts.txt"
    _write_featurecounts(fc, counts_symbols)
    a = io.read_counts(fc)
    assert list(a.columns) == list(counts_symbols.columns)
    assert a.index.name == "gene" and a.shape == counts_symbols.shape
    np.testing.assert_array_equal(a.values, counts_symbols.values)

    # extended table: Gene_Symbol, Gene_IDs_Field, Gene_Summary, coords, counts
    ext = counts_symbols.copy()
    ext.insert(0, "Gene_Summary", "some product")
    ext.insert(0, "Gene_IDs_Field", "b0001,GeneID:1")
    ext.index.name = "Gene_Symbol"
    for c in ("Chr", "Start", "End", "Strand", "Length"):
        ext.insert(2, c, 1)
    ext.to_csv(tmp_path / "ext.tsv", sep="\t")
    b = io.read_counts(tmp_path / "ext.tsv")
    np.testing.assert_array_equal(b.values, counts_symbols.values)

    # plain genes x samples table with b-numbers and GSM_SRR columns
    plain = counts_symbols.copy()
    plain.index.name = "Geneid"
    plain.to_csv(tmp_path / "plain.txt", sep="\t")
    c = io.read_counts(tmp_path / "plain.txt")
    assert c.index.name == "gene"
    np.testing.assert_array_equal(c.values, counts_symbols.values)


def test_genemap_and_clean(feature_table, model_genes):
    gm = GeneMap.from_feature_table(feature_table)
    assert {"rrsA", "rrsB", "ileV", "b9001", "b9002", "b9003"} <= gm.rna_ids
    assert gm.locus_to_symbol["b0003"] == "g003"
    assert "b9101" not in gm.locus_to_symbol and gm.n_ambiguous == 1
    counts = pd.DataFrame(np.ones((5, 2)), index=["b0000", "b0001", "b9001", "ileV", "b9101"], columns=["s1", "s2"])
    X = clean_counts(counts, gm)
    assert list(X.columns) == ["g000", "g001", "b9101"]  # rRNA/tRNA dropped, ambiguous locus kept as-is
    assert X.shape == (2, 3)


def test_cpm_and_zscore():
    X = pd.DataFrame([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]], columns=["a", "b"])
    c = cpm(X)
    assert np.allclose(c.iloc[0].values, [250000, 750000])
    assert np.allclose(c.iloc[1].values, [0, 0])  # empty library handled
    assert np.allclose(c.values, cpm_array(X.values))
    z = ZScore.fit(c.values, rows=np.array([0, 2]))
    assert z.sd.dtype == np.float32 and np.all(z.sd > 0)
    const = ZScore.fit(np.array([[5.0, 1.0], [5.0, 2.0]]))
    assert const.sd[0] == 1.0  # zero variance -> sd 1


def test_align_to_genes(counts_symbols, model_genes):
    X = counts_symbols.T
    X = X.drop(columns=["g000", "g001"])
    aligned, cov = align_to_genes(X, model_genes)
    assert list(aligned.columns) == model_genes
    assert cov.n_present == len(model_genes) - 2 and cov.missing == ["g000", "g001"]
    assert aligned["g000"].sum() == 0 and "novelGene" not in aligned.columns
    assert 0.9 < cov.fraction < 1


def test_prepare_from_ingest_results(tmp_path, counts_symbols):
    res = tmp_path / "exp_results"
    res.mkdir()
    _write_featurecounts(res / "all_samples_raw_counts.txt", counts_symbols)
    meta = pd.DataFrame({"sample_id": counts_symbols.columns, "condition": "wt"}).set_index("sample_id")
    meta.to_csv(tmp_path / "meta.csv")
    counts, samples = prepare.prepare(res, None, tmp_path / "meta.csv", tmp_path / "out")
    assert (tmp_path / "out" / "counts.tsv").exists() and (tmp_path / "out" / "samples.tsv").exists()
    back = io.read_counts(tmp_path / "out" / "counts.tsv")
    np.testing.assert_array_equal(back.values, counts_symbols.values)
    assert samples.loc["S0", "source_column"] == "exp_results/S0/vg/S0_sort.bam"
    assert samples.loc["S0", "condition"] == "wt"
    with pytest.raises(ValueError):
        prepare.prepare(None, None, None, tmp_path / "out2")
