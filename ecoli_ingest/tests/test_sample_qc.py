"""Per-sample QC record: parsers and derived values (no tools needed)."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "ecoli_ingest" / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sample_qc         # noqa: E402
import species_metrics   # noqa: E402


def test_strand_expressions():
    plus, minus = sample_qc.strand_expressions("forward", "SE")
    assert (plus, minus) == ("!flag.reverse", "flag.reverse")
    assert sample_qc.strand_expressions("reverse", "SE") == (minus, plus)        # reverse library: swapped
    p_pe, m_pe = sample_qc.strand_expressions("forward", "PE")
    assert "flag.read1 && !flag.reverse" in p_pe and "flag.read2 && flag.reverse" in p_pe
    assert sample_qc.strand_expressions("reverse", "PE") == (m_pe, p_pe)
    assert sample_qc.strand_expressions("unstranded", "PE") == (None, None)


FASTQC = """##FastQC\t0.12.1
>>Basic Statistics\tpass
#Measure\tValue
Total Sequences\t1000
>>END_MODULE
>>Sequence Length Distribution\twarn
#Length\tCount
15\t100.0
16\t300.0
40-44\t600.0
>>END_MODULE
>>Sequence Duplication Levels\tfail
#Total Deduplicated Percentage\t38.5
#Duplication Level\tPercentage of total
1\t20.0
>>END_MODULE
>>Overrepresented sequences\twarn
#Sequence\tCount\tPercentage\tPossible Source
ACGTACGT\t50\t5.0\tNo Hit
TTTTGGGG\t20\t2.0\tTruSeq Adapter
>>END_MODULE
"""


def test_read_fastqc_and_length_quantiles(tmp_path):
    z = tmp_path / "s_clean_R1_fastqc.zip"
    with zipfile.ZipFile(z, "w") as fh:
        fh.writestr("s_clean_R1_fastqc/fastqc_data.txt", FASTQC)
    fq = sample_qc.read_fastqc(str(z))
    assert fq["length_hist"] == [[15, 100], [16, 300], ["40-44", 600]]          # ranges kept as FastQC writes them
    assert fq["deduplicated_pct"] == 38.5
    assert fq["overrepresented"][0] == ["ACGTACGT", 50, 5.0, "No Hit"] and len(fq["overrepresented"]) == 2
    q = sample_qc.length_quantiles(fq["length_hist"])
    assert (q["q01"], q["q05"], q["q25"], q["q50"], q["q99"]) == (15, 15, 16, 40, 40)


VG_STATS = """Total alignments: 1000
Total primary: 1000
Total secondary: 0
Total aligned: 990
Total perfect: 700
Total gapless (softclips allowed): 980
Total paired: 1000
Total properly paired: 970
Alignment score: mean 46.5, median 47, stdev 3.5, max 48 (400 reads)
Mapping quality: mean 52.1, median 60, stdev 16.2, max 60 (600 reads)
Insertions: 10 bp in 9 read events
Deletions: 12 bp in 8 read events
Substitutions: 150 bp in 150 read events
Softclips: 200 bp in 30 read events
Total time: 10.5 seconds
Speed: 95.2 reads/second
Total time: 123 seconds
Speed: 123 reads/second
"""


def test_read_vg_stats_ignores_placeholder_lines(tmp_path):
    p = tmp_path / "s_giraffe.stats.txt"
    p.write_text(VG_STATS)
    v = sample_qc.read_vg_stats(str(p))
    assert (v["total"], v["aligned"], v["perfect"], v["gapless"], v["properly_paired"]) == (1000, 990, 700, 980, 970)
    assert (v["score_mean"], v["score_median"], v["mapq_mean"], v["mapq_stdev"]) == (46.5, 47.0, 52.1, 16.2)
    assert (v["substitution_bp"], v["insertion_bp"], v["deletion_events"], v["softclip_bp"]) == (150, 10, 8, 200)
    assert (v["time_seconds"], v["reads_per_second"]) == (10.5, 95.2)            # the real lines, not 123


def test_gene_body_slope(tmp_path):
    p = tmp_path / "gb.txt"
    p.write_text("Percentile\t" + "\t".join(map(str, range(1, 101))) + "\ns_sort\t" + "\t".join(["5.0"] * 100) + "\n")
    flat = sample_qc.read_gene_body(str(p))
    assert len(flat) == 100 and sample_qc.gene_body_slope(flat) == 0.0
    falling = [200.0 - i for i in range(100)]                                   # more coverage at the 5' end
    assert sample_qc.gene_body_slope(falling) < -0.5
    assert sample_qc.gene_body_slope([]) is None


GFF = "##gff-version 3\n" + "".join(
    f"NC_1\t{src}\tgene\t1\t{length}\t.\t+\t.\tID={fid};feature_type={ftype};source={source};symbol={sym}\n"
    for fid, ftype, source, src, sym, length in [
        ("CDS_b0002_thrA", "CDS", "NCBI", "RefSeq", "thrA", 1000), ("CDS_b0003_thrB", "CDS", "NCBI", "RefSeq", "thrB", 500),
        ("rRNA_b0201_rrsH", "rRNA", "NCBI", "RefSeq", "rrsH", 1500), ("rRNA_b0204_rrlH", "rRNA", "NCBI", "RefSeq", "rrlH", 2900),
        ("rRNA_b0205_rrfH", "rRNA", "NCBI", "RefSeq", "rrfH", 120), ("tRNA_b0202_ileV", "tRNA", "NCBI", "RefSeq", "ileV", 77),
        ("ncRNA_b2621_ssrA", "ncRNA", "NCBI", "RefSeq", "ssrA", 363), ("ncRNA_b3123_rnpB", "ncRNA", "NCBI", "RefSeq", "rnpB", 377),
        ("ncRNA_Thr_leader", "ncRNA", "Rfam", "Rfam", "Thr_leader", 120), ("pseudo_b0240_crl", "pseudo", "NCBI", "RefSeq", "crl", 300),
    ])


def test_composition_by_type_and_family(tmp_path):
    (tmp_path / "a.gff").write_text(GFF)
    features = sample_qc.read_feature_types(str(tmp_path / "a.gff"))
    assert features["ncRNA_Thr_leader"] == ("Rfam", "Thr_leader") and features["rRNA_b0201_rrsH"] == ("rRNA", "rrsH")
    sense = {"CDS_b0002_thrA": (100, 1000), "CDS_b0003_thrB": (50, 500), "rRNA_b0201_rrsH": (4000, 1500),
             "rRNA_b0204_rrlH": (6000, 2900), "rRNA_b0205_rrfH": (30, 120), "tRNA_b0202_ileV": (70, 77),
             "ncRNA_b2621_ssrA": (300, 363), "ncRNA_b3123_rnpB": (40, 377), "ncRNA_Thr_leader": (9, 120),
             "pseudo_b0240_crl": (1, 300)}
    comp = sample_qc.composition(sense, features)
    assert comp["by_type"] == {"CDS": 150, "rRNA": 10030, "tRNA": 70, "ncRNA": 340, "Rfam": 9, "pseudo": 1}
    assert comp["by_family"] == {"16S": 4000, "23S": 6000, "5S": 30, "ssrA": 300, "rnpB": 40, "ssrS": 0, "ffs": 0, "tRNA": 70}
    # features under 200 nt outside rRNA / tRNA: the Rfam leader (120 nt); the tRNA and 5S genes do not count
    assert comp["short_features"] == {"max_length": 200, "reads": 9}


def test_dna_estimates_recover_floor_and_leakage():
    import random
    rng = random.Random(3)
    features, sense, anti = {}, {}, {}
    floor, leak = 0.002, 0.01                    # antisense reads per base = floor + leak * sense reads per base
    for i in range(400):
        fid, length = f"CDS_g{i}", rng.randrange(300, 3000)
        features[fid] = ("CDS", f"g{i}")
        s = int(length * rng.uniform(0, 5))
        sense[fid] = (s, length)
        anti[fid] = (round(length * floor + leak * s), length)
    for i in range(5):                           # a real antisense RNA; and rRNA, which is left out
        anti[f"CDS_g{i}"] = (50_000, sense[f"CDS_g{i}"][1])
    features["rRNA_x"], sense["rRNA_x"], anti["rRNA_x"] = ("rRNA", "rrsA"), (10 ** 6, 1500), (10 ** 5, 1500)
    assigned = sum(v[0] for v in sense.values())
    d = sample_qc.dna_estimates(sense, anti, features, {"Assigned": assigned, "Unassigned_NoFeatures": assigned // 9},
                                {"Assigned": sum(v[0] for v in anti.values())}, 4_641_652)
    assert d["n_genes_fit"] == 400
    assert d["floor_a"] == pytest.approx(floor, rel=0.25) and d["leakage_b"] == pytest.approx(leak, rel=0.15)
    assert d["intergenic_share"] == pytest.approx(0.1, abs=0.001)
    assert d["dna_fraction_est"] == pytest.approx(2 * d["floor_a"] * 4_641_652 / d["aligned_units"])
    assert 0 < d["antisense_share"] < 1


def test_hashed_ids_and_species_metrics_skip(tmp_path):
    ids = species_metrics.HashedIds(f"SRR1.{i}" for i in range(0, 20000, 2))
    assert len(ids) == 10000 and "SRR1.4" in ids and "SRR1.5" not in ids and "other" not in ids
    (tmp_path / "all.txt").write_text(
        "40.00\t80\t0\t-\t1\troot\n35.00\t70\t0\tF\t543\t  Enterobacteriaceae\n 2.00\t4\t0\tG\t561\t    Escherichia\n"
        " 1.00\t2\t2\tS\t562\t      Escherichia coli\n 1.00\t2\t0\tG\t590\t    Salmonella\n 1.00\t2\t2\tS\t28901\t      Salmonella enterica\n")
    rows = ["readID\tseqID\ttaxID\tscore\t2ndBestScore\thitLength\tqueryLength\tnumMatches"]
    rows += [f"SRR1.{i}\ts\t562\t100\t0\t50\t50\t1" for i in (1, 3, 5)] + [f"SRR1.{i}\ts\t28901\t100\t0\t50\t50\t1" for i in (2, 4)]
    (tmp_path / "cf.tsv").write_text("\n".join(rows) + "\n")
    cf, rep = str(tmp_path / "cf.tsv"), str(tmp_path / "all.txt")
    assert species_metrics.species_metrics(cf, rep)["pct_escherichia_shigella"] == 60.0
    m = species_metrics.species_metrics(cf, rep, ids, "_nostable")              # the even ids are on rRNA / tRNA genes
    assert m == {"species_reads_nostable": 3, "pct_escherichia_shigella_nostable": 100.0, "top_other_genus_nostable": "",
                 "pct_top_other_genus_nostable": 0.0}
    assert species_metrics.n_classified_input(cf) == 5


def test_sample_column_and_summary(tmp_path):
    counts = tmp_path / "counts.txt"
    counts.write_text("# Program:featureCounts\nGeneid\tChr\tStart\tEnd\tStrand\tLength\tresults/A_rep1/vg/A_rep1_sort.bam\t"
                      "results/A_rep10/vg/A_rep10_sort.bam\nCDS_x\tNC_1\t1\t100\t+\t100\t7\t70\n")
    assert sample_qc.read_count_column(str(counts), "A_rep1") == {"CDS_x": (7, 100)}       # not confused with A_rep10
    assert sample_qc.read_count_column(str(counts), "A_rep10") == {"CDS_x": (70, 100)}
    summary = tmp_path / "counts.txt.summary"
    summary.write_text("Status\tresults/A_rep1/vg/A_rep1_sort.bam\tresults/A_rep10/vg/A_rep10_sort.bam\n"
                       "Assigned\t900\t9000\nUnassigned_NoFeatures\t100\t1000\n")
    assert sample_qc.read_summary_column(str(summary), "A_rep1") == {"Assigned": 900, "Unassigned_NoFeatures": 100}
    with pytest.raises(SystemExit):
        sample_qc.read_count_column(str(counts), "missing")
