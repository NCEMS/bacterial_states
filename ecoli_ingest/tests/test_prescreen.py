"""Prescreen: read subsample, metric parsers, sample selection and the QC table (no tools or network needed)."""
import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "atlas_build" / "prescreen"))
sys.path.insert(0, str(ROOT / "src" / "ecoli_ingest" / "workflow" / "scripts"))
import prescreen      # noqa: E402
import qc_metrics     # noqa: E402
import qc_table       # noqa: E402
import sra_reads      # noqa: E402


def fastq(n, tag="r", length=30):
    return "".join(f"@{tag}.{i} x/1\n{'ACGT' * 10}"[: len(f"@{tag}.{i} x/1\n") + length] + f"\n+\n{'F' * length}\n"
                   for i in range(1, n + 1))


# ------------------------------------------------------------------ read subsample

def test_stream_head_takes_first_reads_and_handles_gzip_members(tmp_path):
    src = tmp_path / "a.fastq.gz"
    with open(src, "wb") as fh:                      # two gzip members, as pigz/bgzip can write
        fh.write(gzip.compress(fastq(70).encode()))
        fh.write(gzip.compress(fastq(50, "s").encode()))
    out = tmp_path / "a.fastq"
    assert sra_reads.stream_head(src.as_uri(), str(out), 100) == 100
    lines = out.read_text().splitlines()
    assert len(lines) == 400 and lines[0] == "@r.1 x/1" and lines[-4] == "@s.30 x/1"
    assert sra_reads.stream_head(src.as_uri(), str(out), 1000) == 120      # fewer reads than asked for


def test_fetch_subsample_from_ena_files_equalizes_mates(tmp_path):
    for mate, n in (("1", 40), ("2", 25)):
        with gzip.open(tmp_path / f"R_{mate}.fastq.gz", "wt") as fh:
            fh.write(fastq(n))
    ena = {m: {"url": (tmp_path / f"R_{m}.fastq.gz").as_uri()} for m in ("1", "2")}
    source, counts = sra_reads.fetch_subsample("SRR1", str(tmp_path / "o"), ("1", "2"), 30, 0, "ena", ena)
    assert source == "ena" and counts == {"1": 25, "2": 25}
    assert len((tmp_path / "o" / "SRR1_1.fastq").read_text().splitlines()) == 100
    with pytest.raises(sra_reads.DownloadError):
        sra_reads.fetch_subsample("SRR1", str(tmp_path / "o"), ("1", "2"), 30, 0, "ena", {"1": ena["1"]})


def test_ena_files_from_row():
    row = {"fastq_ftp": "ftp.sra.ebi.ac.uk/x/SRR9_1.fastq.gz;ftp.sra.ebi.ac.uk/x/SRR9_2.fastq.gz",
           "fastq_bytes": "10;12", "library_layout": "PAIRED", "read_count": "77"}
    rec = sra_reads.ena_files_from_row(row, "SRR9")
    assert rec["layout"] == "PAIRED" and rec["spots"] == 77
    assert rec["files"]["2"] == {"url": "https://ftp.sra.ebi.ac.uk/x/SRR9_2.fastq.gz", "md5": None, "bytes": 12}
    assert sra_reads.ena_files_from_row({"fastq_ftp": "ftp.x/SRR9.fastq.gz"}, "SRR9")["layout"] == "SINGLE"


# ------------------------------------------------------------------ metric parsers

FASTP = {"summary": {"before_filtering": {"total_reads": 400000},
                     "after_filtering": {"total_reads": 372000, "read1_mean_length": 42, "read2_mean_length": 41,
                                         "q30_rate": 0.8573}},
         "duplication": {"rate": 0.0561}, "insert_size": {"peak": 55},
         "adapter_cutting": {"adapter_trimmed_reads": 6000}, "read2_after_filtering": {}}
VG_STATS = ("Total alignments: 1000\nTotal primary: 1000\nTotal aligned: 990\nTotal perfect: 700\n"
            "Total paired: 1000\nTotal properly paired: 980\nMapping quality: mean 39.5, median 60\n")
INFER_PE = ('This is PairEnd Data\nFraction of reads failed to determine: 0.0042\n'
            'Fraction of reads explained by "1++,1--,2+-,2-+": 0.0128\n'
            'Fraction of reads explained by "1+-,1-+,2++,2--": 0.9830\n')
OVERLAP = '# id: "feature_overlap"\n# plot_type: "bargraph"\nCDS\t63.59\nrRNA\t22.57\ntRNA\t1.63\nncRNA\t10.03\nRfam\t1.38\npseudo\t0.81\n'
KREPORT = ("  0.00\t0\t0\tU\t0\tunclassified\n100.00\t1000\t0\t-\t1\troot\n 99.00\t990\t0\tD\t2\t    Bacteria\n"
           " 80.00\t800\t0\tG\t561\t          Escherichia\n 76.69\t767\t767\tS\t562\t            Escherichia coli\n"
           "  1.70\t17\t17\tS\t28901\t            Salmonella enterica\n  0.90\t9\t9\tS\t623\t            Shigella flexneri\n")


def write_qc_files(d, sample=None):
    """The five QC files, laid out as the prescreen (flat) or as an inGEST results directory."""
    d.mkdir(parents=True, exist_ok=True)
    if sample is None:
        files = {"fastp": d / "fastp.json", "vg_stats": d / "vg.txt", "infer_experiment": d / "infer.txt",
                 "feature_overlap": d / "fo.tsv", "kreport": d / "kreport.txt"}
    else:
        files = {k: Path(v) for k, v in qc_table.full_run_files(str(d), sample).items()}
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    files["fastp"].write_text(json.dumps(FASTP))
    files["vg_stats"].write_text(VG_STATS)
    files["infer_experiment"].write_text(INFER_PE)
    files["feature_overlap"].write_text(OVERLAP)
    files["kreport"].write_text(KREPORT)
    return {k: str(v) for k, v in files.items()}


def test_collect_reads_every_metric(tmp_path):
    m = qc_metrics.collect(write_qc_files(tmp_path))
    assert set(m) == set(qc_metrics.METRICS) - {
        "pct_unclassified", "pct_enterobacteriaceae", "pct_synthetic", "top_non_enteric",     # need the all-reads report
        "species_reads", "pct_escherichia_shigella", "top_other_genus", "pct_top_other_genus",
        "species_reads_nostable", "pct_escherichia_shigella_nostable", "top_other_genus_nostable",
        "pct_top_other_genus_nostable", "centrifuge_reads"}
    assert (m["reads_in"], m["reads_kept"], m["pct_kept"]) == (400000, 372000, 93.0)
    assert (m["mean_length_r1"], m["mean_length_r2"], m["pct_adapter_trimmed"], m["insert_size_peak"]) == (42, 41, 1.5, 55)
    assert (m["pct_aligned"], m["pct_perfect"], m["pct_properly_paired"], m["mapq_mean"]) == (99.0, 70.0, 98.0, 39.5)
    assert (m["layout"], m["strand_call"], m["fraction_reverse"]) == ("PE", "reverse", 0.983)
    assert (m["pct_CDS"], m["pct_rRNA"], m["pct_Rfam"]) == (63.59, 22.57, 1.38)
    assert m["pct_ecoli"] == 76.69         # the first report line naming E. coli, as the workflow's gate reads it
    assert m["other_taxa"] == "Salmonella enterica 1.7; Shigella flexneri 0.9"


def test_kreport_all_metrics(tmp_path):
    report = ("60.00\t12000\t12000\tU\t0\tunclassified\n40.00\t8000\t300\t-\t1\troot\n"
              "37.00\t7400\t0\tD\t2\t    Bacteria\n35.00\t7000\t6500\tF\t543\t            Enterobacteriaceae\n"
              " 0.80\t160\t160\tS\t562\t                Escherichia coli\n"
              " 0.50\t100\t100\tS\t622\t                Shigella dysenteriae\n"
              " 1.00\t200\t0\tF\t1903409\t            Erwiniaceae\n"
              " 1.00\t200\t200\tS\t549\t              Pantoea agglomerans\n"
              " 1.50\t300\t300\tS\t32630\t      synthetic construct\n")
    p = tmp_path / "all.txt"
    p.write_text(report)
    m = qc_metrics.kreport_all_metrics(str(p))
    assert m == {"pct_unclassified": 60.0, "pct_enterobacteriaceae": 87.5, "pct_synthetic": 3.75,
                 "top_non_enteric": "Pantoea agglomerans 2.5"}      # Shigella is inside the family, not "other"


def test_species_metrics_and_fastq_selection(tmp_path):
    report = ("40.00\t8000\t300\t-\t1\troot\n35.00\t7000\t6500\tF\t543\t      Enterobacteriaceae\n"
              " 2.00\t400\t0\tG\t561\t        Escherichia\n 1.00\t200\t200\tS\t562\t          Escherichia coli\n"
              " 1.00\t200\t0\tG\t620\t        Shigella\n 1.00\t200\t200\tS\t622\t          Shigella dysenteriae\n"
              " 1.00\t200\t0\tG\t590\t        Salmonella\n 1.00\t200\t200\tS\t28901\t          Salmonella enterica\n"
              " 1.50\t300\t300\tS\t32630\t  synthetic construct\n")
    (tmp_path / "all.txt").write_text(report)
    rows = ["readID\tseqID\ttaxID\tscore\t2ndBestScore\thitLength\tqueryLength\tnumMatches"]
    rows += [f"r{i}\ts\t562\t100\t0\t50\t50\t1" for i in range(6)]         # E. coli
    rows += [f"s{i}\ts\t622\t100\t0\t50\t50\t1" for i in range(2)]         # Shigella counts as E. coli
    rows += [f"t{i}\ts\t28901\t100\t0\t50\t50\t1" for i in range(2)]       # Salmonella
    rows += ["u1\ts\t543\t100\t0\t50\t50\t1", "u2\ts\t32630\t100\t0\t50\t50\t1",   # family level, synthetic: ignored
             "u3\ts\t562\t100\t100\t50\t50\t2", "u4\tunclassified\t0\t0\t0\t0\t50\t1"]  # two matches, unclassified
    (tmp_path / "cf.tsv").write_text("\n".join(rows) + "\n")
    m = qc_metrics.species_metrics(str(tmp_path / "cf.tsv"), str(tmp_path / "all.txt"))
    assert m == {"species_reads": 10, "pct_escherichia_shigella": 80.0, "top_other_genus": "Salmonella",
                 "pct_top_other_genus": 20.0}
    # without the reads on rRNA / tRNA genes (here two of the Salmonella-like reads and one E. coli read)
    m2 = qc_metrics.species_metrics(str(tmp_path / "cf.tsv"), str(tmp_path / "all.txt"), {"t0", "t1", "r0"}, "_nostable")
    assert m2 == {"species_reads_nostable": 7, "pct_escherichia_shigella_nostable": 100.0,
                  "top_other_genus_nostable": "", "pct_top_other_genus_nostable": 0.0}

    (tmp_path / "in.fastq").write_text(fastq(10))
    n = prescreen.head_fastq(str(tmp_path / "in.fastq"), str(tmp_path / "out.fastq"), 3, {b"r.1", b"r.3"})
    names = [l for l in (tmp_path / "out.fastq").read_text().splitlines() if l.startswith("@")]
    assert n == 3 and names == ["@r.2 x/1", "@r.4 x/1", "@r.5 x/1"]


def test_step_timeout(tmp_path):
    with pytest.raises(prescreen.StepError, match="timed out"):
        prescreen.sh("sleep 30", "slow", str(tmp_path / "log.txt"), timeout=1)


def test_collect_skips_missing_files(tmp_path):
    files = write_qc_files(tmp_path)
    Path(files["kreport"]).unlink()
    m = qc_metrics.collect(files)
    assert "pct_ecoli" not in m and m["strand_call"] == "reverse"


@pytest.mark.parametrize("change,expected", [
    ({}, ("pass", [])),
    ({"fraction_reverse": 0.52, "fraction_forward": 0.47}, ("fail", ["unstranded"])),
    ({"pct_ecoli": 16.0}, ("pass", [])),                                   # the workflow's gate value no longer decides
    ({"pct_escherichia_shigella": 0.2, "pct_top_other_genus": 99.5}, ("fail", ["not_ecoli"])),      # Salmonella
    ({"pct_escherichia_shigella": 45.0, "pct_top_other_genus": 12.0}, ("pass", [])),   # low share, but the other genus is 0.4 % of all reads
    ({"species_reads": 30000, "pct_escherichia_shigella": 60.0, "pct_top_other_genus": 35.0},
     ("fail", ["mixed_organism"])),                                                    # another organism is 21 % of all reads
    ({"species_reads": 500, "pct_escherichia_shigella": 60.0, "pct_top_other_genus": 35.0}, ("pass", [])),   # a trace: 0.35 %
    # a phage as the largest other genus fails the sample even when E. coli is the majority (1.2 % of all reads)
    ({"species_reads": 3000, "pct_escherichia_shigella": 75.0, "pct_top_other_genus": 20.0, "top_other_genus": "T4virus"},
     ("fail", ["phage"])),
    ({"pct_escherichia_shigella": 1.0, "pct_top_other_genus": 98.0, "top_other_genus": "T7virus"}, ("fail", ["phage"])),
    ({"species_reads": 3000, "pct_escherichia_shigella": 75.0, "pct_top_other_genus": 20.0,
      "top_other_genus": "Phix174microvirus"}, ("pass", [])),                           # the PhiX spike-in is not a phage sample
    ({"species_reads": 3000, "pct_escherichia_shigella": 97.0, "pct_top_other_genus": 2.0, "top_other_genus": "Lambdavirus"},
     ("pass", [])),                                                                    # a trace (0.12 %)
    # the version without rRNA / tRNA reads is used when present
    ({"pct_escherichia_shigella": 35.0, "pct_top_other_genus": 60.0, "species_reads_nostable": 1500,
      "pct_escherichia_shigella_nostable": 85.0, "pct_top_other_genus_nostable": 2.0}, ("pass", [])),
    ({"pct_kept": 60, "mean_length_r1": 26, "pct_aligned": 17}, ("fail", ["trimming", "short_reads", "low_alignment"])),
    ({"species_reads": 40}, ("incomplete", ["no_species"])),
    ({"pct_escherichia_shigella": ""}, ("incomplete", ["no_species"])),
    ({"species_reads": 40, "pct_aligned": 10}, ("fail", ["low_alignment"])),
])
def test_decide(change, expected):
    base = {"fraction_forward": 0.01, "fraction_reverse": 0.98, "pct_ecoli": 77, "pct_kept": 93,
            "mean_length_r1": 42, "pct_aligned": 99, "species_reads": 1800, "pct_escherichia_shigella": 92.0,
            "pct_top_other_genus": 0.9, "centrifuge_reads": 50000}
    assert qc_metrics.decide({**base, **change}) == expected


def test_decide_thresholds_can_be_changed():
    m = {"fraction_forward": 0.75, "fraction_reverse": 0.2, "pct_kept": 93, "mean_length_r1": 42, "pct_aligned": 99,
         "species_reads": 20000, "pct_escherichia_shigella": 45.0, "pct_top_other_genus": 30.0, "centrifuge_reads": 50000}
    assert qc_metrics.decide(m) == ("fail", ["unstranded", "not_ecoli"])
    assert qc_metrics.decide(m, {"min_strand_fraction": 0.7, "min_pct_escherichia_shigella": 40}) == (
        "fail", ["mixed_organism"])                    # the other genus is still 12 % of all reads
    assert qc_metrics.decide(m, {"min_strand_fraction": 0.7, "min_pct_escherichia_shigella": 40,
                                 "max_pct_foreign_of_all": 20}) == ("pass", [])


# ------------------------------------------------------------------ sample selection

def test_select_samples_shards_and_skips_done(tmp_path):
    samples = [{"GSM": f"GSM{i}", "GSE": "GSE1" if i < 30 else "GSE2", "status": "candidate" if i % 5 else "low_reads"}
               for i in range(60)]
    (tmp_path / "samples").mkdir()
    everything = prescreen.select_samples(samples, str(tmp_path))
    assert len(everything) == 48
    parts = [prescreen.select_samples(samples, str(tmp_path), shard=(i, 4)) for i in range(1, 5)]
    assert sorted(s["GSM"] for p in parts for s in p) == sorted(s["GSM"] for s in everything)   # each sample once
    assert all(parts)
    (tmp_path / "samples" / "GSM1.json").write_text(json.dumps({"GSM": "GSM1", "metrics": {}}))
    (tmp_path / "samples" / "GSM2.json").write_text(json.dumps({"GSM": "GSM2", "error": "fetch: x"}))
    assert len(prescreen.select_samples(samples, str(tmp_path))) == 46
    assert [s["GSM"] for s in prescreen.select_samples(samples, str(tmp_path), gsm={"GSM1", "GSM2"}, redo_errors=True)] == ["GSM2"]
    assert len(prescreen.select_samples(samples, str(tmp_path), redo=True)) == 48
    assert len(prescreen.select_samples(samples, str(tmp_path), gse={"GSE2"}, limit=5)) == 5
    assert len(prescreen.select_samples(samples, str(tmp_path), status="any", redo=True)) == 60


# ------------------------------------------------------------------ QC table

def test_qc_table_prescreen_then_full_run(tmp_path):
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    with open(manifest / "samples.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["GSM", "GSE", "status", "runs", "layout", "scientific_name", "tax_id", "instrument_model",
                    "review", "in_atlas_v1"])
        w.writerow(["GSM1", "GSE9", "candidate", "SRR1", "PAIRED", "Escherichia coli", "562", "NovaSeq", "", "yes"])
        for i in range(qc_table.LARGE_STUDY_MIN):
            w.writerow([f"GSM9{i:04d}", "GSE8", "candidate", "SRR9", "PAIRED", "Escherichia coli", "562", "NovaSeq", "", "no"])
        w.writerow(["GSM2", "GSE9", "candidate", "SRR2,SRR3", "PAIRED", "Escherichia coli", "562", "NovaSeq", "", "no"])
        w.writerow(["GSM3", "GSE9", "low_reads", "SRR4", "SINGLE", "Escherichia coli", "562", "NovaSeq", "", "no"])
    recs = tmp_path / "samples"
    recs.mkdir()
    metrics = qc_metrics.collect(write_qc_files(tmp_path / "pre"))
    metrics.update(species_reads=1500, pct_escherichia_shigella=95.0, top_other_genus="Salmonella", pct_top_other_genus=1.0,
                   centrifuge_reads=50000)
    (recs / "GSM1.json").write_text(json.dumps({"GSM": "GSM1", "GSE": "GSE9", "run": "SRR1", "spots": 200000,
                                               "source": "ena", "date": "2026-11-15", "pipeline": "abc1234",
                                               "metrics": metrics}))
    (recs / "GSM2.json").write_text(json.dumps({"GSM": "GSM2", "GSE": "GSE9", "run": "SRR2", "date": "2026-11-15",
                                               "error": "fetch: no reads"}))
    table = str(tmp_path / "qc.tsv")
    rows = qc_table.merge_prescreen(table, str(manifest), str(recs))
    assert len(rows) == 3 + qc_table.LARGE_STUDY_MIN
    assert (rows["GSM1"]["study_candidates"], rows["GSM1"]["large_study"]) == ("2", "no")       # GSM3 is not a candidate
    assert (rows["GSM90000"]["study_candidates"], rows["GSM90000"]["large_study"]) == (str(qc_table.LARGE_STUDY_MIN), "yes")
    assert rows["GSM1"]["pre_decision"] == "pass" and rows["GSM1"]["pre_strand_call"] == "reverse"
    assert rows["GSM1"]["in_atlas_v1"] == "yes" and rows["GSM1"]["pre_run"] == "SRR1"
    assert rows["GSM2"]["pre_decision"] == "error" and rows["GSM2"]["pre_error"] == "fetch: no reads"
    assert rows["GSM3"]["manifest_status"] == "low_reads" and rows["GSM3"]["pre_decision"] == ""

    # a finished inGEST run: sample names map to GSMs through the run's config
    results = tmp_path / "run" / "results"
    write_qc_files(results, "wt_rep1")
    (results / "strandedness.tsv").write_text("sample\tcall\tcounted\nwt_rep1\treverse\tyes\n")
    config = tmp_path / "run" / "config.yaml"
    config.write_text('samples:\n  wt_rep1: {accession: "GSM1", condition: "wt"}\n')
    rows = qc_table.add_full(table, str(results), str(config), "def5678")
    r = rows["GSM1"]
    assert (r["full_sample"], r["full_counted"], r["full_pipeline"]) == ("wt_rep1", "yes", "def5678")
    assert r["full_pct_aligned"] == r["pre_pct_aligned"] == "99.0"      # same parser for both
    assert r["pre_decision"] == "pass" and rows["GSM2"]["full_date"] == ""   # other columns untouched

    # stricter thresholds change the decision without re-running anything
    assert qc_table.main(["decide", "--table", table, "--min-pct-aligned", "99.5"]) == 0
    assert qc_table.read_table(table)["GSM1"]["pre_reasons"] == "low_alignment"
    (tmp_path / "model.txt").write_text("GSM1_SRR1\nGSM99\n")
    assert qc_table.main(["mark-model", "--table", table, "--samples", str(tmp_path / "model.txt"), "--model", "v2"]) == 0
    back = qc_table.read_table(table)
    assert back["GSM1"]["in_model"] == "v2" and back["GSM2"]["in_model"] == ""
    with open(table, newline="") as fh:
        assert next(csv.reader(fh, delimiter="\t")) == qc_table.COLUMNS
