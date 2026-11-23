"""Joining the count tables of the atlas recount batches, and the QC table steps that follow it."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "atlas_build" / "prescreen"))
import qc_table      # noqa: E402


@pytest.fixture(scope="module")
def cr():
    spec = importlib.util.spec_from_file_location("collect_results", ROOT / "atlas_build" / "full_run" / "collect_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FEATURES = [("CDS_b0001_thrL", 66), ("rRNA_b0201_rrsH", 1542), ("ncRNA_RF00001", 120)]


def write_batch(root, batch, counted, failed=None, not_counted=None, features=FEATURES, cds_reads=300000):
    """A batch directory as slurm_batch.sh leaves it. counted: {sample: [count per feature]}."""
    res = root / batch / "results"
    res.mkdir(parents=True)
    cols = [f"results/{s}/vg/{s}_sort.bam" for s in counted]
    for name, factor in (("all_samples_raw_counts.txt", 1), ("all_samples_antisense_counts.txt", 0)):
        lines = ['# Program:featureCounts v2.1.1; Command:"featureCounts"', "\t".join(["Geneid", "Chr", "Start", "End", "Strand", "Length"] + cols)]
        for i, (fid, length) in enumerate(features):
            lines.append("\t".join([fid, "NC_000913.3", "1", str(length), "+", str(length)] + [str(v[i] * factor) for v in counted.values()]))
        (res / name).write_text("\n".join(lines) + "\n")
    (res / "all_samples_raw_counts.txt.summary").write_text(
        "\t".join(["Status"] + cols) + "\n" + "\t".join(["Assigned"] + [str(sum(v) - 1) for v in counted.values()]) + "\n")
    with open(res / "strandedness.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["sample", "layout", "fraction_forward", "fraction_reverse", "fraction_undetermined", "call", "call_source",
                    "counted", "not_counted_reason"])
        for s in counted:
            w.writerow([s, "PE", "0.02", "0.97", "0.01", "reverse", "infer_experiment", "yes", ""])
        for s, reason in (not_counted or {}).items():
            w.writerow([s, "PE", "0.02", "0.97", "0.01", "reverse", "infer_experiment", "no", reason])
    (root / batch / "qc_failures.json").write_text(json.dumps(failed or {}))
    (root / batch / "config.yaml").write_text("samples:\n" + "".join(
        f'  {s}:\n    accession: "{s}"\n' for s in list(counted) + list(failed or {}) + list(not_counted or {})))
    for s in list(counted) + list(failed or {}) + list(not_counted or {}):
        (res / s / "fastp").mkdir(parents=True)
        (res / s / "fastp" / f"{s}_fastp.json").write_text(json.dumps(
            {"summary": {"before_filtering": {"total_reads": 1000}, "after_filtering": {"total_reads": 900, "total_bases": 90000}}}))
    for s in counted:
        (res / s / "qc").mkdir()
        (res / s / "qc" / f"{s}.qc.json").write_text(json.dumps(
            {"composition": {"by_type": {"CDS": cds_reads if not isinstance(cds_reads, dict) else cds_reads[s], "rRNA": 10}},
             "strand": {"layout": "PE", "fraction_forward": 0.02, "fraction_reverse": 0.97},
             "species": {"verdict": "ok"}, "pipeline": {"commit": "abc1234"}}))


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_collect_joins_batches_and_prefers_the_batch_that_counted(tmp_path, cr):
    root = tmp_path / "results"
    write_batch(root, "PE_001", {"GSM1": [5, 100, 2], "GSM2": [7, 50, 0]},
                failed={"GSM3": {"reason": "fastp", "kept_percentage": 72.0, "threshold": 75.0},
                        "GSM4": {"reason": "species", "verdict": "phage"}}, not_counted={"GSM5": "not_ecoli"})
    write_batch(root, "SE_001", {"GSM6": [1, 1, 1]})
    write_batch(root, "PE_R01", {"GSM3": [9, 9, 9]})
    with open(tmp_path / "batches.tsv", "w") as fh:
        fh.write("batch\tn_samples\nPE_001\t5\nSE_001\t1\n")
    prefix = str(tmp_path / "out" / "atlas")
    assert cr.main(["--results", str(root), "--batches", str(tmp_path / "batches.tsv"), "--also", "PE_R01", "--out-prefix", prefix]) == 0
    counts = read_tsv(prefix + "_counts.tsv")
    assert list(counts[0]) == ["Feature_ID", "GSM1", "GSM2", "GSM3", "GSM6"]
    assert [r["Feature_ID"] for r in counts] == [f for f, _ in FEATURES]
    assert [counts[0][s] for s in ("GSM1", "GSM2", "GSM3", "GSM6")] == ["5", "7", "9", "1"]
    assert {r["GSM3"] for r in read_tsv(prefix + "_antisense_counts.tsv")} == {"0"}
    samples = {(r["GSM"], r["batch"]): r for r in read_tsv(prefix + "_samples.tsv")}
    assert len(samples) == 7
    assert samples[("GSM3", "PE_001")]["not_counted_reason"] == "fastp" and samples[("GSM3", "PE_001")]["superseded_by"] == "PE_R01"
    assert samples[("GSM3", "PE_R01")]["counted"] == "yes" and samples[("GSM3", "PE_R01")]["superseded_by"] == ""
    assert samples[("GSM4", "PE_001")]["not_counted_reason"] == "phage"
    assert samples[("GSM5", "PE_001")]["not_counted_reason"] == "not_ecoli"
    assert (samples[("GSM1", "PE_001")]["assignments"], samples[("GSM1", "PE_001")]["reads_assigned"]) == ("107", "106")


def test_collect_stops_when_batches_have_different_features(tmp_path, cr):
    root = tmp_path / "results"
    write_batch(root, "PE_001", {"GSM1": [5, 100, 2]})
    write_batch(root, "PE_002", {"GSM2": [5, 100, 2]}, features=FEATURES[:2] + [("ncRNA_RF00002", 120)])
    with pytest.raises(SystemExit, match="features differ"):
        cr.main(["--results", str(root), "--out-prefix", str(tmp_path / "atlas")])


def test_add_full_batches_and_model_gate(tmp_path):
    root = tmp_path / "results"
    write_batch(root, "PE_001", {"GSM1": [5, 100, 2], "GSM2": [7, 50, 0]}, cds_reads={"GSM1": 250000, "GSM2": 150000},
                failed={"GSM3": {"reason": "fastp", "kept_percentage": 72.0, "threshold": 75.0}}, not_counted={"GSM5": "not_ecoli"})
    write_batch(root, "PE_R01", {"GSM3": [9, 9, 9]})
    table = str(tmp_path / "qc.tsv")
    for _ in range(2):        # a second pass in any order keeps the counted result of the re-run sample
        assert qc_table.main(["add-full", "--batches", str(root), "--table", table]) == 0
    rows = qc_table.read_table(table)
    assert (rows["GSM1"]["full_counted"], rows["GSM1"]["full_reads_CDS"], rows["GSM1"]["full_species_verdict"]) == ("yes", "250000", "ok")
    assert (rows["GSM3"]["full_counted"], rows["GSM3"]["full_not_counted_reason"]) == ("yes", "")
    assert rows["GSM3"]["full_results_dir"].endswith("PE_R01/results")
    assert (rows["GSM5"]["full_counted"], rows["GSM5"]["full_not_counted_reason"]) == ("no", "not_ecoli")
    qc_table.add_full(table, str(root / "PE_001" / "results"), str(root / "PE_001" / "config.yaml"))
    assert qc_table.read_table(table)["GSM3"]["full_counted"] == "yes"

    meta = tmp_path / "metadata.tsv"
    meta.write_text("GSM\tusable\nGSM1\ttrue\nGSM2\ttrue\nGSM3\ttrue\nGSM5\ttrue\nGSM7\tfalse\n")
    assert qc_table.main(["mark-model", "--metadata", str(meta), "--min-cds-reads", "200000", "--model", "atlas_v2", "--table", table]) == 0
    rows = qc_table.read_table(table)
    assert {g: rows[g]["in_model"] for g in ("GSM1", "GSM2", "GSM3", "GSM5")} == {"GSM1": "atlas_v2", "GSM2": "", "GSM3": "atlas_v2", "GSM5": ""}
    assert qc_table.model_gate(rows["GSM2"], True, 200000) == "low_mrna"
    assert qc_table.model_gate(rows["GSM5"], True, 200000) == "not_counted:not_ecoli"
    assert qc_table.model_gate(rows["GSM1"], False, 200000) == "not_usable"
    # a lower threshold takes the shallow sample in; a higher one takes samples out again
    qc_table.main(["mark-model", "--metadata", str(meta), "--min-cds-reads", "100000", "--model", "atlas_v2", "--table", table])
    assert qc_table.read_table(table)["GSM2"]["in_model"] == "atlas_v2"
    qc_table.main(["mark-model", "--metadata", str(meta), "--min-cds-reads", "280000", "--model", "atlas_v2", "--table", table])
    rows = qc_table.read_table(table)
    assert (rows["GSM1"]["in_model"], rows["GSM3"]["in_model"]) == ("", "atlas_v2")


def test_model_gate_leaves_out_samples_with_much_genomic_dna():
    row = {"full_date": "2026-11-21", "full_counted": "yes", "full_not_counted_reason": "", "full_reads_CDS": "500000", "full_dna_fraction_est": "0.12",
           "full_fraction_forward": "0.02", "full_fraction_reverse": "0.97"}
    assert qc_table.model_gate(row, True, 200000) == ""                           # no DNA rule given
    assert qc_table.model_gate(row, True, 200000, 0.10) == "high_dna"
    assert qc_table.model_gate(row | {"full_dna_fraction_est": "0.10"}, True, 200000, 0.10) == ""
    assert qc_table.model_gate(row | {"full_dna_fraction_est": ""}, True, 200000, 0.10) == ""     # no estimate: passes
    assert qc_table.model_gate(row | {"full_reads_CDS": "1000"}, True, 200000, 0.10) == "low_mrna"


def test_add_full_takes_the_gc_bias_from_the_record_or_fits_it_from_the_coverage(tmp_path, monkeypatch):
    root = tmp_path / "results"
    write_batch(root, "PE_001", {"GSM1": [5, 100, 2], "GSM2": [7, 50, 0]})
    res = root / "PE_001" / "results"
    rec = json.loads((res / "GSM1" / "qc" / "GSM1.qc.json").read_text())
    rec["gc_bias"] = {"gc_slope": -0.3, "gc_curv": -0.1}                    # a record made with the reference genome
    (res / "GSM1" / "qc" / "GSM1.qc.json").write_text(json.dumps(rec))
    (res / "GSM2" / "coverage").mkdir()
    (res / "GSM2" / "coverage" / "GSM2.cov100.npz").write_bytes(b"")          # an older record: only the track
    (tmp_path / "g.fna").write_text(">NC_000913.3\nACGT\n")
    (tmp_path / "a.bed").write_text("")
    calls = []
    monkeypatch.setattr(qc_table.gc_bias, "fit_files", lambda cov, bed, fasta, genome: calls.append(cov) or {"gc_slope": 0.2, "gc_curv": -0.05})
    table = str(tmp_path / "qc.tsv")
    assert qc_table.main(["add-full", "--batches", str(root), "--table", table]) == 0
    rows = qc_table.read_table(table)
    assert (rows["GSM1"]["full_gc_slope"], rows["GSM2"]["full_gc_slope"]) == ("-0.3", "") and not calls
    assert qc_table.main(["add-full", "--batches", str(root), "--table", table, "--genome", str(tmp_path / "g.fna"), "--bed", str(tmp_path / "a.bed")]) == 0
    rows = qc_table.read_table(table)
    assert (rows["GSM1"]["full_gc_slope"], rows["GSM2"]["full_gc_slope"], rows["GSM2"]["full_gc_curv"]) == ("-0.3", "0.2", "-0.05")
    assert len(calls) == 1 and calls[0].endswith("GSM2.cov100.npz")
    with pytest.raises(SystemExit, match="needs --bed"):
        qc_table.main(["add-full", "--batches", str(root), "--table", table, "--genome", str(tmp_path / "g.fna")])


def test_model_gate_refuses_unstranded_libraries():
    from qc_table import model_gate
    row = {"full_date": "2026-11-01", "full_counted": "yes", "full_not_counted_reason": "", "full_reads_CDS": "500000",
           "full_dna_fraction_est": "0.01", "full_fraction_forward": "0.02", "full_fraction_reverse": "0.97"}
    assert model_gate(row, True, 200000) == ""
    assert model_gate(row | {"full_fraction_forward": "0.50", "full_fraction_reverse": "0.49"}, True, 200000) == "unstranded"
    assert model_gate(row | {"full_fraction_forward": ""}, True, 200000) == "no_strand_fractions"
