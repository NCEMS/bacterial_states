"""Sample metadata v2: applying label files, validation, lumped columns (atlas_build/metadata)."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

META = Path(__file__).resolve().parents[1] / "atlas_build" / "metadata"

SOFT = """^SERIES = GSE1
!Series_title = Test study
!Series_summary = Wild type and rpoS deletion in LB.
!Series_overall_design = Two strains, exponential and stationary phase, with or without phage T4.
^SAMPLE = GSM1
!Sample_title = WT exponential
!Sample_characteristics_ch1 = genotype: wild type
!Sample_characteristics_ch1 = growth phase: exponential
!Sample_growth_protocol_ch1 = Cells were grown in LB at 37 C with shaking.
^SAMPLE = GSM2
!Sample_title = rpoS stationary
!Sample_characteristics_ch1 = genotype: delta rpoS
!Sample_characteristics_ch1 = growth phase: stationary
!Sample_growth_protocol_ch1 = Cells were grown in LB at 37 C with shaking.
^SAMPLE = GSM3
!Sample_title = WT exponential T4
!Sample_characteristics_ch1 = genotype: wild type
!Sample_characteristics_ch1 = growth phase: exponential
!Sample_growth_protocol_ch1 = Cells were grown in LB at 37 C with shaking.
^SAMPLE = GSM4
!Sample_title = not a candidate
!Sample_characteristics_ch1 = genotype: wild type
"""

LABELS = {
    "gse": "GSE1", "notes": "",
    "default": {
        "base_medium": {"v": "LB", "b": "stated", "q": "Cells were grown in LB at 37 C with shaking."},
        "medium_class": {"v": "complex", "b": "inferred", "q": "Cells were grown in LB at 37 C with shaking."},
        "assay_class": {"v": "total_rnaseq", "b": "study_default", "q": "Wild type and rpoS deletion in LB."},
        "treatment": {"v": "none", "b": "study_default", "q": "Two strains, exponential and stationary phase"},
    },
    "rules": [
        {"field": "ch:growth phase", "regex": "^exponential$", "set": {"growth_phase": {"v": "mid_exponential", "b": "stated", "q": ""}}},
        {"field": "ch:growth phase", "regex": "^stationary$", "set": {"growth_phase": {"v": "stationary", "b": "stated", "q": ""}}},
        {"field": "ch:genotype", "regex": "^wild type$", "set": {"genotype_class": {"v": "wild_type", "b": "stated", "q": ""}}},
        {"field": "ch:genotype", "regex": "rpoS", "set": {"genotype_class": {"v": "deletion", "b": "stated", "q": ""}}},
        {"field": "title", "regex": "T4$", "set": {"treatment": {"v": "phage", "b": "stated", "q": ""},
                                                   "mixed_organism": {"v": True, "b": "inferred", "q": ""}}},
    ],
    "samples": {},
}


def run(tmp_path, labels, qc_rows=None):
    soft, dig, lab = tmp_path / "soft", tmp_path / "digest", tmp_path / "labels"
    for d in (soft, lab):
        d.mkdir(exist_ok=True)
    (soft / "GSE1.soft").write_text(SOFT)
    samples = tmp_path / "samples.tsv"
    samples.write_text("GSM\tstatus\n" + "".join(f"GSM{i}\t{s}\n" for i, s in ((1, "candidate"), (2, "candidate"), (3, "candidate"), (4, "library_type"))))
    subprocess.run([sys.executable, str(META / "geo_digest.py"), str(soft), str(dig), str(samples)], check=True, capture_output=True)
    (lab / "GSE1.json").write_text(json.dumps(labels))
    cmd = [sys.executable, str(META / "apply_labels.py"), str(soft), str(samples), str(tmp_path / "out")]
    if qc_rows:
        qc = tmp_path / "qc.tsv"
        qc.write_text("GSM\tpre_decision\tpre_reasons\tpre_pct_CDS\n" + "".join("\t".join(r) + "\n" for r in qc_rows))
        cmd += ["--qc", str(qc)]
    env = dict(os.environ, LABEL_DIR=str(lab), DIGEST_DIR=str(dig))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout
    rows = {r["GSM"]: r for r in csv.DictReader(open(tmp_path / "out.tsv"), delimiter="\t")}
    problems = list(csv.DictReader(open(tmp_path / "out_problems.tsv"), delimiter="\t"))
    return out, rows, problems


def test_labels_and_lumped_columns(tmp_path):
    out, rows, problems = run(tmp_path, LABELS, [("GSM1", "pass", "", "80"), ("GSM2", "pass", "", "10"), ("GSM3", "fail", "phage", "70")])
    assert "3 samples in 1 studies; 0 problems" in out and not problems
    assert set(rows) == {"GSM1", "GSM2", "GSM3"}                      # non-candidates are left out
    a, b, c = rows["GSM1"], rows["GSM2"], rows["GSM3"]
    assert (a["phase2"], b["phase2"]) == ("exponential", "stationary")
    assert a["medium_type"] == "rich" and a["treated"] == "false"
    assert (a["is_wild_type"], a["is_reference_genotype"]) == ("true", "true")
    assert (b["is_wild_type"], b["is_reference_genotype"]) == ("false", "false")
    # phage-infected: own class, phase label withheld from treated samples, phage RNA is not a second organism
    assert c["phage_infected"] == "true" and c["phase2"] == "" and c["mixed_organism"] == "false"
    assert (a["usable"], c["usable"]) == ("true", "false")
    assert (a["low_mrna_prescreen"], b["low_mrna_prescreen"]) == ("false", "true")


def test_problems_are_reported(tmp_path):
    bad = json.loads(json.dumps(LABELS))
    bad["default"]["base_medium"] = {"v": "LB", "b": "stated", "q": "grown in Luria broth"}        # quote not in the text
    bad["default"]["medium_class"] = {"v": "rich", "b": "inferred", "q": "Cells were grown in LB"}  # value not in the scheme
    bad["rules"].append({"field": "title", "regex": "no such title", "set": {"oxygen": {"v": "aerobic", "b": "stated", "q": ""}}})
    _, rows, problems = run(tmp_path, bad)
    text = " | ".join(p["problem"] for p in problems)
    assert "quote not in text" in text and "value not in scheme" in text and "matches no sample" in text
    assert rows["GSM1"]["medium_class"] == ""                         # an invalid value is not written


def test_supplementary_fields(tmp_path):
    sys.path.insert(0, str(META))
    from geo_digest import add_suppl
    (tmp_path / "GSE1.tsv").write_text("# medium: M9 glucose\nGSM\tstrain\nGSM1\tWT\nGSM9\tother\n")
    rows = add_suppl("GSE1", {"GSM1": {"title": "a"}, "GSM2": {"title": "b"}}, base=tmp_path)
    assert rows["GSM1"] == {"title": "a", "sup:strain": "WT", "sup:medium": "M9 glucose"}
    assert rows["GSM2"] == {"title": "b", "sup:medium": "M9 glucose"}      # header fields reach every sample
    assert add_suppl("GSE2", {"GSM1": {}}, base=tmp_path) == {"GSM1": {}}   # no table for the study
