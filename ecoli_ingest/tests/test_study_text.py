"""atlas_build/metadata/study_text.py: study and sample text for `ecoli_atlas view`."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

META = Path(__file__).resolve().parents[1] / "atlas_build" / "metadata"

SOFT = """^SERIES = GSE1
!Series_title = Test study
!Series_summary = Wild type and rpoS deletion
!Series_summary = in LB.
!Series_overall_design = Two strains,\texponential and stationary.
!Series_type = Expression profiling by high throughput sequencing
!Series_pubmed_id = 111
!Series_pubmed_id = 222
!Series_contributor = Doe,,J
!Series_submission_date = Jan 01 2020
!Series_relation = BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1
!Series_relation = SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRP1
^SAMPLE = GSM1
!Sample_title = WT exponential rep1
!Sample_source_name_ch1 = cells
!Sample_characteristics_ch1 = genotype: wild type
!Sample_characteristics_ch1 = growth phase: exponential
^SAMPLE = GSM2
!Sample_title = rpoS stationary rep1
!Sample_characteristics_ch1 = genotype: delta rpoS
^SAMPLE = GSM9
!Sample_title = not in the atlas
"""


def test_study_text(tmp_path):
    soft, labels = tmp_path / "soft", tmp_path / "labels"
    soft.mkdir(), labels.mkdir()
    (soft / "GSE1.soft").write_text(SOFT)
    (labels / "GSE1.json").write_text(json.dumps({"gse": "GSE1", "notes": "first note | second\tnote"}))
    (tmp_path / "meta.tsv").write_text("GSM\tGSE\tgrowth_phase\nGSM1\tGSE1\tmid_exponential\nGSM2\tGSE1\tstationary\nGSM5\tGSE5\t\n")
    (tmp_path / "qc.tsv").write_text("GSM\tGSE\truns\tlayout\tinstrument_model\tfull_layout\nGSM1\tGSE1\tSRR1,SRR2\tPAIRED\tHiSeq\tPE\n")
    (tmp_path / "studies.tsv").write_text("GSE\tstudy_accession\tfirst_public\nGSE1\tPRJNA1\t2020-02-02\n")
    out = subprocess.run([sys.executable, str(META / "study_text.py"), str(soft), str(tmp_path / "meta.tsv"), str(tmp_path / "out" / "m"),
                          "--qc", str(tmp_path / "qc.tsv"), "--studies", str(tmp_path / "studies.tsv")],
                         check=True, capture_output=True, text=True, env=dict(os.environ, LABEL_DIR=str(labels))).stdout
    assert "1 studies, 2 samples; no SOFT file for 1 studies: GSE5" in out
    studies = list(csv.DictReader(open(tmp_path / "out" / "m_studies.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE))
    samples = {r["GSM"]: r for r in csv.DictReader(open(tmp_path / "out" / "m_sample_text.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)}
    s = studies[0]
    assert s["GSE"] == "GSE1" and s["summary"] == "Wild type and rpoS deletion in LB."
    assert s["overall_design"] == "Two strains, exponential and stationary."           # tab inside a field removed
    assert (s["pubmed_ids"], s["bioproject"], s["sra"]) == ("111,222", "PRJNA1", "SRP1")
    assert s["first_public"] == "2020-02-02" and s["n_samples"] == "2" and s["curator_notes"] == "first note | second note"
    assert set(samples) == {"GSM1", "GSM2"}
    a = samples["GSM1"]
    assert a["title"] == "WT exponential rep1" and a["characteristics"] == "genotype: wild type; growth phase: exponential"
    assert (a["runs"], a["layout"], a["instrument"]) == ("SRR1,SRR2", "PE", "HiSeq")
    assert samples["GSM2"]["runs"] == ""
