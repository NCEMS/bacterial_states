import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "ecoli_ingest" / "workflow" / "scripts" / "annotate_counts.py"


@pytest.fixture(scope="module")
def annotate_mod():
    spec = importlib.util.spec_from_file_location("annotate_counts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["annotate_counts"] = mod
    spec.loader.exec_module(mod)
    return mod


GFF = """\
##gff-version 3
#!ecoli_ingest counting annotation
NC_1\tRefSeq\tgene\t337\t2799\t.\t+\t.\tID=CDS_b0002_thrA;feature_type=CDS;source=NCBI;locus_tag=b0002;symbol=thrA;Dbxref=ASAP:ABE-0000008,ECOCYC:EG10998,GeneID:945803;product=fused aspartate kinase/homoserine dehydrogenase 1%3B bifunctional
NC_1\tRefSeq\tgene\t225381\t225457\t.\t+\t.\tID=tRNA_b0202_ileV;feature_type=tRNA;source=NCBI;locus_tag=b0202;symbol=ileV;Dbxref=GeneID:944884;rfam=RF00005;product=tRNA-Ile
NC_1\tRefSeq\tgene\t5000\t5300\t.\t-\t.\tID=pseudo_b9999;feature_type=pseudo;source=NCBI;locus_tag=b9999
NC_1\tRfam\tgene\t191\t310\t.\t+\t.\tID=ncRNA_Thr_leader;feature_type=ncRNA;source=Rfam;symbol=Thr_leader;rfam=RF00506;product=Threonine operon leader
"""

COUNTS = """\
# Program:featureCounts v2.1.1; Command:"featureCounts" "-a" "x.gff"
Geneid\tChr\tStart\tEnd\tStrand\tLength\tres/s1/vg/s1_sort.bam\tres/s2/vg/s2_sort.bam
CDS_b0002_thrA\tNC_1\t337\t2799\t+\t2463\t388\t233
tRNA_b0202_ileV\tNC_1\t225381\t225457\t+\t77\t0\t5
pseudo_b9999\tNC_1\t5000\t5300\t-\t301\t1\t2
ncRNA_Thr_leader\tNC_1\t191\t310\t+\t120\t9\t8
"""


def _read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh, delimiter="\t"))


def test_annotate_end_to_end(tmp_path, annotate_mod):
    (tmp_path / "a.gff").write_text(GFF)
    (tmp_path / "counts.txt").write_text(COUNTS)
    ext, cnt = tmp_path / "ext.tsv", tmp_path / "cnt.tsv"
    rc = annotate_mod.main(["--counts", str(tmp_path / "counts.txt"), "--gff", str(tmp_path / "a.gff"),
                            "--extended", str(ext), "--counts-out", str(cnt)])
    assert rc == 0

    rows = _read_tsv(ext)
    assert rows[0] == ["Feature_ID", "Symbol", "Locus_Tag", "Type", "Source", "Rfam", "Dbxref", "Product",
                       "Chr", "Start", "End", "Strand", "Length", "res/s1/vg/s1_sort.bam", "res/s2/vg/s2_sort.bam"]
    by_id = {r[0]: r for r in rows[1:]}
    assert by_id["CDS_b0002_thrA"][1:8] == [
        "thrA", "b0002", "CDS", "NCBI", "NA", "ASAP:ABE-0000008,ECOCYC:EG10998,GeneID:945803",
        "fused aspartate kinase/homoserine dehydrogenase 1; bifunctional"]          # percent-decoded
    assert by_id["CDS_b0002_thrA"][8:] == ["NC_1", "337", "2799", "+", "2463", "388", "233"]
    assert by_id["tRNA_b0202_ileV"][3:6] == ["tRNA", "NCBI", "RF00005"]
    assert by_id["pseudo_b9999"][1:8] == ["NA", "b9999", "pseudo", "NCBI", "NA", "NA", "NA"]
    assert by_id["ncRNA_Thr_leader"][1:8] == ["Thr_leader", "NA", "ncRNA", "Rfam", "RF00506", "NA",
                                              "Threonine operon leader"]

    crows = _read_tsv(cnt)
    assert crows[0] == ["Feature_ID", "Chr", "Start", "End", "Strand", "Length",
                        "res/s1/vg/s1_sort.bam", "res/s2/vg/s2_sort.bam"]
    assert len(crows) == len(rows)
    assert crows[1] == ["CDS_b0002_thrA", "NC_1", "337", "2799", "+", "2463", "388", "233"]


def test_counts_from_another_annotation_are_refused(tmp_path, annotate_mod):
    (tmp_path / "a.gff").write_text(GFF)
    (tmp_path / "counts.txt").write_text(COUNTS + "thrA\tNC_1\t1\t2\t+\t2\t0\t0\n")
    with pytest.raises(SystemExit, match="different annotation"):
        annotate_mod.main(["--counts", str(tmp_path / "counts.txt"), "--gff", str(tmp_path / "a.gff"),
                           "--extended", str(tmp_path / "e.tsv"), "--counts-out", str(tmp_path / "c.tsv")])
