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


FILTERED_GFF = """\
##gff-version 3
NC_1\tRefSeq\tfeature\t337\t2799\t.\t+\t.\tID=gene-b0002;Dbxref=ASAP:ABE-0000008,ECOCYC:EG10998,GeneID:945803;Name=thrA;gbkey=Gene;gene=thrA;gene_biotype=protein_coding;gene_synonym=ECK0002,Hs;locus_tag=b0002
NC_1\tRefSeq\tfeature\t225381\t225457\t.\t+\t.\tID=rna-b0202;Parent=gene-b0202;Dbxref=GeneID:944884;Note=tRNA-Ile(GAU);gbkey=tRNA;gene=ileV;locus_tag=b0202;product=tRNA-Ile;Name=ileV
NC_1\tRefSeq\tfeature\t5000\t5300\t.\t-\t.\tID=gene-b9999;Dbxref=GeneID:1;Name=fakP;gbkey=Gene;gene=fakP;gene_biotype=pseudogene;locus_tag=b9999;pseudo=true
NC_1\tRfam\tfeature\t191\t310\t.\t+\t.\tID=Thr_leader;Dbxref=Rfam:RF00506;gbkey=ncRNA;Name=Thr_leader
NC_1\tRefSeq\tfeature\t900\t950\t.\t+\t.\tID=gene-b7777;Name=dupX;gbkey=exon;locus_tag=WRONG
NC_1\tRefSeq\tfeature\t900\t950\t.\t+\t.\tID=gene-b7777;Dbxref=GeneID:7;Name=dupX;gbkey=Gene;locus_tag=b7777;gene_biotype=protein_coding
"""

SOURCE_GFF = """\
##gff-version 3
NC_1\tRefSeq\tgene\t337\t2799\t.\t+\t.\tID=gene-b0002;Name=thrA;gbkey=Gene;locus_tag=b0002
NC_1\tRefSeq\tCDS\t337\t2799\t.\t+\t0\tID=cds-1;Parent=gene-b0002;gbkey=CDS;locus_tag=b0002;product=fused aspartate kinase%2Fhomoserine dehydrogenase 1
NC_1\tRefSeq\tgene\t225381\t225457\t.\t+\t.\tID=gene-b0202;Name=ileV;gbkey=Gene;locus_tag=b0202
NC_1\tRefSeq\ttRNA\t225381\t225457\t.\t+\t.\tID=rna-b0202;Parent=gene-b0202;locus_tag=b0202;product=tRNA-Ile
NC_1\tRefSeq\tgene\t900\t950\t.\t+\t.\tID=gene-b7777;Name=dupX;gbkey=Gene;locus_tag=b7777
NC_1\tRefSeq\tCDS\t900\t950\t.\t+\t0\tID=cds-7;Parent=gene-b7777;locus_tag=b7777;product=hypothetical protein
"""

COUNTS = """\
# Program:featureCounts v2.0.6; Command:"featureCounts" "-a" "x.gff"
Geneid\tChr\tStart\tEnd\tStrand\tLength\tres/s1/vg/s1_sort.bam\tres/s2/vg/s2_sort.bam
thrA\tNC_1\t337\t2799\t+\t2463\t388\t233
ileV\tNC_1\t225381\t225457\t+\t77\t0\t5
fakP\tNC_1\t5000\t5300\t-\t301\t1\t2
Thr_leader\tNC_1\t191\t310\t+\t120\t9\t8
dupX\tNC_1\t900\t950\t+\t51\t3\t4
unknownGene\tNC_1\t1\t2\t+\t2\t0\t0
"""


def _read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh, delimiter="\t"))


def test_annotate_end_to_end(tmp_path, annotate_mod):
    (tmp_path / "f.gff").write_text(FILTERED_GFF)
    (tmp_path / "s.gff").write_text(SOURCE_GFF)
    (tmp_path / "counts.txt").write_text(COUNTS)
    ext, sym = tmp_path / "ext.tsv", tmp_path / "sym.tsv"
    rc = annotate_mod.main(["--counts", str(tmp_path / "counts.txt"), "--gff", str(tmp_path / "f.gff"),
                            "--source-gff", str(tmp_path / "s.gff"), "--extended", str(ext), "--symbols", str(sym)])
    assert rc == 0

    rows = _read_tsv(ext)
    assert rows[0] == ["Gene_Symbol", "Gene_IDs_Field", "Gene_Summary", "Chr", "Start", "End", "Strand", "Length",
                       "res/s1/vg/s1_sort.bam", "res/s2/vg/s2_sort.bam"]
    by_name = {r[0]: r for r in rows[1:]}
    assert by_name["thrA"][1] == "b0002,ASAP:ABE-0000008,ECOCYC:EG10998,GeneID:945803"
    assert by_name["thrA"][2] == "fused aspartate kinase/homoserine dehydrogenase 1"  # percent-decoded
    assert by_name["thrA"][3:] == ["NC_1", "337", "2799", "+", "2463", "388", "233"]
    assert by_name["ileV"][2] == "tRNA-Ile"
    assert by_name["fakP"][2] == "pseudogene"  # no product anywhere -> biotype fallback
    assert by_name["Thr_leader"][1] == "Rfam:RF00506"
    assert by_name["Thr_leader"][2] == "Rfam ncRNA family Thr_leader (RF00506)"
    assert by_name["dupX"][1] == "b7777,GeneID:7"  # gene row wins over the earlier exon row
    assert by_name["unknownGene"][1] == "unknownGene" and by_name["unknownGene"][2] == "NA"

    srows = _read_tsv(sym)
    assert srows[0] == ["Gene_Symbol", "Chr", "Start", "End", "Strand", "Length",
                        "res/s1/vg/s1_sort.bam", "res/s2/vg/s2_sort.bam"]
    assert len(srows) == len(rows)
    assert srows[1] == ["thrA", "NC_1", "337", "2799", "+", "2463", "388", "233"]


def test_synonyms_flag(tmp_path, annotate_mod):
    (tmp_path / "f.gff").write_text(FILTERED_GFF)
    (tmp_path / "s.gff").write_text(SOURCE_GFF)
    (tmp_path / "counts.txt").write_text(COUNTS)
    ext, sym = tmp_path / "ext.tsv", tmp_path / "sym.tsv"
    annotate_mod.main(["--counts", str(tmp_path / "counts.txt"), "--gff", str(tmp_path / "f.gff"),
                       "--source-gff", str(tmp_path / "s.gff"), "--extended", str(ext), "--symbols", str(sym),
                       "--with-synonyms"])
    thrA = {r[0]: r for r in _read_tsv(ext)[1:]}["thrA"]
    assert thrA[2].endswith("[synonyms: ECK0002,Hs]")
