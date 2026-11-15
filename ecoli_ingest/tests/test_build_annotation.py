import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "build" / "build_annotation.py"


@pytest.fixture(scope="module")
def ba():
    spec = importlib.util.spec_from_file_location("build_annotation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_annotation"] = mod
    spec.loader.exec_module(mod)
    return mod


def g(start, end, strand, rowtype, attrs):
    return f"NC_1\tRefSeq\t{rowtype}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n"


NCBI_GFF = "##gff-version 3\n" + "".join([
    g(1, 5000, "+", "region", "ID=NC_1:1..5000"),
    # leader peptide nested in an Rfam leader element: both must be kept
    g(190, 255, "+", "gene", "ID=gene-b0001;Name=thrL;gene_biotype=protein_coding;locus_tag=b0001;Dbxref=GeneID:1"),
    g(190, 255, "+", "CDS", "ID=cds-1;Parent=gene-b0001;locus_tag=b0001;product=thr operon leader peptide"),
    g(337, 900, "+", "gene", "ID=gene-b0002;Name=thrA;gene_biotype=protein_coding;locus_tag=b0002"),
    g(337, 900, "+", "CDS", "ID=cds-2;Parent=gene-b0002;locus_tag=b0002;product=kinase%2C fused"),
    # small ORF nested in an sRNA gene on the same strand
    g(1000, 1227, "+", "gene", "ID=gene-b4577;Name=sgrS;gene_biotype=ncRNA;locus_tag=b4577"),
    g(1000, 1227, "+", "ncRNA", "ID=rna-b4577;Parent=gene-b4577;locus_tag=b4577;product=small regulatory RNA SgrS"),
    g(1050, 1181, "+", "gene", "ID=gene-b4662;Name=sgrT;gene_biotype=protein_coding;locus_tag=b4662"),
    g(1500, 1576, "-", "gene", "ID=gene-b0202;Name=ileV;gene_biotype=tRNA;locus_tag=b0202"),
    g(2000, 3541, "+", "gene", "ID=gene-b0201;Name=rrsH;gene_biotype=rRNA;locus_tag=b0201"),
    g(3600, 3962, "+", "gene", "ID=gene-b2621;Name=ssrA;gene_biotype=other;locus_tag=b2621"),
    g(4000, 4300, "-", "pseudogene", "ID=gene-b0240;Name=crl;gene_biotype=pseudogene;locus_tag=b0240;pseudo=true"),
    g(4400, 4500, "+", "gene", "ID=gene-b9000;Name=b9000;gene_biotype=protein_coding;locus_tag=b9000"),
    g(4600, 4700, "-", "gene", "ID=gene-b4413;Name=sokC;gene_biotype=ncRNA;locus_tag=b4413"),
])

HEAD = ("#idx target name accession query name accession clan name mdl mdl from mdl to seq from seq to strand "
        "trunc pass gc bias score E-value inc olp anyidx afrct1 afrct2 winidx wfrct1 wfrct2 mdl len seq len "
        "description of target\n")


def hit(idx, name, acc, start, end, strand, desc):
    a, b = (start, end) if strand == "+" else (end, start)
    return (f"{idx} {name} {acc} NC_1 - - cm 1 100 {a} {b} {strand} no 1 0.5 0.0 88.8 1e-10 ! * - - - - - - "
            f"100 5000 {desc}\n")


TBLOUT = HEAD + "".join([
    hit(1, "Thr_leader", "RF00506", 191, 310, "+", "Threonine operon leader"),     # kept (overlaps a CDS only)
    hit(2, "SgrS", "RF00534", 1001, 1227, "+", "SgrS RNA"),                        # duplicate of ncRNA sgrS
    hit(3, "tRNA", "RF00005", 1500, 1576, "-", "tRNA"),                            # duplicate of tRNA ileV
    hit(4, "SSU_rRNA_bacteria", "RF00177", 2000, 3541, "+", "Bacterial small subunit ribosomal RNA"),
    hit(5, "tmRNA", "RF00023", 3600, 3962, "+", "transfer-messenger RNA"),         # duplicate of ssrA (biotype other)
    hit(6, "naRNA4", "RF02564", 3000, 3070, "-", "Nucleoid-associated noncoding RNA 4"),   # excluded family
    hit(7, "SECIS_3", "RF01989", 400, 458, "+", "Selenocysteine insertion sequence 3"),    # excluded family
    hit(8, "RtT", "RF00391", 4800, 4900, "+", "RtT RNA"),                          # two copies -> numbered
    hit(9, "RtT", "RF00391", 4350, 4390, "-", "RtT RNA"),
    hit(10, "sok", "RF01794", 4590, 4720, "+", "sok antitoxin"),                   # opposite strand to ncRNA sokC: dropped
    hit(12, "HPnc0260", "RF02194", 1490, 1600, "+", "antisense RNA"),              # opposite strand to a tRNA: kept
    hit(11, "tRNA", "RF00005", 1560, 1640, "-", "tRNA"),                           # overlaps ileV by < 50%: kept
]) + "#\n# Program:         cmscan\n# Version:         1.1.5 (Sep 2023)\n# Option settings: cmscan --cut_ga Rfam.cm x.fa\n"


@pytest.fixture
def built(ba, tmp_path):
    (tmp_path / "n.gff").write_text(NCBI_GFF)
    (tmp_path / "r.tblout").write_text(TBLOUT)
    paths = ba.build(str(tmp_path / "n.gff"), str(tmp_path / "r.tblout"), str(tmp_path / "out"), "ann", "test")
    with open(paths["features.tsv"], newline="") as fh:
        feats = {r["Feature_ID"]: r for r in csv.DictReader(fh, delimiter="\t")}
    with open(paths["rfam_merge_report.tsv"], newline="") as fh:
        report = {r["idx"]: r for r in csv.DictReader(fh, delimiter="\t")}
    return paths, feats, report


def test_feature_ids_and_types(built):
    _, feats, _ = built
    assert set(feats) == {
        "CDS_b0001_thrL", "CDS_b0002_thrA", "ncRNA_b4577_sgrS", "CDS_b4662_sgrT", "tRNA_b0202_ileV",
        "rRNA_b0201_rrsH", "ncRNA_b2621_ssrA", "pseudo_b0240_crl", "CDS_b9000", "ncRNA_b4413_sokC",
        "ncRNA_Thr_leader", "ncRNA_RtT_1", "ncRNA_RtT_2", "ncRNA_HPnc0260", "ncRNA_tRNA"}
    assert feats["CDS_b9000"]["Symbol"] == ""                       # a gene named by its locus tag has no symbol
    assert feats["CDS_b0002_thrA"]["Product"] == "kinase, fused"
    assert feats["ncRNA_b4577_sgrS"]["Rfam"] == "RF00534" and feats["ncRNA_b4577_sgrS"]["Source"] == "NCBI"
    assert feats["ncRNA_b2621_ssrA"]["Rfam"] == "RF00023"           # tmRNA: the NCBI entry is kept
    assert feats["ncRNA_Thr_leader"]["Source"] == "Rfam" and feats["ncRNA_Thr_leader"]["Locus_Tag"] == ""
    assert (feats["ncRNA_RtT_1"]["Start"], feats["ncRNA_RtT_2"]["Start"]) == ("4350", "4800")   # by position


def test_nested_features_are_kept_and_reported(built):
    _, feats, _ = built
    assert feats["CDS_b4662_sgrT"]["Overlaps_Same_Strand"] == "ncRNA_b4577_sgrS"
    assert feats["CDS_b0001_thrL"]["Overlaps_Same_Strand"] == "ncRNA_Thr_leader"
    assert feats["ncRNA_HPnc0260"]["Overlaps_Opposite_Strand"].split(",") == ["tRNA_b0202_ileV", "ncRNA_tRNA"]
    assert feats["ncRNA_b4413_sokC"]["Rfam"] == ""                  # an antisense hit is not credited to the gene


def test_every_hit_has_a_fate(built):
    _, _, report = built
    assert len(report) == 12
    fate = {i: r["fate"] for i, r in report.items()}
    assert fate == {"1": "kept", "2": "duplicate", "3": "duplicate", "4": "duplicate", "5": "duplicate",
                    "6": "excluded", "7": "excluded", "8": "kept", "9": "kept", "10": "antisense_duplicate",
                    "11": "kept", "12": "kept"}
    assert report["2"]["duplicate_of"] == "ncRNA_b4577_sgrS"
    assert report["3"]["duplicate_of"] == "tRNA_b0202_ileV"
    assert report["5"]["duplicate_of"] == "ncRNA_b2621_ssrA"
    assert "REP repeat" in report["6"]["reason"]
    assert report["10"]["duplicate_of"] == "ncRNA_b4413_sokC" and report["10"]["feature_id"] == ""
    assert all(r["duplicate_of"].split("_")[0] in ("ncRNA", "tRNA", "rRNA") for r in report.values()
               if r["fate"] == "duplicate")


def test_gff_and_bed(built):
    paths, feats, _ = built
    lines = Path(paths["gff"]).read_text().splitlines()
    assert lines[0] == "##gff-version 3" and any(l.startswith("#!rfam-search cmscan 1.1.5") for l in lines)
    rows = [l.split("\t") for l in lines if not l.startswith("#")]
    assert len(rows) == len(feats) and all(r[2] == "gene" for r in rows)
    ids = [dict(x.split("=", 1) for x in r[8].split(";"))["ID"] for r in rows]
    assert len(set(ids)) == len(ids)
    bed = [l.split("\t") for l in Path(paths["bed"]).read_text().splitlines()]
    names = {b[3] for b in bed}
    assert "CDS:CDS_b0001_thrL" in names and "Rfam:ncRNA_Thr_leader" in names and "pseudo:pseudo_b0240_crl" in names
    thrL = next(b for b in bed if b[3] == "CDS:CDS_b0001_thrL")
    assert thrL[1:3] == ["189", "255"] and thrL[5] == "+" and len(thrL) == 12


def test_hits_on_unknown_sequence_fail(ba, tmp_path):
    (tmp_path / "n.gff").write_text(NCBI_GFF)
    (tmp_path / "r.tblout").write_text(TBLOUT.replace(" NC_1 ", " NC_2 "))
    with pytest.raises(SystemExit, match="not in the GFF"):
        ba.build(str(tmp_path / "n.gff"), str(tmp_path / "r.tblout"), str(tmp_path / "out"), "ann")
