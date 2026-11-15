#!/usr/bin/env python
"""Build the inGEST counting annotation from an unmodified NCBI GFF and an Infernal/Rfam hit table.

Inputs
  --gff      the NCBI RefSeq GFF of the assembly, exactly as downloaded (plain or .gz).
  --tblout   `cmscan --fmt 2 --tblout` output for the same sequence (Rfam.cm, clan-overlapping
             hits already removed).

Outputs (all in --out-dir, named <prefix>.*)
  .gff                 one `gene` row per counted feature; featureCounts runs with `-t gene -g ID`
  .bed                 the same features as BED12 (RSeQC, feature-overlap plot); name = <Type>:<ID>
  .features.tsv        one line per feature with its identifiers and the features it overlaps
  .rfam_merge_report.tsv   the fate of every Rfam hit

Rules
  * Every NCBI gene and pseudogene gives exactly one feature. Nothing is removed because it is
    nested in, or overlaps, another feature.
  * An Rfam hit is dropped as a duplicate only when an NCBI RNA gene (ncRNA, tRNA, rRNA) lies on the
    same strand and their overlap covers at least half of the shorter of the two. The NCBI entry is
    kept and records the Rfam accession.
  * An Rfam hit is also dropped when an NCBI ncRNA gene lies on the opposite strand with the same
    amount of overlap: such hits are the annotated locus matched on the wrong strand (the sok
    antitoxins, for example). The NCBI entry is kept; it does not record the Rfam accession.
  * Rfam families in RFAM_EXCLUDE are never added (repeats and recoding signals, not RNA genes).
  * Feature ids: <Type>_<locus_tag>_<symbol> for NCBI genes (Type = CDS, ncRNA, tRNA, rRNA, pseudo),
    ncRNA_<Rfam id> for Rfam hits, with _1, _2, ... by genome position when a family has several.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import quote, unquote

# Rfam families that are not added to the annotation: accession -> reason.
RFAM_EXCLUDE = {
    "RF02564": "REP repeat (naRNA4)",
    "RF00382": "ribosomal frameshifting element (DnaX)",
    "RF01988": "selenocysteine insertion sequence (SECIS_2)",
    "RF01989": "selenocysteine insertion sequence (SECIS_3)",
}
RNA_TYPES = ("ncRNA", "tRNA", "rRNA")
DUPLICATE_MIN_FRACTION = 0.5
FEATURE_COLUMNS = ["Feature_ID", "Symbol", "Locus_Tag", "Type", "Source", "Rfam", "Chr", "Start", "End",
                   "Strand", "Length", "Product", "Overlaps_Same_Strand", "Overlaps_Opposite_Strand"]
REPORT_COLUMNS = ["idx", "family", "accession", "start", "end", "strand", "score", "evalue", "fate",
                  "feature_id", "duplicate_of", "reason", "overlaps_same_strand", "overlaps_opposite_strand"]


@dataclass
class Feature:
    seqid: str
    start: int
    end: int
    strand: str
    ftype: str            # CDS, ncRNA, tRNA, rRNA, pseudo
    source: str           # NCBI or Rfam
    fid: str = ""
    symbol: str = ""
    locus_tag: str = ""
    product: str = ""
    dbxref: str = ""
    synonyms: str = ""
    rfam: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class Hit:
    idx: str
    family: str
    accession: str
    seqid: str
    start: int
    end: int
    strand: str
    score: str
    evalue: str
    description: str
    fate: str = ""
    feature_id: str = ""
    duplicate_of: str = ""
    reason: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def md5_of(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_attributes(text: str) -> dict:
    attrs = {}
    for item in text.strip().split(";"):
        key, sep, value = item.partition("=")
        if sep:
            attrs[key.strip()] = unquote(value.strip())
    return attrs


def safe(text: str) -> str:
    """Identifier-safe form of a symbol or family name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def gene_type(row_type: str, biotype: str) -> str:
    if row_type == "pseudogene" or biotype == "pseudogene":
        return "pseudo"
    if biotype == "protein_coding":
        return "CDS"
    if biotype in ("tRNA", "rRNA"):
        return biotype
    return "ncRNA"   # ncRNA, tmRNA ("other"), RNase P RNA, SRP RNA, antisense RNA, ...


def read_ncbi_genes(gff: str) -> list[Feature]:
    """One Feature per gene / pseudogene row, with the product of its child feature."""
    genes: list[Feature] = []
    product: dict[str, str] = {}
    note: dict[str, str] = {}
    with open_text(gff) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue
            attrs = parse_attributes(cols[8])
            locus = attrs.get("locus_tag", "")
            if cols[2] in ("gene", "pseudogene"):
                if not locus:
                    raise SystemExit(f"{gff}: gene row without locus_tag: {cols[8][:80]}")
                name = attrs.get("Name") or attrs.get("gene") or ""
                genes.append(Feature(cols[0], int(cols[3]), int(cols[4]), cols[6],
                                     gene_type(cols[2], attrs.get("gene_biotype", "")), "NCBI",
                                     symbol="" if name == locus else name, locus_tag=locus,
                                     dbxref=attrs.get("Dbxref", ""), synonyms=attrs.get("gene_synonym", "")))
            elif locus:
                if attrs.get("product") and locus not in product:
                    product[locus] = attrs["product"]
                if attrs.get("Note") and locus not in note:
                    note[locus] = attrs["Note"]
    seen: dict[str, int] = defaultdict(int)
    for g in genes:
        g.product = product.get(g.locus_tag) or note.get(g.locus_tag) or ""
        g.fid = "_".join(x for x in (g.ftype, safe(g.locus_tag), safe(g.symbol)) if x)
        seen[g.fid] += 1
    dup = sorted(k for k, n in seen.items() if n > 1)
    if dup:
        raise SystemExit(f"{gff}: feature ids are not unique: {', '.join(dup[:10])}")
    return genes


def read_tblout(path: str) -> tuple[list[Hit], dict]:
    """Hits of a `cmscan --fmt 2` table, plus the program lines of its footer."""
    hits: list[Hit] = []
    meta: dict[str, str] = {}
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.match(r"#\s*(Program|Version|Target file|Option settings|Date):\s*(.*\S)", line)
                if m:
                    meta[m.group(1)] = m.group(2)
                continue
            if not line.strip():
                continue
            f = line.rstrip("\n").split(None, 28)
            if len(f) < 28:
                raise SystemExit(f"{path}: expected a cmscan --fmt 2 table, found {len(f)} columns")
            a, b = int(f[9]), int(f[10])
            if (f[11] == "+") != (a <= b):
                raise SystemExit(f"{path}: hit {f[0]} has coordinates that disagree with its strand")
            hits.append(Hit(f[0], f[1], f[2], f[3], min(a, b), max(a, b), f[11], f[16], f[17],
                            f[28].strip() if len(f) > 28 else ""))
    return hits, meta


def overlap(a, b) -> int:
    if a.seqid != b.seqid:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start) + 1)


def merge_rfam(genes: list[Feature], hits: list[Hit]) -> list[Feature]:
    """Decide the fate of every hit (in place) and return the Rfam features that are added."""
    rna_genes = [g for g in genes if g.ftype in RNA_TYPES]
    kept: list[Hit] = []
    for h in hits:
        if h.accession in RFAM_EXCLUDE:
            h.fate, h.reason = "excluded", RFAM_EXCLUDE[h.accession]
            continue
        best = None
        for g in rna_genes:
            if g.strand != h.strand:
                continue
            o = overlap(h, g)
            if o and o >= DUPLICATE_MIN_FRACTION * min(h.length, g.length) and (best is None or o > best[0]):
                best = (o, g)
        antisense = None
        if not best:
            for g in rna_genes:
                if g.ftype != "ncRNA" or g.strand == h.strand:
                    continue
                o = overlap(h, g)
                if o and o >= DUPLICATE_MIN_FRACTION * min(h.length, g.length) and (
                        antisense is None or o > antisense[0]):
                    antisense = (o, g)
        if best:
            g = best[1]
            h.fate, h.duplicate_of = "duplicate", g.fid
            h.reason = f"same strand as NCBI {g.ftype} gene, overlap {best[0]} bp"
            if h.accession not in g.rfam:
                g.rfam.append(h.accession)
        elif antisense:
            g = antisense[1]
            h.fate, h.duplicate_of = "antisense_duplicate", g.fid
            h.reason = f"opposite strand to NCBI ncRNA gene, overlap {antisense[0]} bp"
        else:
            h.fate = "kept"
            kept.append(h)
    per_family: dict[str, list[Hit]] = defaultdict(list)
    for h in sorted(kept, key=lambda h: (h.seqid, h.start, h.end)):
        per_family[h.family].append(h)
    used = {g.fid for g in genes}
    added: list[Feature] = []
    for family, members in per_family.items():
        for i, h in enumerate(members, 1):
            h.feature_id = f"ncRNA_{safe(family)}" + (f"_{i}" if len(members) > 1 else "")
            if h.feature_id in used:
                raise SystemExit(f"feature id {h.feature_id} is not unique")
            used.add(h.feature_id)
            added.append(Feature(h.seqid, h.start, h.end, h.strand, "ncRNA", "Rfam", fid=h.feature_id,
                                 symbol=family, product=h.description, rfam=[h.accession]))
    return added


def overlap_lists(items: list, features: list[Feature]) -> list[tuple[list[str], list[str]]]:
    """For each item, the ids of the features it overlaps on the same and on the opposite strand."""
    by_start = sorted(features, key=lambda f: (f.seqid, f.start))
    out = []
    for it in items:
        same, opposite = [], []
        for f in by_start:
            if f.seqid != it.seqid or f.end < it.start:
                continue
            if f.start > it.end:
                break
            if f is it or (getattr(it, "fid", None) and f.fid == it.fid):
                continue
            (same if f.strand == it.strand else opposite).append(f.fid)
        out.append((same, opposite))
    return out


def gff_escape(value: str) -> str:
    return quote(value, safe=" ,:/()[]{}'+*?!@#$^|<>._-~\"\\")


def write_outputs(features: list[Feature], hits: list[Hit], out_dir: str, prefix: str, header: list[str],
                  ncbi_genes: list[Feature]) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    features = sorted(features, key=lambda f: (f.seqid, f.start, f.end, f.fid))
    paths = {k: os.path.join(out_dir, f"{prefix}.{k}") for k in
             ("gff", "bed", "features.tsv", "rfam_merge_report.tsv")}

    with open(paths["gff"], "w", encoding="utf-8") as fh:
        fh.write("##gff-version 3\n")
        for line in header:
            fh.write(f"#!{line}\n")
        for f in features:
            attrs = [("ID", f.fid), ("feature_type", f.ftype), ("source", f.source),
                     ("locus_tag", f.locus_tag), ("symbol", f.symbol), ("Dbxref", f.dbxref),
                     ("rfam", ",".join(f.rfam)), ("gene_synonym", f.synonyms), ("product", f.product)]
            text = ";".join(f"{k}={gff_escape(v)}" for k, v in attrs if v)
            fh.write("\t".join([f.seqid, "RefSeq" if f.source == "NCBI" else "Rfam", "gene", str(f.start),
                                str(f.end), ".", f.strand, ".", text]) + "\n")

    with open(paths["bed"], "w", encoding="utf-8") as fh:
        for f in features:
            group = "Rfam" if f.source == "Rfam" else f.ftype
            fh.write("\t".join(map(str, [f.seqid, f.start - 1, f.end, f"{group}:{f.fid}", 0, f.strand,
                                         f.start - 1, f.end, 0, 1, f"{f.length},", "0,"])) + "\n")

    with open(paths["features.tsv"], "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(FEATURE_COLUMNS)
        for f, (same, opposite) in zip(features, overlap_lists(features, features)):
            w.writerow([f.fid, f.symbol, f.locus_tag, f.ftype, f.source, ",".join(f.rfam), f.seqid, f.start,
                        f.end, f.strand, f.length, " ".join(f.product.split()), ",".join(same),
                        ",".join(opposite)])

    with open(paths["rfam_merge_report.tsv"], "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(REPORT_COLUMNS)
        # overlaps are reported against the NCBI genes only, which is what the decision used
        for h, (same, opposite) in zip(hits, overlap_lists(hits, ncbi_genes)):
            w.writerow([h.idx, h.family, h.accession, h.start, h.end, h.strand, h.score, h.evalue, h.fate,
                        h.feature_id, h.duplicate_of, h.reason, ",".join(same), ",".join(opposite)])
    return paths


def build(gff: str, tblout: str, out_dir: str, prefix: str, ncbi_release: str = "") -> dict:
    genes = read_ncbi_genes(gff)
    hits, meta = read_tblout(tblout)
    seqids = {g.seqid for g in genes}
    foreign = sorted({h.seqid for h in hits} - seqids)
    if foreign:
        raise SystemExit(f"{tblout}: hits on sequences that are not in the GFF: {', '.join(foreign)}")
    added = merge_rfam(genes, hits)
    header = [
        f"ecoli_ingest counting annotation, built by build_annotation.py",
        f"ncbi-gff {os.path.basename(gff)} md5={md5_of(gff)}" + (f" release={ncbi_release}" if ncbi_release else ""),
        f"rfam-tblout {os.path.basename(tblout)} md5={md5_of(tblout)}",
        f"rfam-search {meta.get('Program', 'cmscan')} {meta.get('Version', '')}; {meta.get('Option settings', '')}",
        f"rule one row per NCBI gene or pseudogene; nothing removed for nesting or overlap",
        f"rule Rfam hit dropped only if it overlaps a same-strand NCBI {'/'.join(RNA_TYPES)} gene by "
        f">= {DUPLICATE_MIN_FRACTION:.0%} of the shorter feature; the NCBI entry is kept",
        f"rule Rfam hit also dropped if it overlaps an opposite-strand NCBI ncRNA gene by "
        f">= {DUPLICATE_MIN_FRACTION:.0%} of the shorter feature (same locus matched on the wrong strand)",
        "rule Rfam families excluded: " + ", ".join(f"{a} {r}" for a, r in sorted(RFAM_EXCLUDE.items())),
    ]
    paths = write_outputs(genes + added, hits, out_dir, prefix, header, genes)
    fates = defaultdict(int)
    for h in hits:
        fates[h.fate] += 1
    types = defaultdict(int)
    for f in genes:
        types[f.ftype] += 1
    print(f"NCBI features: {len(genes)} (" + ", ".join(f"{k} {v}" for k, v in sorted(types.items())) + ")",
          file=sys.stderr)
    print(f"Rfam hits: {len(hits)} (" + ", ".join(f"{k} {v}" for k, v in sorted(fates.items())) + ")",
          file=sys.stderr)
    print(f"counting features: {len(genes) + len(added)}", file=sys.stderr)
    for path in paths.values():
        print(f"wrote {path}", file=sys.stderr)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gff", required=True, help="unmodified NCBI RefSeq GFF (.gff or .gff.gz)")
    ap.add_argument("--tblout", required=True, help="cmscan --fmt 2 --tblout table of Rfam hits")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True, help="output file prefix, e.g. GCF_000005845.2_ASM584v2_counting_v2")
    ap.add_argument("--ncbi-release", default="", help="free text recorded in the GFF header, e.g. the download date")
    args = ap.parse_args(argv)
    build(args.gff, args.tblout, args.out_dir, args.prefix, args.ncbi_release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
