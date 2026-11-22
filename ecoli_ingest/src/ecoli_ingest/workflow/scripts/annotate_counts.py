#!/usr/bin/env python
"""Annotate a featureCounts table with feature identifiers and product descriptions, fully offline.

Inputs
  --counts   featureCounts output: a '# Program' comment line, then a header
             ``Geneid Chr Start End Strand Length <bam>...`` and one row per feature.
  --gff      the counting annotation featureCounts was run with (made by build_annotation.py).
             Every row has ``ID`` (the featureCounts key) and, where known, ``symbol``,
             ``locus_tag``, ``feature_type``, ``source``, ``rfam``, ``Dbxref`` and ``product``.

Outputs
  --extended  Feature_ID Symbol Locus_Tag Type Source Rfam Dbxref Product Chr Start End Strand Length <counts...>
  --counts-out  Feature_ID Chr Start End Strand Length <counts...>

Feature ids are ``<Type>_<locus_tag>_<symbol>`` for NCBI genes (Type = CDS, ncRNA, tRNA, rRNA,
pseudo) and ``ncRNA_<Rfam id>`` for Rfam hits. Empty fields are written as ``NA``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from urllib.parse import unquote

ID_COLUMNS = ["Feature_ID", "Symbol", "Locus_Tag", "Type", "Source", "Rfam", "Dbxref", "Product"]
_ATTRIBUTE_OF = {"Symbol": "symbol", "Locus_Tag": "locus_tag", "Type": "feature_type", "Source": "source",
                 "Rfam": "rfam", "Dbxref": "Dbxref", "Product": "product"}


def parse_attributes(text: str) -> dict:
    """Parse a GFF3 column-9 string into a dict. Values are percent-decoded."""
    attrs = {}
    for item in text.strip().split(";"):
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep:
            continue
        attrs[key.strip()] = unquote(value.strip())
    return attrs


def read_features(gff: str) -> dict:
    """feature id -> attributes of its row in the counting annotation."""
    features: dict[str, dict] = {}
    with open(gff, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue
            attrs = parse_attributes(cols[8])
            fid = attrs.get("ID")
            if not fid:
                raise SystemExit(f"{gff}: row without an ID attribute: {cols[8][:80]}")
            if fid in features:
                raise SystemExit(f"{gff}: feature id {fid} occurs more than once")
            features[fid] = attrs
    return features


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def read_counts(path: str) -> tuple[list[str], list[list[str]]]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
            else:
                rows.append(cols)
    if header is None:
        raise SystemExit(f"{path}: no header line found")
    if header[0] != "Geneid":
        raise SystemExit(f"{path}: expected first column 'Geneid', found {header[0]!r}")
    return header, rows


def annotate(counts: str, gff: str, extended_out: str, counts_out: str) -> int:
    header, rows = read_counts(counts)
    features = read_features(gff)
    unknown = [row[0] for row in rows if row[0] not in features]
    if unknown:
        raise SystemExit(f"{counts}: {len(unknown)} features are not in {gff} (e.g. {', '.join(unknown[:5])}); "
                         "the count table was made with a different annotation")

    with open(extended_out, "w", encoding="utf-8", newline="") as ext_fh, \
            open(counts_out, "w", encoding="utf-8", newline="") as cnt_fh:
        ext = csv.writer(ext_fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        cnt = csv.writer(cnt_fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        ext.writerow(ID_COLUMNS + header[1:])
        cnt.writerow(["Feature_ID"] + header[1:])
        for row in rows:
            attrs = features[row[0]]
            ids = [clean_text(attrs.get(_ATTRIBUTE_OF[c], "")) or "NA" for c in ID_COLUMNS[1:]]
            ext.writerow([row[0]] + ids + row[1:])
            cnt.writerow(row)

    print(f"annotate_counts: {len(rows)} features", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="featureCounts output table")
    ap.add_argument("--gff", required=True, help="counting annotation used by featureCounts")
    ap.add_argument("--extended", required=True, help="output: annotated table")
    ap.add_argument("--counts-out", required=True, help="output: Feature_ID + coordinates + counts")
    args = ap.parse_args(argv)
    return annotate(args.counts, args.gff, args.extended, args.counts_out)


if __name__ == "__main__":
    raise SystemExit(main())
