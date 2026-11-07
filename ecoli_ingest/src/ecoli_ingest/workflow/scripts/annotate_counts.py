#!/usr/bin/env python
"""Annotate a featureCounts table with gene identifiers and product descriptions, fully offline.

Inputs
  --counts      featureCounts output: a '# Program' comment line, then a header
                ``Geneid Chr Start End Strand Length <bam>...`` and one row per feature.
  --gff         the filtered GFF that featureCounts was run with. Every row carries a ``Name``
                attribute, which featureCounts used as ``Geneid``.
  --source-gff  the raw NCBI RefSeq GFF for the same assembly. Its CDS / tRNA / rRNA / ncRNA child
                features carry ``product=`` descriptions keyed by ``locus_tag``.

Outputs
  --extended    Gene_Symbol  Gene_IDs_Field  Gene_Summary  Chr Start End Strand Length <counts...>
  --symbols     Gene_Symbol  Chr Start End Strand Length <counts...>

``Gene_IDs_Field`` is ``locus_tag,<Dbxref...>`` (e.g. ``b0002,ASAP:ABE-0000008,ECOCYC:EG10998,GeneID:945803``),
taken from the gene row of the filtered GFF when there is one. ``Gene_Summary`` is the RefSeq
``product`` of the gene's child feature, falling back to ``Note``, then ``gene_biotype``, then the Rfam
family for Infernal-predicted ncRNAs, and finally ``NA``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from urllib.parse import unquote


@dataclass
class Feature:
    seqid: str
    source: str
    ftype: str
    attrs: dict = field(default_factory=dict)


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


def read_gff(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue
            yield Feature(cols[0], cols[1], cols[2], parse_attributes(cols[8]))


def build_id_maps(filtered_gff: str) -> tuple[dict, dict]:
    """Return (name -> 'locus_tag,Dbxref...', name -> attributes) from the featureCounts GFF.

    Gene rows (``gbkey=Gene``) take priority over any other row with the same Name.
    """
    ids: dict[str, str] = {}
    info: dict[str, dict] = {}
    for feat in read_gff(filtered_gff):
        name = feat.attrs.get("Name")
        if not name:
            continue
        is_gene = feat.attrs.get("gbkey") == "Gene"
        if name in info and not is_gene:
            continue
        parts = [feat.attrs[k] for k in ("locus_tag", "Dbxref") if feat.attrs.get(k)]
        if parts:
            ids[name] = ",".join(parts)
        info[name] = feat.attrs
    return ids, info


def build_product_maps(source_gff: str) -> tuple[dict, dict]:
    """Return (locus_tag -> product, locus_tag -> Note) from the raw RefSeq GFF."""
    product: dict[str, str] = {}
    note: dict[str, str] = {}
    for feat in read_gff(source_gff):
        locus = feat.attrs.get("locus_tag")
        if not locus:
            continue
        if "product" in feat.attrs and locus not in product:
            product[locus] = feat.attrs["product"]
        if "Note" in feat.attrs and locus not in note:
            note[locus] = feat.attrs["Note"]
    return product, note


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def gene_summary(name: str, attrs: dict, product: dict, note: dict, with_synonyms: bool) -> str:
    locus = attrs.get("locus_tag")
    summary = None
    if locus:
        summary = product.get(locus) or note.get(locus)
    if not summary:
        summary = attrs.get("product") or attrs.get("Note")
    if not summary and attrs.get("gene_biotype") and attrs["gene_biotype"] != "protein_coding":
        summary = attrs["gene_biotype"]
    if not summary:
        rfam = [x for x in attrs.get("Dbxref", "").split(",") if x.startswith("Rfam:")]
        if rfam:
            family = attrs.get("ID", name)
            summary = f"Rfam ncRNA family {family} ({rfam[0].split(':', 1)[1]})"
    if not summary:
        return "NA"
    summary = clean_text(summary)
    if with_synonyms and attrs.get("gene_synonym"):
        summary += f" [synonyms: {clean_text(attrs['gene_synonym'])}]"
    return summary


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


def annotate(counts: str, gff: str, source_gff: str, extended_out: str, symbols_out: str,
             with_synonyms: bool = False) -> int:
    header, rows = read_counts(counts)
    ids, info = build_id_maps(gff)
    product, note = build_product_maps(source_gff)

    n_ids = n_summary = 0
    with open(extended_out, "w", encoding="utf-8", newline="") as ext_fh, \
            open(symbols_out, "w", encoding="utf-8", newline="") as sym_fh:
        ext = csv.writer(ext_fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        sym = csv.writer(sym_fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        ext.writerow(["Gene_Symbol", "Gene_IDs_Field", "Gene_Summary"] + header[1:])
        sym.writerow(["Gene_Symbol"] + header[1:])
        for row in rows:
            name = row[0]
            attrs = info.get(name, {})
            gene_ids = ids.get(name, name)
            summary = gene_summary(name, attrs, product, note, with_synonyms)
            n_ids += name in ids
            n_summary += summary != "NA"
            ext.writerow([name, gene_ids, summary] + row[1:])
            sym.writerow([name] + row[1:])

    print(f"annotate_counts: {len(rows)} features; identifiers for {n_ids}; "
          f"product descriptions for {n_summary}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="featureCounts output table")
    ap.add_argument("--gff", required=True, help="filtered GFF used by featureCounts (rows have Name)")
    ap.add_argument("--source-gff", required=True, help="raw NCBI RefSeq GFF with product= attributes")
    ap.add_argument("--extended", required=True, help="output: extended annotated table")
    ap.add_argument("--symbols", required=True, help="output: Gene_Symbol + coordinates + counts")
    ap.add_argument("--with-synonyms", action="store_true", help="append gene synonyms to Gene_Summary")
    args = ap.parse_args(argv)
    return annotate(args.counts, args.gff, args.source_gff, args.extended, args.symbols, args.with_synonyms)


if __name__ == "__main__":
    raise SystemExit(main())
