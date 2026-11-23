"""Build the GO gene sets that `ecoli_atlas` ships for gene-set enrichment (`ecoli_atlas.gsea`).

    python ecoli_atlas/scripts/go_gene_sets.py --gaf ECOLI-mod.gaf.gz --obo go-basic.obo \
        --out ecoli_atlas/src/ecoli_atlas/reference/go_gene_sets_ecocyc.tsv.gz

Sources (Gene Ontology Consortium, CC BY 4.0):
    https://current.geneontology.org/annotations/gaf/ECOLI-mod.gaf.gz   (EcoCyc annotations of E. coli K-12)
    https://current.geneontology.org/ontology/go-basic.obo

One row per GO term that has at least one E. coli gene: term, namespace, name, number of genes and the
genes as lower-case symbols (GAF column 3; annotations with a NOT qualifier are left out). The table
holds the direct annotations, as the analysis this reproduces did (no propagation to parent terms).
The header lines (`#`) record the GAF and ontology versions; rebuild with newer files to update.
"""
from __future__ import annotations

import argparse
import gzip
from collections import defaultdict


def read_gaf(path: str) -> tuple[dict, dict]:
    sets, header = defaultdict(set), {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                key, _, value = line[1:].partition(":")
                if key.strip() in ("gaf-version", "date-generated", "generated-by"):
                    header[key.strip()] = value.strip()
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 17 or "NOT" in f[3].split("|"):
                continue
            sets[f[4]].add(f[2].lower())
    return sets, header


def read_obo(path: str) -> tuple[dict, str]:
    terms, cur, version = {}, None, ""
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("data-version:") and not version:
                version = line.split(":", 1)[1].strip()
            elif line == "[Term]":
                cur = {}
            elif line.startswith("[") and line.endswith("]"):
                cur = None
            elif cur is not None and line.startswith("id: "):
                cur["id"] = line[4:]
                terms[cur["id"]] = cur
            elif cur is not None and line.startswith(("name: ", "namespace: ")):
                key, _, value = line.partition(": ")
                cur[key] = value
    return terms, version


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gaf", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets, gaf_header = read_gaf(a.gaf)
    terms, go_version = read_obo(a.obo)
    rows = []
    for term in sorted(sets):
        info = terms.get(term, {})
        rows.append((term, info.get("namespace", ""), info.get("name", ""), str(len(sets[term])), ",".join(sorted(sets[term]))))
    with gzip.open(a.out, "wt") as out:
        out.write(f"# GO gene sets of E. coli K-12 for ecoli_atlas.gsea; built by scripts/go_gene_sets.py\n")
        out.write(f"# annotations: ECOLI-mod.gaf (EcoCyc), date-generated {gaf_header.get('date-generated', '?')}\n")
        out.write(f"# ontology: go-basic.obo, data-version {go_version or '?'}\n")
        out.write("# Gene Ontology Consortium data, CC BY 4.0 (geneontology.org)\n")
        out.write("term\tnamespace\tname\tn_genes\tgenes\n")
        for r in rows:
            out.write("\t".join(r) + "\n")
    missing = sum(1 for r in rows if not r[2])
    print(f"{len(rows)} GO terms ({missing} without a name in the ontology) -> {a.out}")


if __name__ == "__main__":
    main()
