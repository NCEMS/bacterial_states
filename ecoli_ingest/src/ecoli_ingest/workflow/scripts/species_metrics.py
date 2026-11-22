#!/usr/bin/env python
"""Is the library E. coli? Species statistics from Centrifuge's per-read classification table.

The workflow's older check (`<sample>_report.txt`: share of the species E. coli among reads with a
single match) splits E. coli reads between Escherichia and Shigella genomes and counts synthetic
constructs against the sample, so real E. coli libraries can score under 20 %. This script uses
the same Centrifuge output differently. Among reads with a single match that resolves to a genus
(synthetic constructs left out):

  species_reads             number of such reads
  pct_escherichia_shigella  share in Escherichia or Shigella (the two cannot be told apart)
  top_other_genus, pct_top_other_genus   the largest other genus and its share

Each is reported for all reads and, with `--bam` and `--bed`, again without the reads that lie on
rRNA / tRNA genes (`*_nostable`): in libraries that are mostly stable RNA, rRNA reads are assigned
to Salmonella and other enteric genera and pull the share down. From the all-reads report
(`centrifuge-kreport`, multi-match reads at their lowest common ancestor) it also gives
pct_unclassified, pct_enterobacteriaceae, pct_synthetic and top_non_enteric.

    species_metrics.py --classifications initial.tsv --index <centrifuge index> \
        --bam sample_sort.bam --bed counting.bed --report-all sample_report_all.txt --out sample_species.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

ENTEROBACTERIACEAE, SYNTHETIC = "543", "32630"
ECOLI_GENERA = ("Escherichia", "Shigella")
# Bacteriophage genera (Centrifuge names). A sample whose largest other genus is one of these is a
# phage infection or induction experiment and is refused. Not listed on purpose: Phix174microvirus
# (the Illumina PhiX spike-in) and viruses of plants and animals (e.g. Begomovirus, which matches
# parts of some engineered plasmids).
PHAGE_GENERA = frozenset({
    "Lambdavirus", "T7virus", "T4virus", "T5virus", "T1virus", "Rb69virus", "P1virus", "P2virus", "Muvirus",
    "Mulikevirus", "N4virus", "N15virus", "Hk97virus", "Levivirus", "Allolevivirus", "Inovirus", "Epsilon15virus",
    "P22virus", "Phikzvirus", "Felixo1virus", "Sp6virus", "T3virus", "Phieco32virus", "Rtpvirus", "Tunavirus",
})
# Thresholds of the species gate, set on the prescreen of 10,829 atlas candidates (docs/prescreen.md).
# "of all reads" = % of the reads given to Centrifuge.
SPECIES_THRESHOLDS = {
    "min_pct_escherichia_shigella": 50.0,   # share of genus-resolved reads in Escherichia / Shigella ...
    "min_pct_foreign_of_all": 2.0,          # ... below it: not E. coli, if the largest other genus is >= this % of all reads
    "max_pct_foreign_of_all": 10.0,         # largest other genus above this % of all reads: mixed sample
    "min_pct_phage_of_all": 0.5,            # largest other genus is a phage at this % of all reads: phage sample
    "min_species_reads": 100,               # fewer genus-resolved reads than this: no decision
    # early stop, before alignment, on the all-reads values: only samples that are clearly not E. coli
    "early_max_pct_escherichia_shigella": 20.0,
    "early_min_pct_foreign_of_all": 5.0,
}


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def genus_of_taxa(kreport_path: str) -> dict:
    """taxid -> genus name (None above genus level), read from the indented tree of a Centrifuge
    report. The report must be the all-reads one so that every taxid in the classifications is in it."""
    out, stack = {}, []
    with open(kreport_path, encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            depth = len(cols[5]) - len(cols[5].lstrip(" "))
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, cols[3], cols[5].strip()))
            out[cols[4]] = next((name for _, rank, name in stack if rank == "G"), None)
    return out


def species_metrics(classifications_path: str, kreport_all_path: str, skip_ids=None, suffix: str = "") -> dict:
    """The four species values (see the module text). `skip_ids` is any container of read ids
    supporting `in` (a set, or a HashedIds for a whole run); those reads are left out and the keys
    get `suffix`."""
    genus = genus_of_taxa(kreport_all_path)
    ecoli = 0
    other: dict[str, int] = {}
    with open(classifications_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8 or cols[2] == "0" or cols[7] != "1" or cols[2] == SYNTHETIC:
                continue
            g = genus.get(cols[2])
            if g is None:
                continue
            if skip_ids is not None and cols[0] in skip_ids:      # last: the costly test, on few rows
                continue
            if g in ECOLI_GENERA:
                ecoli += 1
            else:
                other[g] = other.get(g, 0) + 1
    n = ecoli + sum(other.values())
    top = max(other.items(), key=lambda kv: kv[1]) if other else ("", 0)
    return {"species_reads" + suffix: n,
            "pct_escherichia_shigella" + suffix: _round(100 * ecoli / n, 2) if n else None,
            "top_other_genus" + suffix: top[0],
            "pct_top_other_genus" + suffix: _round(100 * top[1] / n, 2) if n else None}


def kreport_all_metrics(path: str) -> dict:
    """From a Centrifuge report made from every read (multi-match reads at their lowest common
    ancestor): pct_unclassified of all reads; of classified reads, pct_enterobacteriaceae (the family
    of E. coli, taxid 543) and pct_synthetic (vectors, adapters); top_non_enteric, the largest
    species outside the family."""
    unclassified = root = family = synthetic = 0
    inside = False
    family_depth = None
    species = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            clade, taxid, name = int(cols[1]), cols[4], cols[5]
            depth = len(name) - len(name.lstrip(" "))
            if taxid == "0":
                unclassified = clade
            elif taxid == "1":
                root = clade
            if inside and depth <= family_depth:
                inside = False
            if taxid == ENTEROBACTERIACEAE:
                family, inside, family_depth = clade, True, depth
            elif taxid == SYNTHETIC:
                synthetic = clade
            elif cols[3] == "S" and not inside:
                species.append((clade, name.strip()))
    total = unclassified + root
    species.sort(reverse=True)
    top = f"{species[0][1]} {100 * species[0][0] / root:.1f}" if species and root else ""
    return {"pct_unclassified": _round(100 * unclassified / total, 2) if total else None,
            "pct_enterobacteriaceae": _round(100 * family / root, 2) if root else None,
            "pct_synthetic": _round(100 * synthetic / root, 2) if root else None,
            "top_non_enteric": top}


def _number(metrics: dict, key: str):
    value = metrics.get(key)
    return None if value in (None, "") else float(value)


def species_verdict(metrics: dict, thresholds: dict | None = None, early: bool = False) -> str:
    """'ok', 'no_species' (too few genus-resolved reads to decide), 'phage', 'not_ecoli' or
    'mixed_organism', from the values of `species_metrics`.

    The values without reads on rRNA / tRNA genes (`*_nostable`) are used when they are present.
    A low Escherichia/Shigella share alone is not enough: libraries of short reads leave few
    classifiable reads, spread over many genera, so the largest other genus must also be a real
    part of the library (% of all reads given to Centrifuge).

    `early=True` is the check made before alignment, on the all-reads values: it only reports
    samples that are clearly something else ('phage', 'not_ecoli'), so that they are not aligned
    (vg giraffe is very slow on reads that do not match the graph). On the prescreen it stopped 362
    of the 547 samples the full rule refuses and 1 of the 10,125 it accepts.
    """
    t = {**SPECIES_THRESHOLDS, **(thresholds or {})}
    sfx = "" if early or _number(metrics, "species_reads_nostable") is None else "_nostable"
    n, share, top = (_number(metrics, k + sfx) for k in ("species_reads", "pct_escherichia_shigella", "pct_top_other_genus"))
    n_all = _number(metrics, "centrifuge_reads")
    if n is None or share is None or n < t["min_species_reads"]:
        return "no_species"
    foreign = (top or 0) * n / n_all if n_all else (top or 0)
    if metrics.get("top_other_genus" + sfx) in PHAGE_GENERA and foreign >= t["min_pct_phage_of_all"]:
        return "phage"
    if early:
        return "not_ecoli" if (share < t["early_max_pct_escherichia_shigella"]
                               and foreign >= t["early_min_pct_foreign_of_all"]) else "ok"
    if share < t["min_pct_escherichia_shigella"] and foreign >= t["min_pct_foreign_of_all"]:
        return "not_ecoli"
    if foreign >= t["max_pct_foreign_of_all"]:
        return "mixed_organism"
    return "ok"


class HashedIds:
    """A set of read ids held as sorted 64-bit hashes, for runs with tens of millions of reads
    (about 8 bytes per id). Membership can report a false positive once in about 10^12 tests."""

    def __init__(self, ids):
        import numpy as np
        self._np = np
        self.hashes = np.unique(np.fromiter((hash(i) for i in ids), dtype=np.int64))

    def __len__(self) -> int:
        return len(self.hashes)

    def __contains__(self, read_id) -> bool:
        h = hash(read_id)
        i = self._np.searchsorted(self.hashes, h)
        return bool(i < len(self.hashes) and self.hashes[i] == h)


def stable_read_ids(bam: str, bed: str) -> HashedIds:
    """Ids of the reads that overlap an rRNA or tRNA feature of the counting BED (names `rRNA:...`, `tRNA:...`)."""
    cmd = (f"bedtools intersect -u -a '{bam}' -b <(awk -F'\\t' '$4 ~ /^(rRNA|tRNA):/' '{bed}') "
           "| samtools view - | cut -f1")
    proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE, text=True)
    ids = HashedIds(line.rstrip("\n") for line in proc.stdout)
    if proc.wait() != 0:
        raise SystemExit("species_metrics: listing the reads on rRNA / tRNA genes failed")
    return ids


def n_classified_input(classifications_path: str) -> int:
    """Number of distinct reads in the classification table (a read with several matches has several rows)."""
    n, last = 0, None
    with open(classifications_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            rid = line.split("\t", 1)[0]
            if rid != last:
                n, last = n + 1, rid
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifications", required=True, help="table written by `centrifuge -S` for all reads")
    ap.add_argument("--index", help="Centrifuge index prefix; needed unless --report-all already exists")
    ap.add_argument("--thresholds", default="{}", help="JSON object overriding SPECIES_THRESHOLDS")
    ap.add_argument("--report-all", required=True, help="all-reads centrifuge-kreport; written if --index is given")
    ap.add_argument("--bam", help="sorted alignments; with --bed gives the *_nostable values")
    ap.add_argument("--bed", help="counting annotation BED")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.index:
        with open(args.report_all, "w", encoding="utf-8") as fh:
            res = subprocess.run(["centrifuge-kreport", "-x", args.index, args.classifications], stdout=fh,
                                 stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            raise SystemExit("species_metrics: centrifuge-kreport failed")
    out = {"centrifuge_reads": n_classified_input(args.classifications)}
    out.update(kreport_all_metrics(args.report_all))
    out.update(species_metrics(args.classifications, args.report_all))
    thresholds = json.loads(args.thresholds or "{}")
    if args.bam and args.bed:
        stable = stable_read_ids(args.bam, args.bed)
        out["stable_rna_reads"] = len(stable)
        out.update(species_metrics(args.classifications, args.report_all, stable, "_nostable"))
        out["verdict"] = species_verdict(out, thresholds)
    else:
        out["verdict_early"] = species_verdict(out, thresholds, early=True)
    out["thresholds"] = {**SPECIES_THRESHOLDS, **thresholds}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"species: {out['species_reads']} genus-resolved reads, {out['pct_escherichia_shigella']} % Escherichia/Shigella, "
          f"largest other genus {out['top_other_genus'] or 'none'} {out['pct_top_other_genus']} %; "
          f"{out.get('verdict') or out.get('verdict_early')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
