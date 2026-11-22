#!/usr/bin/env python
"""Make technical twins of a sample: the same reads with a changed sequencing property.

A twin is biologically identical to its source by construction, so it shows what read length,
layout, depth or DNA contamination alone do to counts, QC values and the atlas embedding
(docs/atlas_qc_inputs.md, "Technical twins"). Twins are made from the raw reads and then run
through the unchanged workflow under the name <sample>__<variant>.

Variants
  t<N>      both mates truncated to N bases                      t36, t50, t75
  se        mate 2 dropped
  se_t<N>   mate 2 dropped and mate 1 truncated to N bases
  d<N>      random subsample of N reads (pairs); k and M allowed  d250k, d1M
  g<P>      simulated genomic reads added so that they make up P % of the reads: fragments drawn
            evenly along the genome on both strands, with the source's read length and a given
            fragment-size distribution                            g1, g5, g20

    make_twins.py --r1 a_1.fastq.gz --r2 a_2.fastq.gz --name GSM1 --variants t50 se d1M g5 \
        --genome NC_000913.3.fasta --out-dir twins/
writes twins/GSM1__t50_R1.fastq (and _R2) and so on, and twins/GSM1__twins.json with what was made.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import subprocess
import sys

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def open_fastq(path: str):
    """Text stream of a FASTQ file; .gz is read through pigz/gzip when available (much faster)."""
    if path.endswith((".gz", ".gzip")):
        for tool in ("pigz", "gzip"):
            try:
                proc = subprocess.Popen([tool, "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
                return proc.stdout
            except FileNotFoundError:
                continue
        return gzip.open(path, "rt")
    return open(path, encoding="utf-8")


def records(path: str):
    with open_fastq(path) as fh:
        while True:
            name = fh.readline()
            if not name:
                return
            seq, plus, qual = fh.readline(), fh.readline(), fh.readline()
            yield name.rstrip("\n"), seq.rstrip("\n"), qual.rstrip("\n")


def count_reads(path: str) -> int:
    return sum(1 for _ in records(path))


def parse_variant(variant: str) -> dict:
    """{'truncate': N|None, 'single_end': bool, 'depth': N|None, 'dna_pct': P|None}"""
    out = {"truncate": None, "single_end": False, "depth": None, "dna_pct": None}
    m = re.fullmatch(r"(se)?_?(?:t(\d+))?", variant)
    if variant and m and (m.group(1) or m.group(2)):
        out["single_end"] = bool(m.group(1))
        out["truncate"] = int(m.group(2)) if m.group(2) else None
        return out
    m = re.fullmatch(r"d(\d+(?:\.\d+)?)([kKmM]?)", variant)
    if m:
        out["depth"] = int(float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6}[m.group(2).lower()])
        return out
    m = re.fullmatch(r"g(\d+(?:\.\d+)?)", variant)
    if m:
        out["dna_pct"] = float(m.group(1))
        return out
    raise ValueError(f"unknown variant {variant!r} (t<N>, se, se_t<N>, d<N>[k|M], g<percent>)")


def read_genome(fasta: str) -> str:
    """The first sequence of a FASTA file, upper case."""
    seq, started = [], False
    with (gzip.open(fasta, "rt") if fasta.endswith(".gz") else open(fasta, encoding="utf-8")) as fh:
        for line in fh:
            if line.startswith(">"):
                if started:
                    break
                started = True
            elif started:
                seq.append(line.strip())
    return "".join(seq).upper()


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def simulate_genomic(genome: str, n: int, read_length: int, fragment_sizes: list, paired: bool, rng: random.Random,
                     prefix: str = "SIMDNA"):
    """n simulated reads (pairs) from a circular genome: fragment start uniform, strand random,
    fragment size drawn from `fragment_sizes` ([[size, weight], ...]); mate 1 is the start of the
    fragment, mate 2 the reverse complement of its end. Yields (name, seq1, seq2 or None)."""
    sizes = [int(s) for s, _ in fragment_sizes]
    weights = [float(w) for _, w in fragment_sizes]
    doubled = genome + genome[:max(sizes) + 1]
    for i, size in enumerate(rng.choices(sizes, weights, k=n), 1):
        start = rng.randrange(len(genome))
        frag = doubled[start:start + size]
        if rng.random() < 0.5:
            frag = revcomp(frag)
        yield f"{prefix}.{i}", frag[:read_length], (revcomp(frag)[:read_length] if paired else None)


def make_twin(r1: str, r2: str | None, out_prefix: str, variant: str, seed: int = 1, n_reads: int | None = None,
              genome: str | None = None, fragment_sizes: list | None = None, read_length: int | None = None) -> dict:
    """Write <out_prefix>_R1.fastq (and _R2.fastq) for one variant; return what was done."""
    v = parse_variant(variant)
    rng = random.Random(seed)
    paired_out = r2 is not None and not v["single_end"]
    keep_p = 1.0
    if v["depth"] is not None:
        n_reads = n_reads if n_reads is not None else count_reads(r1)
        keep_p = min(1.0, v["depth"] / n_reads) if n_reads else 1.0
    info = {"variant": variant, "seed": seed, "r1": out_prefix + "_R1.fastq",
            "r2": out_prefix + "_R2.fastq" if paired_out else None, "reads": 0, "added_genomic_reads": 0}
    out1 = open(info["r1"], "w", encoding="utf-8")
    out2 = open(info["r2"], "w", encoding="utf-8") if paired_out else None
    cut = v["truncate"]
    longest = 0
    try:
        mates = zip(records(r1), records(r2)) if r2 else ((a, None) for a in records(r1))
        for a, b in mates:
            if keep_p < 1.0 and rng.random() >= keep_p:
                continue
            info["reads"] += 1
            longest = max(longest, len(a[1]))
            out1.write(f"{a[0]}\n{a[1][:cut]}\n+\n{a[2][:cut]}\n")
            if out2:
                out2.write(f"{b[0]}\n{b[1][:cut]}\n+\n{b[2][:cut]}\n")
        if v["dna_pct"]:
            if not genome or not fragment_sizes:
                raise ValueError("a g<percent> variant needs --genome and a fragment-size distribution")
            frac = v["dna_pct"] / 100
            n_dna = round(info["reads"] * frac / (1 - frac))
            length = read_length or longest
            for name, s1, s2 in simulate_genomic(genome, n_dna, length, fragment_sizes, paired_out, rng):
                out1.write(f"@{name} sim/1\n{s1}\n+\n{'I' * len(s1)}\n")
                if out2:
                    out2.write(f"@{name} sim/2\n{s2}\n+\n{'I' * len(s2)}\n")
            info.update(added_genomic_reads=n_dna, genomic_read_length=length,
                        genomic_fraction_of_reads=n_dna / (n_dna + info["reads"]) if n_dna else 0.0)
    finally:
        out1.close()
        if out2:
            out2.close()
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", default=None)
    ap.add_argument("--name", required=True, help="sample name; twins are <name>__<variant>")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--genome", help="FASTA of the genome, for g<percent> variants")
    ap.add_argument("--fragment-sizes", help="JSON [[size, weight], ...] or a sample QC record (.qc.json, its "
                                             "fragment.tlen_hist is used); default: 200 bp")
    ap.add_argument("--read-length", type=int, default=None, help="length of the simulated reads (default: longest read)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    sizes = [[200, 1]]
    if args.fragment_sizes:
        with open(args.fragment_sizes, encoding="utf-8") as fh:
            data = json.load(fh)
        sizes = data["fragment"]["tlen_hist"] if isinstance(data, dict) else data
        sizes = [[s, w] for s, w in sizes if 30 <= int(s) <= 1000] or [[200, 1]]
    genome = read_genome(args.genome) if args.genome else None
    n_reads = count_reads(args.r1) if any(v.startswith("d") for v in args.variants) else None
    made = []
    for variant in args.variants:
        info = make_twin(args.r1, args.r2, os.path.join(args.out_dir, f"{args.name}__{variant}"), variant, args.seed,
                         n_reads, genome, sizes, args.read_length)
        made.append(info)
        print(f"{args.name}__{variant}: {info['reads']} reads" + (f" + {info['added_genomic_reads']} genomic"
              if info["added_genomic_reads"] else ""), file=sys.stderr)
    with open(os.path.join(args.out_dir, f"{args.name}__twins.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": {"r1": args.r1, "r2": args.r2}, "twins": made}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
