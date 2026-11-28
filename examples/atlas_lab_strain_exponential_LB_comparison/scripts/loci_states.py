"""States of three loci from BAMs of reads aligned with bowtie2 --local to those regions only (scripts/loci_bam.sh), so
the clipped parts of junction reads are kept:
  IS1H  the reference IS1 ~300 bp upstream of flhD (1978503-1979270)
  IS1I  the reference IS1 inside crl (257908-258675; crl is a pseudogene in the reference because of it)
  IS5U  the reference IS5 between ychE and oppA (1299499-1300693), upstream of the Lrp-repressed opp operon; with
        IS1I a marker of the CGSC 7740 stock of MG1655 (Freddolino et al. 2012). An IS5 there in the other
        orientation (as in W3110) gives no spanning reads in this orientation: no_info or other_copy_reads
  fimS  the fim switch (4542682-4542995, 314 bp with the 9-bp inverted repeats; promoter of fimA)

IS loci. Every clipped part (>= 16 bases) is aligned back to the whole genome (bowtie2 end-to-end) and labelled: an IS
family when it lands in a mobile element, else `genome` with its position (unplaced when it does not align).
  <IS>_span_left / _right  reads aligned through an IS end with >= 10 bases on each side: IS present
  <IS>_absent_reads        reads clipped at an IS end whose clipped part lands on the far flank (the flanks joined,
                           one copy of the 9-bp target duplication left: IS absent)
  <IS>_other_copy_reads    reads clipped at an IS end whose clipped part lands elsewhere (reads of another IS1 copy
                           placed here): no information on this locus
  <IS>_new_IS              clip piles (>= 3 reads) outside the IS ends whose clipped part is an IS: family@pos x n
  <IS>_state, <IS>_reads   present / absent (>= 2 reads; the other kind one read or < 10% of them), conflict,
                           no_info (incl. a single read); the reads
                           behind the state
oppA depth (depth_oppA). flhDC depths (mean over flhC, flhD, the flhD-IS1H gap, the IS1H body, 400 bp beyond IS1H; depth_upTSS 1978410-1978495,
upstream of the flhDC P1 start; upTSS_frac = depth_upTSS / depth_flhD) and crl depths (crl outside IS1I).
fimS. The reference has fimS in one orientation; contig NC_000913_fimS_inverted has it inverted. A read spanning a
switch end (>= 10 bases outside and >= 3 inside beyond the inverted repeat; the orientations differ from the first
base inside) aligns fully only to the orientation it comes from. The fimA transcripts start at the inner repeat, so
RNA-seq gives few such reads.
  The reference orientation is OFF: inverted-orientation reads are the ones at the fimA end (transcripts from the
  fimA promoter inside the switch reading into fimA), reference-orientation reads mostly at the fimE end.
  fim_off_reads / fim_on_reads    spanning reads (both ends added) in the reference (OFF) and inverted (ON)
                                  orientation; fim_on_frac = ON / (ON + OFF), weighted by transcription across the ends
                                  (the ON promoter reads out at the fimA end, the OFF one at the fimE end)
  fim_off_left, ... per end; depth_fimA, depth_fimE
  fim_state  ON / OFF (>= 90% of >= 5 reads), mixed, no_info
    python scripts/loci_states.py --index reference/NC_000913 --gbk reference/NC_000913.3.gbk --out T --junctions J BAM...
"""
import argparse, collections, os, re, subprocess, tempfile
import numpy as np
import pandas as pd

SEQ, INV = "NC_000913", "NC_000913_fimS_inverted"
IS_LOCI = {"IS1H": {"is": (1978503, 1979270), "region": (1975001, 1982000)},
           "IS1I": {"is": (257908, 258675), "region": (254001, 262000)},
           "IS5U": {"is": (1299499, 1300693), "region": (1296001, 1304000)}}
FLHD, FLHC = (1977847, 1978197), (1977266, 1977844)
OPPA = (1301182, 1302813)
UP_TSS = (1978410, 1978495)      # between the flhDC P1 start (~1978395) and the IS1H end: read-through from upstream
CRL = (257829, 259006)
FIM_REGION, FIMS, IR = (4539001, 4548000), (4542682, 4542995), 9
FIMA, FIME = (4543115, 4543663), (4542037, 4542633)
MIN_CLIP, SPAN, MIN_STATE = 16, 10, 5
MIN_IS_READS = 2      # an IS state needs >= 2 reads; a single read gives no_info
FIM_IN = 3     # the two orientations differ from the first base inside each inverted repeat


def mobile_elements(gbk):
    out, pos = [], None
    for line in open(gbk):
        m = re.match(r"^     mobile_element\s+(complement\()?(\d+)\.\.(\d+)", line)
        if m:
            pos = (int(m.group(2)), int(m.group(3)))
            continue
        if pos and "/mobile_element_type=" in line:
            name = line.split(":")[-1].strip().strip('"')
            out.append((re.match(r"IS\d+|IS[A-Z]+", name).group(0), *pos))
            pos = None
    return out


def read_bam(bam, seq, lo, hi):
    """Primary alignments in the region: (pos, end, aligned blocks, clips [(junction pos, side, seq)])"""
    out = subprocess.run(["samtools", "view", "-F", "0x904", bam, f"{seq}:{lo}-{hi}"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        f = line.split("\t")
        pos, cigar, s = int(f[3]), f[5], f[9]
        ops = [(int(n), op) for n, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar)]
        blocks, ref = [], pos
        for n, op in ops:
            if op in "M=X":
                blocks.append((ref, ref + n - 1))
            if op in "MDN=X":
                ref += n
        clips = []
        if ops and ops[0][1] == "S" and ops[0][0] >= MIN_CLIP:
            clips.append((pos, "left", s[:ops[0][0]]))
        if ops and ops[-1][1] == "S" and ops[-1][0] >= MIN_CLIP:
            clips.append((ref - 1, "right", s[len(s) - ops[-1][0]:]))
        yield pos, ref - 1, blocks, clips


def place(seqs, index, threads=4):
    """Genome position of each clipped sequence: {i: (pos, mapq)}; unaligned ones are left out."""
    if not seqs:
        return {}
    with tempfile.TemporaryDirectory() as d:
        fa = f"{d}/c.fa"
        with open(fa, "w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">{i}\n{s}\n")
        out = subprocess.run(["bowtie2", "-f", "-p", str(threads), "--no-hd", "-x", index, "-U", fa],
                             capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.splitlines():
        f = line.split("\t")
        if int(f[1]) & 4:
            continue
        res[int(f[0])] = (int(f[3]), int(f[4]))
    return res


def is_locus(bam, gsm, name, cfg, index, label):
    (L, R), (lo, hi) = cfg["is"], cfg["region"]
    depth = np.zeros(hi - lo + 2)
    span_l = span_r = 0
    clips = []
    for pos, end, blocks, cl in read_bam(bam, SEQ, lo, hi):
        for s, e in blocks:
            s, e = max(s, lo), min(e, hi)
            if s <= e:
                depth[s - lo:e - lo + 1] += 1
        span_l += any(s <= L - SPAN and e >= L + SPAN for s, e in blocks)
        span_r += any(s <= R - SPAN and e >= R + SPAN for s, e in blocks)
        clips += cl
    hits = place([c[2] for c in clips], index)
    near = lambda q, x, w: abs(q - x) <= w
    absent = other = 0
    piles, jrows = collections.Counter(), []
    for i, (jp, side, s) in enumerate(clips):
        if jp < lo + 50 or jp > hi - 50:          # the ends of the region reference
            continue
        lab, gpos = label(hits.get(i))
        end_l = near(jp, L, 12) or near(jp, L - 1, 12)
        end_r = near(jp, R, 12) or near(jp, R + 1, 12)
        at_end = end_l or end_r
        if at_end and lab == "genome":
            far = (end_l and R - 20 <= gpos <= R + 400) or (end_r and L - 400 <= gpos <= L + 20)
            absent += far
            other += not far
        jrows.append({"gsm": gsm, "locus": name, "position": jp, "clip_side": side, "clip_source": lab,
                      "clip_genome_pos": gpos if gpos is not None else -1, "at_IS_end": at_end})
        if lab.startswith("IS") and not at_end and not L <= jp <= R:
            piles[(lab, jp)] += 1
    span = span_l + span_r
    # a minority of one read, or under 10% of the reads (e.g. 1 against 1,984), does not make a conflict
    n = span + absent
    minor = lambda k: k <= 1 or k < 0.1 * n
    state = "present" if minor(absent) and span >= MIN_IS_READS else "absent" if minor(span) and absent >= MIN_IS_READS \
        else "conflict" if not (minor(absent) or minor(span)) else "no_info"
    row = {f"{name}_span_left": span_l, f"{name}_span_right": span_r, f"{name}_absent_reads": absent,
           f"{name}_other_copy_reads": other,
           f"{name}_new_IS": ";".join(f"{l}@{q}x{n}" for (l, q), n in piles.most_common() if n >= 3),
           f"{name}_state": state,
           f"{name}_reads": absent if state == "absent" else span if state == "present" else n}
    d = lambda x, y: float(depth[x - lo:y - lo + 1].mean())
    return row, d, jrows


def fim_switch(bam):
    """Reads spanning a fimS end in each orientation (both contigs use genome coordinates)."""
    (A, B), (lo, hi) = FIMS, FIM_REGION
    ends = {"left": (A - SPAN, A + IR - 1 + FIM_IN), "right": (B - IR + 1 - FIM_IN, B + SPAN)}
    row, depth = {}, np.zeros(hi - lo + 2)
    for tag, seq in (("off", SEQ), ("on", INV)):
        for end in ends:
            row[f"fim_{tag}_{end}"] = 0
        for pos, e, blocks, _ in read_bam(bam, seq, lo, hi):
            for end, (s0, e0) in ends.items():
                row[f"fim_{tag}_{end}"] += any(s <= s0 and e >= e0 for s, e in blocks)
            for s, e in blocks:                 # gene depths outside the switch are the same on both contigs
                s, e = max(s, lo), min(e, hi)
                if s <= e:
                    depth[s - lo:e - lo + 1] += 1
    off, on = row["fim_off_left"] + row["fim_off_right"], row["fim_on_left"] + row["fim_on_right"]
    n = off + on
    row.update({"fim_off_reads": off, "fim_on_reads": on, "fim_on_frac": on / n if n else np.nan,
                "fim_state": "no_info" if n < MIN_STATE else "OFF" if off >= 0.9 * n else "ON" if on >= 0.9 * n
                else "mixed",
                "depth_fimA": float(depth[FIMA[0] - lo:FIMA[1] - lo + 1].mean()),
                "depth_fimE": float(depth[FIME[0] - lo:FIME[1] - lo + 1].mean())})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--gbk", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--junctions", required=True)
    p.add_argument("bams", nargs="+")
    a = p.parse_args()
    mobile = mobile_elements(a.gbk)

    def label(hit):
        if hit is None:
            return "unplaced", None
        pos, _ = hit
        for fam, s, e in mobile:
            if s - 5 <= pos <= e + 5:
                return fam, pos
        return "genome", pos

    rows, jrows = [], []
    for bam in a.bams:
        gsm = os.path.basename(bam).replace(".bam", "")
        row = {"gsm": gsm}
        r1, d1, j1 = is_locus(bam, gsm, "IS1H", IS_LOCI["IS1H"], a.index, label)
        (L, R) = IS_LOCI["IS1H"]["is"]
        row.update({"depth_flhC": d1(*FLHC), "depth_flhD": d1(*FLHD), "depth_gap": d1(FLHD[1] + 1, L - 1),
                    "depth_upTSS": d1(*UP_TSS), "upTSS_frac": d1(*UP_TSS) / max(d1(*FLHD), 1e-9),
                    "depth_IS1H": d1(L + 50, R - 50), "depth_beyond": d1(R + 1, R + 400)})
        row.update(r1)
        r2, d2, j2 = is_locus(bam, gsm, "IS1I", IS_LOCI["IS1I"], a.index, label)
        (L2, R2) = IS_LOCI["IS1I"]["is"]
        row.update({"depth_crl": (d2(CRL[0], L2 - 1) * (L2 - CRL[0]) + d2(R2 + 1, CRL[1]) * (CRL[1] - R2))
                                 / (L2 - CRL[0] + CRL[1] - R2)})
        row.update(r2)
        r3, d3, j3 = is_locus(bam, gsm, "IS5U", IS_LOCI["IS5U"], a.index, label)
        row.update({"depth_oppA": d3(*OPPA)})
        row.update(r3)
        row.update(fim_switch(bam))
        rows.append(row)
        jrows += j1 + j2 + j3
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False, float_format="%.3f")
    cols = ["gsm", "locus", "position", "clip_side", "clip_source", "clip_genome_pos", "at_IS_end"]
    j = pd.DataFrame(jrows, columns=cols)
    j = j.groupby(cols).size().rename("reads").reset_index()
    j.sort_values(["gsm", "locus", "reads"], ascending=[True, True, False]).to_csv(a.junctions, sep="\t", index=False)


if __name__ == "__main__":
    main()
