#!/usr/bin/env python3
"""Growth-curve time courses of the atlas studies: atlas_data/time_courses.tsv.

One series is one culture line followed along its growth curve (a replicate, a strain), with its
samples' position along the curve as time, OD or time-point order. Only untreated batch growth
curves are listed. Series after an induction, a shift, an infection or a starvation are left out,
as are fed-batch cultures (GSE221706, GSE252784), since there time is not growth progress.

Used by `ecoli_atlas train --time-series` (ordered pairs inside a series) and
`ecoli_atlas axes --series`. Columns: sample, series, x, x_label. Only the order of x inside a
series is used; the units differ between studies.

The values come from the per-sample fields of the GEO digests (geo_digest.py), read from
DIGEST_DIR (default: work/metadata_v2/digest). Run from the repository root:
  python ecoli_ingest/atlas_build/metadata/time_courses.py [out.tsv]
"""
import csv
import os
import re
import sys

DIGEST_DIR = os.environ.get("DIGEST_DIR", "work/metadata_v2/digest")


def table(gse):
    text = open(os.path.join(DIGEST_DIR, f"{gse}.md"), encoding="utf-8").read().split("## Differs between samples")[1].strip().split("\n")
    header = text[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in text[1:] if line.startswith("GSM")]


def rows():
    for r in table("GSE197541"):                       # planktonic_7_62h_r1
        m = re.match(r"planktonic_(\d+)_(\d+)h_r(\d)", r["title"])
        if m:
            yield r["GSM"], f"GSE197541 replicate {m.group(3)}", float(f"{m.group(1)}.{m.group(2)}"), "hours after subculture"
    for r in table("GSE67402"):                        # "3 hours", "... biological replicate 2[, rRNA not depleted]"
        m, rep = re.match(r"(\d+) hours", r["ch:time point"]), re.search(r"replicate (\d)", r["title"])
        if m and rep:
            kind = " (rRNA not depleted)" if "not depleted" in r["title"] else ""
            yield r["GSM"], f"GSE67402 replicate {rep.group(1)}{kind}", float(m.group(1)), "hours after inoculation"
    for r in table("GSE152619"):                       # "4 hours", LB4a
        m, t = re.match(r"(\d+) hours", r["ch:time incubated"]), re.match(r"([A-Za-z]+)\d+([a-z])$", r["title"])
        if m and t:
            yield r["GSM"], f"GSE152619 {t.group(1)} culture {t.group(2)}", float(m.group(1)), "hours incubated"
    for r in table("GSE224463"):                       # "wild-type, stationary phase, 120 min, replicate SP_A" (titles; the fields disagree with them)
        p = [x.strip() for x in r["title"].split(",")]
        m = re.match(r"(\d+)\s*min", p[2]) if len(p) == 4 else None
        if m and "stationary" in p[1]:
            yield r["GSM"], f"GSE224463 {p[0]} {p[3]}", float(m.group(1)), "minutes"
    for r in table("GSE288896"):                       # "E. coli, 6 hr, Untreated control, Rep 1"
        p = [x.strip() for x in r["title"].split(",")]
        m = re.match(r"(\d+) hr", p[1]) if len(p) == 4 else None
        if m and p[2] == "Untreated control":
            yield r["GSM"], f"GSE288896 {p[3]}", float(m.group(1)), "hours"
    order = {"CAS": ["P1", "P2", "P3", "S1", "S2"], "GLU": ["L", "S1", "S2"]}
    for r in table("GSE271069"):                       # "CAS condition, timepoint P1, RNA-seq, replicate 1"
        m = re.match(r"(CAS|GLU) condition, timepoint (\w+), RNA-seq, replicate (\d)", r["title"])
        if m and m.group(2) in order[m.group(1)]:
            yield r["GSM"], f"GSE271069 {m.group(1)} replicate {m.group(3)}", float(order[m.group(1)].index(m.group(2)) + 1), "time point (in order)"
    for r in table("GSE206047"):                       # uninduced samples only, one series per strain
        if r.get("sup:IPTG") == "none" and r.get("sup:arabinose") == "none" and r.get("sup:timepoint_h"):
            yield r["GSM"], f"GSE206047 {r['sup:strain']}", float(r["sup:timepoint_h"]), "hours"
    for r in table("GSE123048"):                       # "First Replicate RNA-seq sample OD600=0.4"; LB, OD 0.4 to 4.0
        m = re.match(r"(First|Second) Replicate\s+RNA-seq sample OD600=([\d.]+)", r["title"])
        if m:
            yield r["GSM"], f"GSE123048 {m.group(1).lower()} replicate", float(m.group(2)), "OD600"
    for r in table("GSE226643"):                       # "... grown in a minimal medium at 3 h, replicate 1"; not the N-/S-starvation series
        m = re.match(r"E\. coli K-12 MG1655 grown in a minimal medium at (\d+) h, replicate (\d)$", r["title"])
        if m:
            yield r["GSM"], f"GSE226643 M9 glucose replicate {m.group(2)}", float(m.group(1)), "hours"
    for r in table("GSE64021"):                        # "LB medium OD600=0.5"; not the heat-shock, starvation or MOPS-transfer series
        m = re.match(r"LB medium OD600=([\d.]+)", r.get("ch:condition", ""))
        if m:
            yield r["GSM"], "GSE64021 LB", float(m.group(1)), "OD600"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "atlas_data/time_courses.tsv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["sample", "series", "x", "x_label"])
        n, series = 0, set()
        for row in rows():
            w.writerow(row)
            n += 1
            series.add(row[1])
    print(f"{n} samples in {len(series)} series written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
