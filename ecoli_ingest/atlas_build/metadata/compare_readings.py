#!/usr/bin/env python3
"""Agreement between two label tables written by apply_labels.py.
Usage: compare_readings.py A.tsv B.tsv OUT.tsv   (prints per-label agreement; OUT lists the differences)"""
import csv, sys, collections
a = {r["GSM"]: r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t")}
b = {r["GSM"]: r for r in csv.DictReader(open(sys.argv[2]), delimiter="\t")}
cols = [c for c in next(iter(a.values())) if c not in ("GSM", "GSE") and not c.endswith("_basis")
        and c not in ("perturbed_genes", "strain_name")]
diff = collections.Counter()
print(f"{'label':28s} {'both':>5s} {'agree':>5s} {'%':>5s} {'A only':>6s} {'B only':>6s}  studies that differ")
for c in cols:
    both = agree = ao = bo = 0; st = set()
    for g in a.keys() & b.keys():
        x, y = a[g][c], b[g][c]
        if x and y:
            both += 1
            if x == y: agree += 1
            else: diff[(c, a[g]["GSE"], x, y)] += 1; st.add(a[g]["GSE"])
        elif x: ao += 1; diff[(c, a[g]["GSE"], x, "")] += 1
        elif y: bo += 1; diff[(c, a[g]["GSE"], "", y)] += 1
    print(f"{c:28s} {both:5d} {agree:5d} {100*agree/both if both else 0:5.1f} {ao:6d} {bo:6d}  {len(st)}")
with open(sys.argv[3], "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["label", "GSE", "A", "B", "samples"])
    for (c, g, x, y), n in sorted(diff.items()): w.writerow([c, g, x, y, n])
