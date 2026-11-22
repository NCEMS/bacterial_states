#!/usr/bin/env python3
"""Apply labels/<GSE>.json to the candidate samples, validate, derive the lumped columns.

Run from the repository root (see docs/metadata_v2.md). Environment: LABEL_DIR (default: labels/ beside
this file), DIGEST_DIR (default: digest/ beside this file; the digests written by geo_digest.py),
ONLY (comma-separated GSEs to restrict to).

Usage: apply_labels.py SOFT_DIR samples.tsv OUT_PREFIX [--v1 metadata_revised.csv] [--qc sample_qc.tsv]
With --qc: adds pre_decision, pre_reasons and usable (standard_rnaseq true and prescreen pass) and
writes OUT_PREFIX_counts_usable.tsv (class counts over usable samples only).
Writes OUT_PREFIX.tsv (one row per GSM), OUT_PREFIX_evidence.tsv, OUT_PREFIX_problems.tsv,
OUT_PREFIX_counts.tsv (samples and studies per class) and, with --v1, OUT_PREFIX_vs_v1.tsv.
"""
import csv, json, re, sys, pathlib, collections
import yaml
from geo_digest import parse, flat, add_suppl

here = pathlib.Path(__file__).parent
LABEL_DIR = pathlib.Path(__import__("os").environ.get("LABEL_DIR", str(pathlib.Path(__file__).parent / "labels")))
DIGEST_DIR = pathlib.Path(__import__("os").environ.get("DIGEST_DIR", str(here / "digest")))
scheme = yaml.safe_load(open(here / "label_scheme.yaml"))
LABELS, LUMPED = scheme["labels"], scheme["lumped"]
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()

def check_value(label, v):
    spec = LABELS.get(label)
    if spec is None: return "unknown label"
    if spec == "number":
        try: float(v); return None
        except (TypeError, ValueError): return "not a number"
    if spec == "text": return None
    parts = str(v).split(";") if label == "treatment" else [v]
    allowed = {str(x).lower() for x in spec}
    bad = [p for p in parts if str(p).strip().lower() not in allowed]
    return f"value not in scheme: {bad}" if bad else None

def lump(name, spec, row):
    v = row.get(spec["from"], "")
    if v == "": return {False: "false", True: "true"}.get(spec.get("empty"), spec.get("empty", ""))
    if "bins" in spec:
        x = float(v)
        return next((k for k, (lo, hi) in spec["bins"].items() if lo <= x <= hi), "")
    first = str(v).split(";")[0].strip()
    key = {"true": True, "false": False}.get(first.lower(), first)
    out = spec["map"].get(key, spec.get("default", ""))
    return {True: "true", False: "false"}.get(out, out)

def main():
    soft_dir, samples_tsv, prefix = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    v1 = sys.argv[sys.argv.index("--v1") + 1] if "--v1" in sys.argv else None
    status = {r["GSM"]: r["status"] for r in csv.DictReader(open(samples_tsv), delimiter="\t")}
    rows, evidence, problems = [], [], []
    only = set(filter(None, __import__("os").environ.get("ONLY", "").split(",")))
    for lf in sorted(LABEL_DIR.glob("GSE*.json")):
        if only and lf.stem not in only: continue
        gse = lf.stem
        try: lab = json.load(open(lf))
        except Exception as e: problems.append((gse, "", "", f"JSON error: {e}")); continue
        series, samples = parse(soft_dir / f"{gse}.soft")
        fields = add_suppl(gse, {g: flat(s) for g, s in samples.items() if status.get(g) == "candidate"})
        text = norm((DIGEST_DIR / f"{gse}.md").read_text())
        for i, r in enumerate(lab.get("rules", [])):
            try: n = sum(bool(re.search(r["regex"], f.get(r["field"], ""))) for f in fields.values())
            except re.error as e: problems.append((gse, "", f"rule {i}", f"bad regex: {e}")); r["regex"] = "(?!)"; continue
            if n == 0: problems.append((gse, "", f"rule {i}", f"matches no sample: {r['field']} ~ {r['regex']}"))
        for g in lab.get("samples", {}):
            if g not in fields: problems.append((gse, g, "", "sample entry is not a candidate sample of this study"))
        for g, f in fields.items():
            cur = {}
            def put(d, fallback=""):
                for k, x in d.items():
                    if not isinstance(x, dict): x = {"v": x, "b": "", "q": ""}
                    if x.get("v") is None: cur.pop(k, None)
                    else: cur[k] = (x["v"], x.get("b", ""), x.get("q") or fallback)
            put(lab.get("default", {}))
            for r in lab.get("rules", []):
                if re.search(r["regex"], f.get(r["field"], "")): put(r["set"], f.get(r["field"], "")[:200])
            put(lab.get("samples", {}).get(g, {}))
            row = {"GSM": g, "GSE": gse}
            for k, (v, b, q) in cur.items():
                err = check_value(k, v)
                if err: problems.append((gse, g, k, err)); continue
                if b not in ("stated", "inferred", "study_default"): problems.append((gse, g, k, f"basis '{b}'"))
                if not q: problems.append((gse, g, k, "no quote"))
                elif norm(q) not in text: problems.append((gse, g, k, f"quote not in text: {q[:80]}"))
                row[k] = {True: "true", False: "false"}.get(v, str(v)); row[k + "_basis"] = b
                evidence.append((g, gse, k, row[k], b, norm(q)))
            for name, spec in LUMPED.items(): row[name] = lump(name, spec, row)
            # phage-infected samples are their own class; phage RNA alone does not make a sample mixed-organism
            toks = [x.strip() for x in row.get("treatment", "").split(";") if x.strip()]
            row["phage_infected"] = "" if not toks else ("true" if "phage" in toks else "false")
            if "phage" in toks and not {"coculture", "conjugation"} & set(toks) and row.get("in_host") != "true":
                row["mixed_organism"] = "false"
            for name, spec in LUMPED.items():
                if any(row.get(k, "") != v for k, v in spec.get("requires", {}).items()): row[name] = ""
            rows.append(row)
    qc = sys.argv[sys.argv.index("--qc") + 1] if "--qc" in sys.argv else None
    extra = []
    if qc:
        dec = {r["GSM"]: (r["pre_decision"], r["pre_reasons"], r.get("pre_pct_CDS", "")) for r in csv.DictReader(open(qc), delimiter="\t")}
        for r in rows:
            r["pre_decision"], r["pre_reasons"], r["pre_pct_CDS"] = dec.get(r["GSM"], ("", "", ""))
            # under 20% of prescreen reads on CDS: candidates for the minimum-mRNA-read gate of the full run
            r["low_mrna_prescreen"] = "" if r["pre_pct_CDS"] in ("", "nan") else ("true" if float(r["pre_pct_CDS"]) < 20 else "false")
            r["usable"] = "true" if r["standard_rnaseq"] == "true" and r["pre_decision"] == "pass" else "false"
        extra = ["pre_decision", "pre_reasons", "pre_pct_CDS", "low_mrna_prescreen", "usable"]
    dup_file = here / "duplicate_samples.tsv"   # later copies of samples deposited twice (same read counts)
    dups = {r["GSM"]: r["duplicate_of_GSM"] for r in csv.DictReader(open(dup_file), delimiter="\t")} if dup_file.exists() else {}
    for r in rows:
        r["duplicate_of"] = dups.get(r["GSM"], "")
        if r["duplicate_of"] and "usable" in r: r["usable"] = "false"
    extra.append("duplicate_of")
    flag_file = here / "study_flags.tsv"       # studies to be able to leave out of modelling steps
    flags = {r["GSE"]: r["flag"] for r in csv.DictReader(open(flag_file), delimiter="\t")} if flag_file.exists() else {}
    for r in rows: r["study_flag"] = flags.get(r["GSE"], "")
    extra += ["study_flag"]
    cols = ["GSM", "GSE"] + [c for k in LABELS for c in (k, k + "_basis")] + list(LUMPED) + ["phage_infected"] + extra
    with open(prefix + ".tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", restval=""); w.writeheader(); w.writerows(rows)
    for name, hdr, data in (("_evidence", ["GSM", "GSE", "label", "value", "basis", "quote"], evidence),
                            ("_problems", ["GSE", "GSM", "label", "problem"], sorted(set(problems)))):
        with open(prefix + name + ".tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t"); w.writerow(hdr); w.writerows(data)
    # class counts (all samples, and usable samples only when --qc is given)
    for suffix, sel in [("_counts", rows)] + ([("_counts_usable", [r for r in rows if r.get("usable") == "true"])] if qc else []):
        with open(prefix + suffix + ".tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t"); w.writerow(["label", "kind", "value", "samples", "studies"])
            for k in [k for k in LABELS if LABELS[k] != "text"] + list(LUMPED):
                cnt, st = collections.Counter(), collections.defaultdict(set)
                for r in sel:
                    for v in ([r.get(k, "")] if k != "treatment" else str(r.get(k, "")).split(";")):
                        v = v or "(empty)"; cnt[v] += 1; st[v].add(r["GSE"])
                for v, n in cnt.most_common(): w.writerow([k, "lumped" if k in LUMPED else "fine", v, n, len(st[v])])
    print(f"{len(rows)} samples in {len({r['GSE'] for r in rows})} studies; {len(set(problems))} problems")
    if v1:
        old = {}
        for r in csv.DictReader(open(v1)): old.setdefault(r["sample_id"].split("_")[0], r)
        ph = {"exponential": "exponential", "stationary": "stationary", "late_stationary": "stationary"}
        cmp = [("growth_phase", "phase2", lambda x: ph.get(x, "(other:" + x + ")")),
               ("medium_type", "medium_type", lambda x: {"complex": "rich", "defined": "minimal"}.get(x, "(other:" + x + ")")),
               ("oxygen", "anaerobic", lambda x: {"aerobic": "false", "aeorbic": "false", "anaerobic": "true"}.get(x, "(other:" + x + ")")),
               ("carbon_source", "glucose", lambda x: "true" if x == "glucose" else "false")]
        with open(prefix + "_vs_v1.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t"); w.writerow(["label", "v1", "v2", "v2_fine", "samples", "studies", "example_GSE"])
            for oc, nc, f in cmp:
                tab = collections.defaultdict(lambda: [0, set()])
                for r in rows:
                    o = old.get(r["GSM"])
                    if o is None: continue
                    a = f(o[oc]) if o[oc] else "(empty)"; b = r.get(nc) or "(empty)"
                    fine = r.get(LUMPED[nc]["from"], "") if a != b else ""
                    t = tab[(a, b, fine)]; t[0] += 1; t[1].add(r["GSE"])
                for (a, b, fine), (n, s) in sorted(tab.items()): w.writerow([oc, a, b, fine, n, len(s), ",".join(sorted(s)[:4])])

if __name__ == "__main__":
    main()
