"""`ecoli_atlas view`: one self-contained HTML page with an interactive 3-D map of the atlas.

The atlas is the embedded training corpus of a bundle (``embedding.tsv``, ``predictions.tsv``,
``manifest.json``). Optional inputs add the curated sample metadata (``metadata_v2.tsv``), the GEO text
of studies and samples (``metadata_v2_studies.tsv`` / ``metadata_v2_sample_text.tsv``, written by
``ecoli_ingest/atlas_build/metadata/study_text.py``) and samples of `ecoli_atlas embed`, drawn among the
atlas samples with their nearest atlas neighbours.

All data go into the page as one JSON block; plotly.js is inlined (from the ``plotly`` Python package
or a local file) so the page works offline, or loaded from a CDN (``plotly_js="cdn"``).
"""
from __future__ import annotations

import json
import re
from datetime import date
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .. import __version__
from ..compare import calls_frame, read_embedding
from ..config import HEADS
from .layouts import build_layouts

PLOTLY_VERSION = "2.35.3"
PLOTLY_CDN = f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{PLOTLY_VERSION}/plotly-gl3d.min.js"
SPLIT_NAMES = {"0": "projected", "1": "training", "3": "TEST"}

# label columns of metadata_v2 shown in the sample panel (in this order) and offered for colouring
LABEL_FIELDS = ("growth_phase", "culture_mode", "base_medium", "medium_class", "medium_type", "amino_acids_supplied",
                "carbon_source", "oxygen", "growth_temperature", "temp_class", "genotype_class", "perturbed_genes",
                "global_regulator_perturbed", "inducer_added", "heterologous_expression", "treatment", "exposure",
                "stress_family", "treated", "acute", "is_control", "is_reference_genotype", "is_wild_type", "strain_group",
                "strain_name", "pathogenic_isolate", "in_host", "assay_class", "rrna_removal", "transcription_arrested",
                "phage_infected", "mixed_organism", "study_flag")
# columns that may differ between the conditions of one study
CONDITION_FIELDS = ("growth_phase", "culture_mode", "base_medium", "medium_class", "carbon_source", "oxygen",
                    "growth_temperature", "strain_name", "genotype_class", "perturbed_genes", "inducer_added",
                    "heterologous_expression", "treatment", "exposure")
NUMERIC_FIELDS = {"growth_temperature", "phase_logit", "n_missing_genes"} | {f"p_{pos}" for _, (_, pos) in HEADS.items()}
SAMPLE_TEXT_FIELDS = ("title", "source_name", "characteristics", "description", "runs", "layout", "instrument")
STUDY_FIELDS = ("title", "summary", "overall_design", "type", "pubmed_ids", "bioproject", "sra", "contributors",
                "submission_date", "first_public", "study_flag", "curator_notes")
REPLICATE = re.compile(r"([\s_,;:-]*(biological|technical)?\s*(replicate|rep|repl|br|bio|r|clone|sample)[\s_#.-]*\d+[a-z]?\b"
                       r"|[\s_-]+\d+$|[\s_-]+[A-C]$)", re.IGNORECASE)


def _text_table(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.strip()
    return df


def title_group(title: str) -> str:
    """A GEO sample title without its replicate part (``WT 3h rep2`` -> ``WT 3h``)."""
    t = (title or "").strip()
    for _ in range(3):
        new = REPLICATE.sub("", t).strip(" _-,;:")
        if new == t or not new:
            break
        t = new
    return t


def condition_groups(meta: pd.DataFrame, gse: pd.Series, titles: pd.Series | None) -> tuple[pd.Series, pd.Series, dict]:
    """Per sample: the condition from the label columns that vary within its study, and the GEO title
    without replicate; per study: the varying label columns."""
    cond = pd.Series("", index=meta.index, dtype=object)
    tgrp = pd.Series("", index=meta.index, dtype=object)
    varying = {}
    cols = [c for c in CONDITION_FIELDS if c in meta.columns]
    for study, idx in gse.groupby(gse).groups.items():
        sub = meta.loc[idx, cols].fillna("").astype(str)
        var = [c for c in cols if sub[c].nunique() > 1]
        varying[study] = var
        if var:
            cond.loc[idx] = sub[var].apply(lambda r: " · ".join(v if v else "–" for v in r), axis=1)
        else:
            cond.loc[idx] = "all samples"
        if titles is not None:
            tgrp.loc[idx] = [title_group(t) or "–" for t in titles.reindex(idx).fillna("")]
    return cond, tgrp, varying


def _missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() in ("", "nan", "NaN")
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def encode_field(name: str, values: pd.Series, kind: str | None = None) -> dict:
    """A column of the page: categorical (levels + codes, -1 = missing) or numeric (null = missing)."""
    vals = [None if _missing(v) else v for v in values]
    if kind is None:
        kind = "num" if name in NUMERIC_FIELDS else "cat"
    if kind == "num":
        x = pd.to_numeric(pd.Series(vals, dtype=object), errors="coerce")
        if x.notna().sum() >= max(1, sum(v is not None for v in vals) * 0.9):
            return {"name": name, "kind": "num", "values": [None if np.isnan(v) else round(float(v), 5) for v in x]}
    # plain lists: pandas turns None back into NaN in object Series operations
    strs = [None if v is None else str(v) for v in vals]
    levels = sorted({v for v in strs if v is not None})
    code = {v: i for i, v in enumerate(levels)}
    return {"name": name, "kind": "cat", "levels": levels, "values": [-1 if v is None else code[v] for v in strs]}


def load_overlay(dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[dict]]:
    """(embedding, calls + probabilities, info, covariates or None, sets) of `embed` output directories;
    ``DIR`` or ``DIR=NAME``. The covariates come from ``covariates.tsv`` (written by `embed`). ``sets``: per
    directory its name, path, the samples' own ids and their ids in the page."""
    embs, probs, infos, covs, sets = [], [], [], [], []
    for spec in dirs:
        path, _, name = spec.partition("=")
        d = Path(path)
        name = name or (d.name if d.is_dir() else d.parent.name)
        emb = read_embedding(d)
        base = d if d.is_dir() else d.parent
        pred = pd.read_csv(base / "predictions.tsv", sep="\t", index_col=0) if (base / "predictions.tsv").exists() \
            else pd.DataFrame(index=emb.index)
        pred.index = pred.index.astype(str)
        pred = pred.reindex(emb.index)
        info = pd.DataFrame({"set": name}, index=emb.index)
        if (base / "samples.tsv").exists():
            st = pd.read_csv(base / "samples.tsv", sep="\t", index_col=0)
            st.index = st.index.astype(str)
            if "n_missing_genes" in st.columns:
                info["n_missing_genes"] = st["n_missing_genes"].reindex(emb.index)
        cov = None
        if (base / "covariates.tsv").exists():
            cov = pd.read_csv(base / "covariates.tsv", sep="\t", index_col=0)
            cov.index = cov.index.astype(str)
            cov = cov.reindex(emb.index)
        own = list(emb.index)
        emb.index = [f"{name}:{s}" if len(dirs) > 1 else s for s in emb.index]
        sets.append({"name": name, "dir": base, "samples": own, "ids": list(emb.index)})
        pred.index, info.index = emb.index, emb.index
        if cov is not None:
            cov.index = emb.index
            covs.append(cov)
        info["sample"] = [s.split(":", 1)[-1] for s in emb.index]
        embs.append(emb), probs.append(pred), infos.append(info)
    names = [s["name"] for s in sets]
    if len(set(names)) < len(names):
        raise ValueError(f"two sets of your samples have the same name ({names}); name them with --embeddings DIR=NAME")
    return pd.concat(embs), pd.concat(probs), pd.concat(infos), (pd.concat(covs) if covs else None), sets


def _per_set(specs: list[str] | None, sets: list[dict], what: str) -> dict:
    """``[NAME=]FILE`` options of the sets of your samples -> {set name: file}. Without names, one file goes to the
    only set, or the files to the sets in their order."""
    out, unnamed = {}, []
    names = {s["name"] for s in sets}
    for spec in specs or []:
        name, sep, path = str(spec).partition("=")
        if sep and name in names:
            out[name] = path
        else:
            unnamed.append(str(spec))
    free = [s["name"] for s in sets if s["name"] not in out]
    if len(unnamed) > len(free) or (unnamed and len(unnamed) != len(free) and len(free) > 1):
        raise ValueError(f"{what}: {len(unnamed)} file(s) without a set name for the sets {free}; give NAME=FILE")
    out.update(dict(zip(free, unnamed)))
    return out


def read_sheet(path) -> pd.DataFrame:
    """A sample sheet (csv or tsv): first column = sample id, the other columns as text."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False, comment="#")
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.strip()
    return df


def overlay_sheets(specs: list[str] | None, sets: list[dict], log=print) -> pd.DataFrame | None:
    """The sample sheets of your samples as one table over the page's sample ids (None without sheets)."""
    files = _per_set(specs, sets, "--overlay-metadata")
    if not files:
        return None
    parts = []
    for s in sets:
        if s["name"] not in files:
            parts.append(pd.DataFrame(index=s["ids"]))
            continue
        sheet = read_sheet(files[s["name"]])
        lacking = [x for x in s["samples"] if x not in sheet.index]
        if len(lacking) == len(s["samples"]):
            raise ValueError(f"{files[s['name']]}: none of the samples of {s['name']} is in its first column "
                             f"(e.g. {s['samples'][:3]}; the sheet has {list(sheet.index[:3])})")
        if lacking:
            log(f"{files[s['name']]}: no row for {len(lacking)} sample(s) of {s['name']}, e.g. {lacking[:3]}")
        part = sheet[~sheet.index.duplicated()].reindex(s["samples"])
        part.index = s["ids"]
        parts.append(part)
    return pd.concat(parts).fillna("")


def sheet_field(name: str, values: pd.Series) -> dict:
    """A sheet column: numeric when every value is a number and there are many of them, else categorical."""
    given = [v for v in values if not _missing(v)]
    x = pd.to_numeric(pd.Series(given, dtype=object), errors="coerce")
    return encode_field(name, values, "num" if len(given) and x.notna().all() and x.nunique() > 12 else "cat")


def series_of_sheet(sheet: pd.DataFrame, spec: str, sets: list[dict]) -> list[dict]:
    """``COLUMN[:ORDER_COLUMN]`` -> the series of your samples: per set and value of COLUMN, the samples in the order
    of ORDER_COLUMN (numbers when it holds numbers; the sheet's row order without it)."""
    col, _, order = spec.partition(":")
    for c in (col, order):
        if c and c not in sheet.columns:
            raise ValueError(f"--overlay-series: the sample sheet has no column {c!r} (it has {list(sheet.columns)})")
    pos = {s: j for j, s in enumerate(sheet.index)}
    out = []
    for s in sets:
        sub = sheet.loc[s["ids"]]
        for value, grp in sub[sub[col] != ""].groupby(col, sort=False):
            if order:
                key = pd.to_numeric(grp[order], errors="coerce")
                grp = grp.assign(_k=key if key.notna().all() else grp[order]).sort_values("_k", kind="stable")
            if len(grp) > 1:
                out.append({"name": f"{col} {value}" + (f" ({s['name']})" if len(sets) > 1 else ""), "idx": [pos[i] for i in grp.index],
                            "x": grp[order].tolist() if order else [], "x_label": order})
    return out


def atlas_series(path, ids: list) -> list[dict]:
    """The time courses of the atlas (``time_courses.tsv``: sample, series, x, x_label) as lists of sample
    indices in the order of x; a series needs two samples of the atlas."""
    t = pd.read_csv(path, sep="\t", dtype={"sample": str, "series": str})
    missing = {"sample", "series", "x"} - set(t.columns)
    if missing:
        raise ValueError(f"{path}: the time-series table lacks the column(s) {sorted(missing)}")
    pos = {s: i for i, s in enumerate(ids)}
    t = t[t["sample"].isin(pos)].assign(x=lambda d: pd.to_numeric(d["x"], errors="coerce")).dropna(subset=["x"])
    out = []
    for name, grp in t.groupby("series", sort=False):
        grp = grp.sort_values("x", kind="stable")
        if len(grp) > 1:
            out.append({"name": str(name), "idx": [pos[s] for s in grp["sample"]], "x": [round(float(v), 4) for v in grp["x"]],
                        "x_label": str(grp["x_label"].iloc[0]) if "x_label" in grp.columns else ""})
    return out


def seed_agreement(models: Path, raw: pd.DataFrame) -> tuple[dict, int]:
    """Per call, the number of the bundle's seeds (``seed_<n>/predictions.tsv``) on the side of the bundle's
    call (probability 0.5), and the number of seeds; ({}, 0) for a bundle without seeds."""
    files = sorted(models.glob("seed_*/predictions.tsv"))
    out = {}
    if len(files) < 2:
        return out, 0
    seeds = []
    for f in files:
        t = pd.read_csv(f, sep="\t", index_col=0)
        t.index = t.index.astype(str)
        seeds.append(t.reindex(raw.index))
    for h, (_, pos) in HEADS.items():
        c = f"p_{pos}"
        if c not in raw.columns or not all(c in t.columns for t in seeds):
            continue
        main = pd.to_numeric(raw[c], errors="coerce")
        same = sum(((pd.to_numeric(t[c], errors="coerce") >= 0.5) == (main >= 0.5)).astype(float) for t in seeds)
        out[h] = same.where(main.notna() & np.all([t[c].notna() for t in seeds], axis=0))
    return out, len(files)


def atlas_covariates(manifest: dict, ids: list, counts=None, qc=None, log=print) -> pd.DataFrame | None:
    """The model's covariates of the atlas samples before scaling, computed as in `train`: the
    composition columns from the count table (``counts``, as read by ``read_counts``), the QC columns from the QC table
    (``qc``). Columns that cannot be computed from what was given are left out."""
    from ..data import COMPOSITION_NAMES, qc_covariates, read_table
    parts = []
    if counts is not None:
        from ..features import FeatureSet
        from ..preprocess import align_counts, composition_covariates
        table = counts
        lacking = sorted(set(ids) - set(table.columns))
        if lacking:
            log(f"{len(lacking)} atlas samples are not in the count table (e.g. {lacking[:3]}); no composition values for them")
        fs = FeatureSet.from_dict(manifest["features"])
        Y, fam, _ = align_counts(table.reindex(columns=ids), fs)
        comp = pd.DataFrame(composition_covariates(np.nan_to_num(Y), np.nan_to_num(fam), fs.is_mrna), index=ids,
                            columns=list(COMPOSITION_NAMES))
        comp.loc[lacking] = np.nan
        parts.append(comp)
    if qc is not None:
        parts.append(qc_covariates(read_table(qc), ids))
    return pd.concat(parts, axis=1) if parts else None


def covariate_columns(cov: pd.DataFrame) -> dict:
    """Display columns of the covariates: numeric values as they are, the instrument flags as one
    ``instrument`` column, the other flags as yes/no; flags of a sample without QC row are missing."""
    out = {}
    has_qc = cov["has_qc"].astype(float) > 0 if "has_qc" in cov.columns else pd.Series(True, index=cov.index)
    inst = [c for c in cov.columns if c.startswith("instrument_")]
    for c in cov.columns:
        if c in inst:
            continue
        if c in ("paired", "has_qc", "dna_below_limit"):
            yes = cov[c].astype(float) > 0
            s = pd.Series(np.where(yes, "yes", "no"), index=cov.index, dtype=object)
            out[c] = s if c == "has_qc" else s.where(has_qc, None)
        else:
            out[c] = pd.to_numeric(cov[c], errors="coerce")
    if inst:
        names = pd.Series("other", index=cov.index, dtype=object)
        for c in inst:
            names[cov[c].astype(float) > 0] = c.removeprefix("instrument_")
        out["instrument"] = names.where(has_qc, None)
    return out


def neighbours(atlas: np.ndarray, query: np.ndarray, k: int, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(k, len(atlas)), metric=metric).fit(atlas)
    dist, idx = nn.kneighbors(query)
    return idx, dist


def load_presets(path: str | Path | None, studies: set, layouts: set, log=print) -> list[dict]:
    if path is None:
        text = resources.files("ecoli_atlas.viewer").joinpath("presets.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    out = []
    for p in json.loads(text):
        gse = [g for g in p.get("gse", []) if g in studies]
        missing = [g for g in p.get("gse", []) if g not in studies]
        if missing:
            log(f"preset {p.get('id')}: {', '.join(missing)} not in the atlas" + ("" if gse else "; preset dropped"))
        if not gse:
            continue
        q = dict(p, gse=gse)
        if q.get("layout") not in layouts:
            q["layout"] = "pca_all"
        out.append(q)
    return out


def _round(a: np.ndarray, digits: int = 3) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(a, dtype=float).ravel()]


def page_layout(L: dict, layouts: list[dict]) -> dict:
    """A layout as the page stores it. Columns that repeat those of another layout (the residual PCs
    of the block layouts) are not stored twice: ``base`` names that layout and ``own`` the number of
    columns stored here; column ``own + k`` is column ``k`` of the base."""
    out = {"name": L["name"], "label": L["label"], "axes": L["axes"], "ncol": int(L["atlas"].shape[1]),
           "default": L.get("default", [0, 1, 2]), "choose": L.get("choose")}
    own = out["ncol"]
    for B in layouts:
        nb = B["atlas"].shape[1]
        if B is not L and 0 < own - nb and B["atlas"].shape[0] == L["atlas"].shape[0] \
                and np.array_equal(L["atlas"][:, own - nb:], B["atlas"]) \
                and ("overlay" in L) == ("overlay" in B) and ("overlay" not in L or np.array_equal(L["overlay"][:, own - nb:], B["overlay"])):
            out.update(base=B["name"], own=own - nb)
            own -= nb
            break
    out["atlas"] = _round(L["atlas"][:, :own])
    if "overlay" in L:
        out["overlay"] = _round(L["overlay"][:, :own])
    return out


def build_payload(models: str | Path, metadata: str | Path | None = None, studies: str | Path | None = None,
                  sample_text: str | Path | None = None, embeddings: list[str] | None = None, presets: str | Path | None = None,
                  umap: bool = True, k: int = 10, metric: str = "cosine", seed: int = 77, title: str | None = None,
                  counts: str | Path | None = None, qc: str | Path | None = None, n_pcs: int = 10,
                  modules: str | Path | None = None, time_series: str | Path | None = None,
                  overlay_metadata: list[str] | None = None, overlay_series: str | None = None,
                  overlay_counts: list[str] | None = None, log=print) -> dict:
    d = Path(models)
    if modules is not None and counts is None:
        raise ValueError("--modules needs the count table of the atlas samples (--counts): the regulon values are computed from it")
    if not embeddings and (overlay_metadata or overlay_series or overlay_counts):
        raise ValueError("--overlay-metadata, --overlay-series and --overlay-counts describe your own samples; give them with --embeddings")
    if overlay_series and not overlay_metadata:
        raise ValueError("--overlay-series names columns of the sample sheet; give the sheet with --overlay-metadata")
    manifest = json.loads((d / "manifest.json").read_text())
    emb = read_embedding(d)
    raw = pd.read_csv(d / "predictions.tsv", sep="\t", index_col=0, dtype={"gse": str, "split": str})
    raw.index = raw.index.astype(str)
    raw = raw.reindex(emb.index)
    ids = list(emb.index)
    gse = raw["gse"].fillna("").astype(str)
    if (gse == "").any() and metadata is None:
        raise ValueError("predictions.tsv lacks the study of some samples; give --metadata")
    meta = _text_table(metadata).reindex(emb.index) if metadata is not None else pd.DataFrame(index=emb.index)
    if "GSE" in meta.columns:
        gse = gse.where(gse != "", meta["GSE"].fillna(""))
    stext = _text_table(sample_text).reindex(emb.index) if sample_text is not None else None
    stable = _text_table(studies) if studies is not None else None

    fields = []
    split = raw["split"].astype(str).map(lambda s: SPLIT_NAMES.get(s.split(".")[0], s)) if "split" in raw.columns else None
    if split is not None:
        fields.append(encode_field("split", split))
    fields.append(encode_field("gse", gse))
    calls = calls_frame(raw)
    for h, (neg, pos) in HEADS.items():
        if f"{h}_label" in raw.columns:
            fields.append(encode_field(f"{h}_label", raw[f"{h}_label"]))
        if f"{h}_pred" in calls.columns:
            fields.append(encode_field(f"{h}_call", calls[f"{h}_pred"]))
        if f"p_{pos}" in raw.columns:
            fields.append(encode_field(f"p_{pos}", raw[f"p_{pos}"]))
    for c in ("phase_logit", "n_missing_genes"):
        if c in raw.columns:
            fields.append(encode_field(c, raw[c]))
    for c in LABEL_FIELDS:
        if c in meta.columns:
            fields.append(encode_field(c, meta[c]))
            if f"{c}_basis" in meta.columns:
                fields.append(encode_field(f"{c}_basis", meta[f"{c}_basis"]))
    cond, tgrp, varying = condition_groups(meta, gse, stext["title"] if stext is not None and "title" in stext.columns else None)
    fields.append(encode_field("condition", cond))
    if stext is not None:
        fields.append(encode_field("title_group", tgrp))
    agreement, n_seeds = seed_agreement(d, raw)
    for h, series in agreement.items():
        fields.append(encode_field(f"{h}_seed_agreement", series, "num"))
    count_table = None
    if counts is not None:
        from ..io import read_counts
        count_table = read_counts(counts)
    cov = atlas_covariates(manifest, ids, count_table, qc, log=log)
    qc_fields = []
    if cov is not None:
        for c, series in covariate_columns(cov).items():
            fields.append(encode_field(f"cov:{c}", series, "num" if pd.api.types.is_numeric_dtype(series) else "cat"))
            qc_fields.append(f"cov:{c}")

    overlay = overlay_probs = overlay_info = overlay_cov = sheet = None
    overlay_sets = []
    if embeddings:
        overlay, overlay_probs, overlay_info, overlay_cov, overlay_sets = load_overlay(embeddings)
        sheet = overlay_sheets(overlay_metadata, overlay_sets, log=log)
        missing = [c for c in emb.columns if c not in overlay.columns]
        if missing:
            raise ValueError(f"the embedded samples lack embedding columns {missing[:5]}; embed them with this bundle")
    layouts = build_layouts(emb, manifest.get("embedding_blocks", []), raw, overlay, overlay_probs, umap=umap, seed=seed, log=log,
                            n_pcs=n_pcs)

    study_info = {}
    for g in sorted(set(gse) - {""}):
        n = int((gse == g).sum())
        row = {"n": n, "varying": varying.get(g, [])}
        if stable is not None and g in stable.index:
            row.update({f: stable.at[g, f] for f in STUDY_FIELDS if f in stable.columns and stable.at[g, f]})
        study_info[g] = row

    payload = {
        "meta": {"title": title or "E. coli expression atlas", "bundle": str(d.resolve().name), "n_atlas": len(ids),
                 "n_studies": len(study_info), "built": date.today().isoformat(), "version": __version__,
                 "embedding_dim": emb.shape[1], "blocks": manifest.get("embedding_blocks", []),
                 "heads": {h: list(v) for h, v in HEADS.items()}, "label_fields": [c for c in LABEL_FIELDS if c in meta.columns],
                 "has_text": stext is not None, "has_studies": stable is not None, "qc_fields": qc_fields, "n_seeds": n_seeds},
        "ids": ids,
        "fields": fields,
        "layouts": [page_layout(L, layouts) for L in layouts],
        "studies": study_info,
        "sample_text": ({f: stext[f].fillna("").tolist() for f in SAMPLE_TEXT_FIELDS if f in stext.columns}
                        if stext is not None else {}),
    }
    if overlay is not None:
        idx, dist = neighbours(emb.to_numpy(float), overlay[emb.columns].to_numpy(float), k, metric)
        oc = calls_frame(overlay_probs)
        ov = {"ids": list(overlay.index), "sample": overlay_info["sample"].tolist(), "set": overlay_info["set"].tolist(),
              "neighbors": [[[int(i), round(float(x), 4)] for i, x in zip(ri, rd)] for ri, rd in zip(idx, dist)],
              "metric": metric, "calls": {}}
        for h, (neg, pos) in HEADS.items():
            if f"{h}_pred" in oc.columns:
                ov["calls"][h] = {"call": [None if v is None else str(v) for v in oc[f"{h}_pred"]],
                                  "p": _round(oc[f"{h}_prob_{pos}"], 5), "positive": pos}
        if "n_missing_genes" in overlay_info.columns:
            ov["n_missing_genes"] = _round(overlay_info["n_missing_genes"], 0)
        if overlay_cov is not None:
            ov["covariates"] = {}
            for c, series in covariate_columns(overlay_cov).items():
                ov["covariates"][f"cov:{c}"] = (_round(series, 4) if pd.api.types.is_numeric_dtype(series)
                                                else [None if _missing(v) else str(v) for v in series])
        seeds_n, seeds_called = {}, {}
        for s in overlay_sets:                       # `embed --seeds`: the calls of every seed of the bundle
            f = s["dir"] / "predictions_seeds.tsv"
            if f.exists():
                t = pd.read_csv(f, sep="\t", index_col=0)
                t.index = t.index.astype(str)
                t = t.reindex(s["samples"])
                seeds_n.update(zip(s["ids"], t["n_seeds"]))
                for h, (_, pos) in HEADS.items():
                    if f"p_{pos}_n_called" in t.columns:
                        seeds_called.setdefault(h, {}).update(zip(s["ids"], t[f"p_{pos}_n_called"]))
        if seeds_called:
            ov["n_seeds"] = [None if pd.isna(seeds_n.get(i, np.nan)) else int(seeds_n[i]) for i in overlay.index]
            ov["seed_agreement"] = {}
            for h, called in seeds_called.items():
                if h not in ov["calls"]:
                    continue
                side = [c == ov["calls"][h]["positive"] for c in ov["calls"][h]["call"]]
                ov["seed_agreement"][h] = [None if n is None or pd.isna(called.get(i, np.nan)) else int(called[i] if up else n - called[i])
                                           for i, n, up in zip(overlay.index, ov["n_seeds"], side)]
        if sheet is not None:
            ov["fields"] = [sheet_field(c, sheet[c]) for c in sheet.columns if (sheet[c] != "").any()]
            if overlay_series:
                ov["series"] = series_of_sheet(sheet, overlay_series, overlay_sets)
        payload["overlay"] = ov
    if time_series is not None:
        payload["series"] = atlas_series(time_series, ids)
    if modules is not None:
        import torch
        from ..artifacts import PREPROCESSING_FILE
        from ..features import FeatureSet
        from ..io import read_counts
        from .regulons import build_regulons
        ref_log = torch.load(d / PREPROCESSING_FILE, map_location="cpu", weights_only=True)["ref_log"].numpy().astype(np.float64)
        own = None
        if overlay_sets:
            files = _per_set(overlay_counts, overlay_sets, "--overlay-counts")
            own = [(s["ids"], s["samples"], read_counts(files[s["name"]]) if s["name"] in files else None,
                    s["dir"] / "missing_genes.tsv") for s in overlay_sets]
            if not files:
                log("no counts of your samples (--overlay-counts): they get no regulon values")
        payload["regulons"] = build_regulons(modules, count_table, FeatureSet.from_dict(manifest["features"]), ref_log, ids,
                                             missing_path=d / "missing_genes.tsv", overlay=own, log=log)
        mf = Path(modules).parent / "modules_manifest.json"
        if mf.exists():
            urls = [v.get("url", "") for v in (json.loads(mf.read_text()).get("sources", {}).get("precise1k", {}) or {}).values()]
            commit = next((m.group(1) for m in (re.search(r"/precise1k/([0-9a-f]{7,40})/", u) for u in urls) if m), "")
            payload["regulons"]["source"] = "RegulonDB, as the regulatory network of SBRG/precise1k" + (f" (commit {commit[:10]})" if commit else "")
    elif overlay_counts:
        raise ValueError("--overlay-counts is used for the regulon values only; give --modules too")
    payload["presets"] = load_presets(presets, set(study_info), {L["name"] for L in layouts}, log=log)
    return payload


def plotly_source(plotly_js: str = "auto", log=print) -> str:
    """The <script> element that provides plotly.js."""
    if plotly_js == "cdn":
        return f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    if plotly_js not in ("auto", "inline"):
        code = Path(plotly_js).read_text(encoding="utf-8")
        return f"<script>{code}</script>"
    try:
        from plotly.offline import get_plotlyjs
        return f"<script>{get_plotlyjs()}</script>"
    except ImportError:
        if plotly_js == "inline":
            raise ValueError("--plotly inline needs the plotly Python package (pip install plotly) or a plotly.min.js path")
        log(f"the plotly Python package is not installed: the page loads plotly.js {PLOTLY_VERSION} from cdnjs and needs "
            "a network connection (pip install plotly, or --plotly PATH/TO/plotly.min.js, for an offline page)")
        return f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'


def render_html(payload: dict, plotly_js: str = "auto", log=print) -> str:
    static = resources.files("ecoli_atlas.viewer").joinpath("static")
    html = static.joinpath("template.html").read_text(encoding="utf-8")
    css = static.joinpath("viewer.css").read_text(encoding="utf-8")
    js = static.joinpath("viewer.js").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    title = payload["meta"]["title"].replace("<", "&lt;")
    parts = {"<!--PLOTLY-->": plotly_source(plotly_js, log), "/*CSS*/": css, "/*JS*/": js, "__TITLE__": title,
             "__DATA__": data}
    # one pass over the template, so a placeholder string inside the data or plotly.js is never replaced
    pattern = re.compile("|".join(re.escape(k) for k in parts))
    return pattern.sub(lambda m: parts[m.group(0)], html)


def write_view(out: str | Path, plotly_js: str = "auto", log=print, **kwargs) -> Path:
    payload = build_payload(log=log, **kwargs)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload, plotly_js, log), encoding="utf-8")
    return out
