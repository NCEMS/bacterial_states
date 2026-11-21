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


def load_overlay(dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(embedding, calls + probabilities, info) of `embed` output directories; ``DIR`` or ``DIR=NAME``."""
    embs, probs, infos = [], [], []
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
        emb.index = [f"{name}:{s}" if len(dirs) > 1 else s for s in emb.index]
        pred.index, info.index = emb.index, emb.index
        info["sample"] = [s.split(":", 1)[-1] for s in emb.index]
        embs.append(emb), probs.append(pred), infos.append(info)
    return pd.concat(embs), pd.concat(probs), pd.concat(infos)


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


def build_payload(models: str | Path, metadata: str | Path | None = None, studies: str | Path | None = None,
                  sample_text: str | Path | None = None, embeddings: list[str] | None = None, presets: str | Path | None = None,
                  umap: bool = True, k: int = 10, metric: str = "cosine", seed: int = 77, title: str | None = None,
                  log=print) -> dict:
    d = Path(models)
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

    overlay = overlay_probs = overlay_info = None
    if embeddings:
        overlay, overlay_probs, overlay_info = load_overlay(embeddings)
        missing = [c for c in emb.columns if c not in overlay.columns]
        if missing:
            raise ValueError(f"the embedded samples lack embedding columns {missing[:5]}; embed them with this bundle")
    layouts = build_layouts(emb, manifest.get("embedding_blocks", []), raw, overlay, overlay_probs, umap=umap, seed=seed, log=log)

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
                 "has_text": stext is not None, "has_studies": stable is not None},
        "ids": ids,
        "fields": fields,
        "layouts": [{"name": L["name"], "label": L["label"], "axes": L["axes"], "atlas": _round(L["atlas"])}
                    | ({"overlay": _round(L["overlay"])} if "overlay" in L else {}) for L in layouts],
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
        payload["overlay"] = ov
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
