/* ecoli_atlas view: interactive 3-D map of the atlas. Data come from the JSON block #atlas-data
   (written by ecoli_atlas/viewer/build.py); drawing uses plotly.js scatter3d. */
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("atlas-data").textContent);
  const plot = document.getElementById("plot");
  const N = D.ids.length;
  const F = {};                       // field name -> field
  D.fields.forEach(f => { F[f.name] = f; });
  const L = {};                       // layout name -> layout
  D.layouts.forEach(l => { L[l.name] = l; });
  const OV = D.overlay || null;
  const NO = OV ? OV.ids.length : 0;
  const PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
    "#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#9edae5", "#dbdb8d", "#393b79",
    "#637939", "#8c6d31", "#843c39", "#7b4173"];
  const MAX_LEVELS = 20, OTHER = "#c8c8c8", MISSING = "#e6e6e6", FADED = "#cfcfcf", OVERLAY = ["#e6007e", "#00a087", "#3c5488", "#f39b7f"];
  const OVF = {};                     // sheet column of your samples -> field
  ((OV && OV.fields) || []).forEach(f => { OVF[f.name] = f; });
  const R = D.regulons || null;       // regulon sets: one byte per sample and set
  const DIVERGING = [[0, "#2166ac"], [0.25, "#92c5de"], [0.5, "#f2f2f2"], [0.75, "#f4a582"], [1, "#b2182b"]];
  const HEAD_POS = {};                // head -> positive class
  Object.entries(D.meta.heads).forEach(([h, v]) => { HEAD_POS[h] = v[1]; });
  const NICE = {
    split: "split", gse: "study (GSE)", condition: "condition (labels that vary in the study)", title_group: "GEO sample title (without replicate)",
    phase_label: "phase label (curated)", medium_label: "medium label (curated)", lab_strain_label: "lab-strain label (curated)",
    phase_call: "phase call (model)", medium_call: "medium call (model)", lab_strain_call: "lab-strain call (model)",
    p_stationary: "p(stationary)", p_minimal: "p(minimal medium)", p_lab: "p(lab strain)", phase_logit: "phase logit",
    n_missing_genes: "genes called missing", phase_seed_agreement: "phase call: seeds that agree",
    medium_seed_agreement: "medium call: seeds that agree", lab_strain_seed_agreement: "lab-strain call: seeds that agree"
  };
  const COV = {                       // QC and composition covariates of the model (fields "cov:<name>")
    log_mrna_reads: "mRNA reads (log10)", log_16S_to_mrna: "16S / mRNA reads (log10)", log_23S_to_mrna: "23S / mRNA reads (log10)",
    log_5S_to_mrna: "5S / mRNA reads (log10)", log_tRNA_to_mrna: "tRNA / mRNA reads (log10)", length_q05_r1: "read 1 length, 5th percentile",
    length_q50_r1: "read 1 length, median", length_q95_r1: "read 1 length, 95th percentile", cycles_r1: "sequencing cycles, read 1",
    pct_kept: "% reads kept by fastp", pct_adapter_trimmed: "% reads adapter-trimmed", duplication_rate: "duplication rate",
    gc_content: "GC content", strand_purity: "strand purity", length_q50_r2: "read 2 length, median", fragment_q25: "fragment length, 25th pct",
    fragment_q50: "fragment length, median", fragment_q75: "fragment length, 75th pct", log_dna_fraction: "DNA fraction (log10)",
    leakage_b: "antisense leakage", intergenic_share: "intergenic share", gene_body_slope: "gene-body coverage slope",
    pct_short_features: "% reads on short features", log_other_genus: "other-genus share (log10)",
    log_substitution_rate: "substitution rate vs MG1655 (log10)", paired: "paired-end", instrument: "instrument",
    has_qc: "has QC row", dna_below_limit: "DNA below detection"
  };
  const nice = n => n.startsWith("cov:") ? (COV[n.slice(4)] || n.slice(4).replace(/_/g, " ")) : (NICE[n] || n.replace(/_/g, " "));
  const $ = id => document.getElementById(id);
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const natural = (a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });

  function val(name, i) {             // display value of a field for atlas sample i ('' when missing)
    const f = F[name];
    if (!f) return "";
    const v = f.values[i];
    if (f.kind === "cat") return v < 0 ? "" : f.levels[v];
    return v == null ? "" : v;
  }
  const gseOf = i => val("gse", i);
  const text = (name, i) => (D.sample_text && D.sample_text[name]) ? D.sample_text[name][i] || "" : "";
  const fmtP = p => (p === "" || p == null) ? "" : Number(p).toFixed(3);
  function ovVal(name, j) {           // value of a sheet column, or of "call:<head>", for your sample j ('' when missing)
    if (name.startsWith("call:")) { const c = (OV.calls || {})[name.slice(5)]; return c ? (c.call[j] || "") : ""; }
    const f = OVF[name];
    if (!f) return "";
    const v = f.values[j];
    if (f.kind === "cat") return v < 0 ? "" : f.levels[v];
    return v == null ? "" : v;
  }
  const regCache = {};
  function regulonValues(k, overlay) {   // values of set k for the atlas samples (or yours); null = no value
    const key = (overlay ? "o" : "a") + k;
    if (regCache[key]) return regCache[key];
    const src = overlay ? (R.overlay ? R.overlay[k] : null) : R.sets[k].values;
    if (!src) return null;
    const bytes = atob(src), lim = R.sets[k].limit, out = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) { const c = bytes.charCodeAt(i); out[i] = c === 0 ? null : ((c - 1) / 127 - 1) * lim; }
    regCache[key] = out;
    return out;
  }
  const seedsOf = (h, i) => (F[h + "_seed_agreement"] && val(h + "_seed_agreement", i) !== "") ? `${val(h + "_seed_agreement", i)}/${D.meta.n_seeds}` : "";

  // studies -> sample indices
  const studySamples = {};
  for (let i = 0; i < N; i++) { const g = gseOf(i); (studySamples[g] = studySamples[g] || []).push(i); }
  const idIndex = {};
  D.ids.forEach((s, i) => { idIndex[s] = i; });

  // hover text
  const hover = new Array(N);
  for (let i = 0; i < N; i++) {
    const t = text("title", i);
    const lines = [`<b>${esc(D.ids[i])}</b> · ${esc(gseOf(i))}${F.split ? " · " + esc(val("split", i)) : ""}`];
    if (t) lines.push(esc(t.length > 80 ? t.slice(0, 77) + "…" : t));
    Object.keys(HEAD_POS).forEach(h => {
      const lab = val(h + "_label", i), call = val(h + "_call", i), p = val("p_" + HEAD_POS[h], i);
      if (call !== "" || lab !== "") lines.push(`${esc(h.replace("_", " "))}: ${esc(call)} (p=${fmtP(p)})${lab !== "" ? " · label " + esc(lab) : ""}`);
    });
    ["growth_phase", "base_medium", "strain_name", "treatment"].forEach(c => { const v = val(c, i); if (v !== "" && v !== "none") lines.push(`${c.replace("_", " ")}: ${esc(v)}`); });
    hover[i] = lines.join("<br>");
  }
  const ovHover = [];
  for (let j = 0; j < NO; j++) {
    const lines = [`<b>${esc(OV.sample[j])}</b> · your samples: ${esc(OV.set[j])}`];
    Object.keys(OV.calls || {}).forEach(h => { const c = OV.calls[h]; lines.push(`${esc(h.replace("_", " "))}: ${esc(c.call[j])} (p=${fmtP(c.p[j])})`); });
    Object.keys(OVF).slice(0, 5).forEach(c => { const v = ovVal(c, j); if (v !== "") lines.push(`${esc(c.replace(/_/g, " "))}: ${esc(v)}`); });
    const nb = OV.neighbors[j][0];
    if (nb) lines.push(`nearest atlas sample: ${esc(D.ids[nb[0]])} (${esc(gseOf(nb[0]))})`);
    ovHover.push(lines.join("<br>"));
  }

  // ------------------------------------------------------------------ state
  const S = {
    layout: L.pca_all ? "pca_all" : D.layouts[0].name,
    color: "split" in F ? "split" : "gse",
    study: null,                     // list of GSE in study mode
    studyColor: "condition",
    onlyGroup: null,                 // a group value shown alone in study mode
    hideOthers: false,
    splits: new Set(F.split ? F.split.levels : []),
    size: 2.5,
    selected: null,                  // {kind: "atlas"|"overlay", i}
    hl: null,                        // atlas index highlighted from a table row
    caption: null,
    cols: {},                        // layout name -> the three columns drawn (layouts with selectable columns)
    regulon: null,                   // index of the regulon set that colours the map
    lines: true,                     // time courses as lines
    ovColor: null,                   // your samples coloured by a sheet column or "call:<head>"; null = by set
    rotate: false
  };
  const colsOf = lay => S.cols[lay.name] || lay.default || [0, 1, 2];

  // ------------------------------------------------------------------ controls
  function fillControls() {
    $("subtitle").textContent = `${D.meta.n_atlas.toLocaleString()} samples · ${D.meta.n_studies} studies · model ${D.meta.bundle} · ${D.meta.embedding_dim}-d embedding`;
    const lay = $("layout");
    D.layouts.forEach(l => { const o = document.createElement("option"); o.value = l.name; o.textContent = l.label; lay.appendChild(o); });
    const col = $("color");
    const groups = [["Model", ["split", "phase_call", "medium_call", "lab_strain_call", "p_stationary", "p_minimal", "p_lab", "phase_logit", "n_missing_genes",
      "phase_seed_agreement", "medium_seed_agreement", "lab_strain_seed_agreement"]],
      ["Curated labels", ["phase_label", "medium_label", "lab_strain_label"].concat(D.meta.label_fields)],
      ["QC covariates", D.meta.qc_fields || []], ["Study", ["gse"]]];
    groups.forEach(([title, names]) => {
      const og = document.createElement("optgroup"); og.label = title;
      names.filter(n => F[n]).forEach(n => { const o = document.createElement("option"); o.value = n; o.textContent = nice(n); og.appendChild(o); });
      if (og.children.length) col.appendChild(og);
    });
    if (F.split) {
      const order = ["training", "TEST", "projected"];
      F.split.levels.slice().sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99)).forEach(s => {
        const l = document.createElement("label"), c = document.createElement("input");
        c.type = "checkbox"; c.checked = true; c.value = s;
        c.addEventListener("change", () => { c.checked ? S.splits.add(s) : S.splits.delete(s); draw(); });
        l.appendChild(c); l.appendChild(document.createTextNode(" " + s)); $("splits").appendChild(l);
      });
    }
    const pre = $("preset");
    (D.presets || []).forEach((p, k) => { const o = document.createElement("option"); o.value = k; o.textContent = p.title; pre.appendChild(o); });
    if (!(D.presets || []).length) pre.parentElement.hidden = true;
    const dl = $("search-list");
    Object.keys(D.studies).sort(natural).forEach(g => {
      const o = document.createElement("option"); o.value = g;
      o.label = (D.studies[g].title || "") + ` (${D.studies[g].n})`; dl.appendChild(o);
    });
    lay.value = S.layout; col.value = S.color;
    lay.addEventListener("change", () => { const from = positions(); S.layout = lay.value; fillPcs(); draw(from); writeHash(); });
    ["pc-x", "pc-y", "pc-z"].forEach((id, d) => $(id).addEventListener("change", e => {
      const from = positions(), l = L[S.layout], c = colsOf(l).slice();
      c[d] = +e.target.value; S.cols[l.name] = c; draw(from); writeHash();
    }));
    fillPcs();
    col.addEventListener("change", () => { S.color = col.value; if (S.regulon !== null) setRegulon(null, true); else draw(); });
    // regulons
    $("regulon-box").hidden = !R;
    document.querySelector('.tabs button[data-tab="regulon"]').hidden = !R;
    if (R) {
      $("regulon-list").innerHTML = R.sets.map(t => `<option value="${esc(t.name)}" label="${t.genes.length} genes"></option>`).join("");
      $("regulon").addEventListener("keydown", e => { if (e.key === "Enter") regulonSearch($("regulon").value); });
      $("regulon").addEventListener("change", e => { if (R.sets.some(t => t.name === e.target.value.trim())) regulonSearch(e.target.value); });
      $("regulon-clear").addEventListener("click", () => setRegulon(null));
      renderRegulon();
    }
    // time courses, rotation, your samples
    const anySeries = (D.series || []).length || (OV && (OV.series || []).length);
    $("lines-box").hidden = !anySeries;
    $("lines").addEventListener("change", e => { S.lines = e.target.checked; draw(); });
    $("rotate").addEventListener("click", () => setRotate(!S.rotate));
    document.querySelector('.tabs button[data-tab="mine"]').hidden = !OV;
    if (OV) renderMine();
    $("size").addEventListener("input", e => { S.size = +e.target.value; draw(); });
    $("reset").addEventListener("click", () => { S.study = null; S.selected = null; S.hl = null; S.onlyGroup = null; setCaption(null); draw(); renderSample(); renderStudy(); writeHash(); flyHome(); });
    pre.addEventListener("change", () => { if (pre.value !== "") applyPreset(D.presets[+pre.value]); });
    $("search").addEventListener("keydown", e => { if (e.key === "Enter") search($("search").value); });
    $("search").addEventListener("change", e => { if (D.studies[e.target.value.trim()]) search(e.target.value); });
    document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
    renderAbout();
  }

  function fillPcs() {                // the X / Y / Z menus of a layout with selectable columns
    const l = L[S.layout];
    $("pcs").hidden = !l.choose;
    if (!l.choose) return;
    const c = colsOf(l);
    ["pc-x", "pc-y", "pc-z"].forEach((id, d) => {
      const sel = $(id);
      sel.innerHTML = l.choose[d].map(k => `<option value="${k}">${esc(l.axes[k])}</option>`).join("");
      sel.value = String(c[d]);
      sel.disabled = l.choose[d].length < 2;
    });
  }

  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + name));
  }

  function setCaption(p) {
    const c = $("caption");
    S.caption = p;
    if (!p) { c.hidden = true; $("preset").value = ""; return; }
    c.hidden = false;
    c.innerHTML = `<b>${esc(p.title)}</b>${esc(p.caption || "")}<button id="cap-close">close</button>`;
    $("cap-close").addEventListener("click", () => setCaption(null));
  }

  function applyPreset(p) {
    S.layout = L[p.layout] ? p.layout : S.layout;
    $("layout").value = S.layout;
    if (Array.isArray(p.pcs) && L[S.layout].choose) S.cols[S.layout] = p.pcs; else delete S.cols[S.layout];   // the example's own axes
    fillPcs();
    if (p.color_by && F[p.color_by] && !p.gse) { S.color = p.color_by; $("color").value = S.color; }
    S.selected = null; S.hl = null; S.onlyGroup = null; S.hideOthers = !!p.hide_others;
    setCaption(p);
    enterStudy(p.gse, p.color_by || "condition");
  }

  function search(q) {
    q = (q || "").trim();
    if (!q) return;
    const up = q.toUpperCase();
    if (D.studies[up]) { enterStudy([up]); return; }
    if (idIndex[up] !== undefined) { select({ kind: "atlas", i: idIndex[up] }); enterStudy([gseOf(idIndex[up])], null, true); return; }
    const oj = OV ? OV.sample.findIndex(s => s.toUpperCase() === up) : -1;
    if (oj >= 0) { select({ kind: "overlay", i: oj }); return; }
    const words = q.toLowerCase().split(/\s+/);
    const hits = Object.entries(D.studies).filter(([g, s]) => {
      const t = ((s.title || "") + " " + (s.summary || "")).toLowerCase();
      return words.every(w => t.includes(w));
    }).sort((a, b) => b[1].n - a[1].n);
    const el = $("tab-study");
    el.innerHTML = `<h2>Studies matching “${esc(q)}”</h2><p class="muted">${hits.length} studies</p><div class="results">` +
      hits.slice(0, 200).map(([g, s]) => `<div data-gse="${esc(g)}"><b>${esc(g)}</b> (${s.n}) ${esc(s.title || "")}</div>`).join("") + "</div>";
    el.querySelectorAll("[data-gse]").forEach(d => d.addEventListener("click", () => enterStudy([d.dataset.gse])));
    showTab("study");
  }

  // ------------------------------------------------------------------ study mode
  function studyColorOptions(gses) {
    const opts = ["condition"];
    if (F.title_group) opts.push("title_group");
    const vary = new Set();
    gses.forEach(g => (D.studies[g].varying || []).forEach(c => vary.add(c)));
    vary.forEach(c => { if (F[c]) opts.push(c); });
    if (gses.length > 1) opts.push("gse");
    ["phase_call", "p_stationary", "medium_call", "p_minimal", "lab_strain_call", "p_lab", "split"].forEach(c => { if (F[c]) opts.push(c); });
    return opts;
  }

  function enterStudy(gses, colorBy, keepSelection) {
    gses = gses.filter(g => D.studies[g]);
    if (!gses.length) return;
    S.study = gses; S.onlyGroup = null;
    const opts = studyColorOptions(gses);
    if (colorBy && F[colorBy]) S.studyColor = colorBy;
    else if (!opts.includes(S.studyColor)) S.studyColor = "condition";
    else if (!colorBy) S.studyColor = defaultStudyColor(gses);
    if (!keepSelection) S.selected = null;
    draw(); renderStudy(); if (!keepSelection) renderSample();
    showTab("study"); writeHash();
    flyTo(studyIndices(gses).filter(visible));
  }

  function defaultStudyColor(gses) {     // label conditions when they split the study, else GEO titles
    const idx = studyIndices(gses);
    const nCond = new Set(idx.map(i => val("condition", i))).size;
    if (nCond > 1 || !F.title_group) return "condition";
    const nTitle = new Set(idx.map(i => val("title_group", i))).size;
    return nTitle > 1 && nTitle <= Math.max(2, idx.length / 1.5) ? "title_group" : "condition";
  }

  const studyIndices = gses => [].concat(...gses.map(g => studySamples[g] || []));

  // ------------------------------------------------------------------ drawing
  function visible(i) { return !F.split || S.splits.has(val("split", i)); }

  function colorGroups(field, idx) {     // [{name, color, idx}] for a categorical field; numeric -> null
    const f = F[field];
    if (!f || f.kind !== "cat") return null;
    const by = new Map();
    idx.forEach(i => { const v = val(field, i); const k = v === "" ? null : v; if (!by.has(k)) by.set(k, []); by.get(k).push(i); });
    let names = [...by.keys()].filter(k => k !== null);
    const ordered = (field === "condition" || field === "title_group" || field === "growth_temperature") ? names.sort(natural)
      : names.sort((a, b) => by.get(b).length - by.get(a).length || natural(a, b));
    const groups = [];
    const keep = ordered.slice(0, MAX_LEVELS), rest = ordered.slice(MAX_LEVELS);
    keep.forEach((k, n) => groups.push({ name: k, color: PALETTE[n % PALETTE.length], idx: by.get(k) }));
    if (rest.length) groups.push({ name: `other (${rest.length} values)`, color: OTHER, idx: [].concat(...rest.map(k => by.get(k))) });
    if (by.has(null)) groups.push({ name: "not given", color: MISSING, idx: by.get(null) });
    return groups;
  }

  function hashColor(s) {
    let h = 0; for (let k = 0; k < s.length; k++) h = (h * 31 + s.charCodeAt(k)) | 0;
    return PALETTE[Math.abs(h) % PALETTE.length];
  }

  function column(lay, key, c) {       // accessor of column c of a layout (stored here or in its base layout)
    const own = lay.base ? lay.own : (lay.ncol || 3);
    if (lay.base && c >= own) return column(L[lay.base], key, c - own);
    const a = lay[key] || [];
    return i => a[own * i + c];
  }
  function xyz(lay, idx, key) {        // coordinates of samples idx in the three columns drawn
    key = key || "atlas";
    const [fx, fy, fz] = colsOf(lay).map(c => column(lay, key, c));
    return { x: idx.map(fx), y: idx.map(fy), z: idx.map(fz) };
  }

  let regNow = null;                   // the regulon values being drawn (atlas samples), for the hover text
  const fmtReg = v => (v > 0 ? "+" : "") + v.toFixed(2) + " SD";
  const hoverText = i => (regNow && regNow[i] != null) ? `${hover[i]}<br><b>${esc(R.sets[S.regulon].name)}</b>: ${fmtReg(regNow[i])}` : hover[i];

  function pointTrace(lay, idx, opts) {
    const c = xyz(lay, idx);
    return Object.assign({
      type: "scatter3d", mode: "markers", x: c.x, y: c.y, z: c.z, customdata: idx, text: idx.map(hoverText),
      hoverinfo: "text", hoverlabel: { align: "left" }, marker: { size: S.size, opacity: 0.85, line: { width: 0 } }
    }, opts);
  }

  function numericTrace(lay, idx, field, name, size) {
    const f = F[field];
    const have = idx.filter(i => f.values[i] != null), miss = idx.filter(i => f.values[i] == null);
    const isP = field.startsWith("p_");
    const out = [pointTrace(lay, have, {
      name, showlegend: false,
      marker: { size, opacity: 0.9, color: have.map(i => f.values[i]), colorscale: isP ? "RdBu" : "Viridis", reversescale: isP,
        cmin: isP ? 0 : undefined, cmax: isP ? 1 : undefined, colorbar: { title: { text: nice(field), side: "right" }, thickness: 12, len: 0.6 } }
    })];
    if (miss.length) out.push(pointTrace(lay, miss, { name: "not given", marker: { size, color: MISSING, opacity: 0.6 } }));
    return out;
  }

  function regulonTraces(lay, idx, size) {   // atlas samples coloured by the set's value, on the set's own symmetric scale
    const t = R.sets[S.regulon], v = regNow;
    const have = idx.filter(i => v[i] != null), miss = idx.filter(i => v[i] == null);
    const out = [pointTrace(lay, have, {
      name: t.name, showlegend: false,
      marker: { size, opacity: 0.9, color: have.map(i => v[i]), colorscale: DIVERGING, cmin: -t.limit, cmax: t.limit,
        colorbar: { title: { text: `${t.name} · mean of ${t.genes.length} gene${t.genes.length === 1 ? "" : "s"} (SD from the atlas mean)`, side: "right" }, thickness: 12, len: 0.6 } }
    })];
    if (miss.length) out.push(pointTrace(lay, miss, { name: "none of the set's genes", marker: { size, color: MISSING, opacity: 0.6 } }));
    return out;
  }

  function overlayTraces(lay) {        // your samples: diamonds, by set, by a sheet column or call, or by the regulon value
    const base = (jj, name, marker) => {
      const c = xyz(lay, jj, "overlay");
      return { type: "scatter3d", mode: "markers", name, x: c.x, y: c.y, z: c.z, customdata: jj.map(j => -1 - j),
        text: jj.map(j => ovHover[j] + (ovReg && ovReg[j] != null ? `<br><b>${esc(R.sets[S.regulon].name)}</b>: ${fmtReg(ovReg[j])}` : "")),
        hoverinfo: "text", hoverlabel: { align: "left" },
        marker: Object.assign({ size: S.size * 2.6, symbol: "diamond", line: { width: 1, color: "#000" } }, marker) };
    };
    const all = OV.ids.map((_, j) => j), out = [];
    const ovReg = S.regulon !== null ? regulonValues(S.regulon, true) : null;
    if (ovReg) {
      const t = R.sets[S.regulon], have = all.filter(j => ovReg[j] != null);
      if (have.length) {
        out.push(base(have, `your samples (${have.length})`, { color: have.map(j => ovReg[j]), colorscale: DIVERGING, cmin: -t.limit, cmax: t.limit, showscale: false }));
        const rest = all.filter(j => ovReg[j] == null);
        if (rest.length) out.push(base(rest, `your samples without a value (${rest.length})`, { color: MISSING }));
        return out;
      }
    }
    const field = S.ovColor;
    if (field && OVF[field] && OVF[field].kind === "num") {
      const have = all.filter(j => ovVal(field, j) !== ""), rest = all.filter(j => ovVal(field, j) === "");
      out.push(base(have, `your samples: ${field} (${have.length})`, { color: have.map(j => ovVal(field, j)), colorscale: "Viridis", showscale: false }));
      if (rest.length) out.push(base(rest, `your samples: not given (${rest.length})`, { color: MISSING }));
      return out;
    }
    const by = new Map();
    all.forEach(j => { const v = field ? (ovVal(field, j) === "" ? "not given" : String(ovVal(field, j))) : OV.set[j]; if (!by.has(v)) by.set(v, []); by.get(v).push(j); });
    const keys = field ? [...by.keys()].sort(natural) : [...by.keys()];
    ovColors = {};
    keys.forEach((k, n) => {
      const color = k === "not given" && field ? MISSING : (keys.length <= OVERLAY.length ? OVERLAY[n] : PALETTE[n % PALETTE.length]);
      ovColors[k] = color;
      out.push(base(by.get(k), `your samples: ${k} (${by.get(k).length})`, { color }));
    });
    return out;
  }
  let ovColors = {};                   // group of your samples -> colour (for the "Your samples" tab)

  function lineTraces(lay) {           // time courses: a line through each series in its order, a cross on its last sample
    if (!S.lines) return [];
    const out = [];
    const one = (list, key, color, name) => {
      if (!list.length) return;
      const x = [], y = [], z = [], ex = [], ey = [], ez = [];
      list.forEach(sr => {
        const c = xyz(lay, sr.idx, key);
        x.push(...c.x, null); y.push(...c.y, null); z.push(...c.z, null);
        const n = sr.idx.length - 1; ex.push(c.x[n]); ey.push(c.y[n]); ez.push(c.z[n]);
      });
      out.push({ type: "scatter3d", mode: "lines", name: `${name} (${list.length})`, x, y, z, hoverinfo: "skip", connectgaps: false,
        line: { color, width: key === "overlay" ? 2.5 : 4 } });
      out.push({ type: "scatter3d", mode: "markers", name: "last sample of a time course", showlegend: false, x: ex, y: ey, z: ez, hoverinfo: "skip",
        marker: { size: S.size * 2.2, symbol: "cross", color, line: { width: 0 } } });
    };
    if (S.study && D.series) {
      const inStudy = new Set(studyIndices(S.study));
      one(D.series.map(sr => Object.assign({}, sr, { idx: sr.idx.filter(i => inStudy.has(i) && visible(i)) })).filter(sr => sr.idx.length > 1),
        "atlas", "#444", "time courses");
    }
    if (OV && OV.series) one(OV.series, "overlay", "#7a0043", "your time courses");
    return out;
  }

  function categoricalTraces(lay, idx, field, size, legendgroup) {
    if (field === "gse" && new Set(idx.map(gseOf)).size > MAX_LEVELS) {
      return [pointTrace(lay, idx, { name: "studies", showlegend: false, marker: { size, opacity: 0.85, color: idx.map(i => hashColor(gseOf(i))) } })];
    }
    return colorGroups(field, idx).map(g => pointTrace(lay, g.idx, {
      name: `${g.name} (${g.idx.length})`, legendgroup, marker: { size, color: g.color, opacity: 0.9, line: { width: 0 } }
    }));
  }

  let groupColors = {};                // study mode: group value -> colour (for the study panel)

  function buildTraces() {
    const lay = L[S.layout];
    const traces = [];
    const all = [];
    for (let i = 0; i < N; i++) if (visible(i)) all.push(i);
    groupColors = {};
    regNow = S.regulon !== null ? regulonValues(S.regulon) : null;
    if (S.study) {
      const inStudy = new Set(studyIndices(S.study));
      const field = S.studyColor;
      let idx = all.filter(i => inStudy.has(i));
      if (!S.hideOthers) {
        const bg = all.filter(i => !inStudy.has(i));
        traces.push(pointTrace(lay, bg, { name: "other atlas samples", marker: { size: Math.max(1, S.size * 0.7), color: FADED, opacity: 0.25 } }));
      }
      const big = S.size * 2.2;
      if (regNow) traces.push(...regulonTraces(lay, idx, big));
      else if (F[field] && F[field].kind === "num") traces.push(...numericTrace(lay, idx, field, nice(field), big));
      else {
        let groups = colorGroups(field, idx) || [];
        groups.forEach(g => { groupColors[g.name.replace(/ \(\d+ values\)$/, "")] = g.color; });
        groups.forEach(g => {
          const off = S.onlyGroup !== null && g.name !== S.onlyGroup;
          traces.push(pointTrace(lay, g.idx, {
            name: `${g.name} (${g.idx.length})`,
            marker: { size: off ? S.size : big, color: off ? FADED : g.color, opacity: off ? 0.35 : 1, line: { width: off ? 0 : 0.5, color: "#333" } }
          }));
        });
      }
    } else {
      const field = S.color;
      if (regNow) traces.push(...regulonTraces(lay, all, S.size));
      else if (F[field].kind === "num") traces.push(...numericTrace(lay, all, field, nice(field), S.size));
      else traces.push(...categoricalTraces(lay, all, field, S.size));
    }
    traces.push(...lineTraces(lay));
    if (OV) traces.push(...overlayTraces(lay));
    const mark = [];
    if (S.selected) mark.push(S.selected);
    if (S.hl !== null) mark.push({ kind: "atlas", i: S.hl });
    if (S.selected && S.selected.kind === "overlay") OV.neighbors[S.selected.i].forEach(([i]) => mark.push({ kind: "nb", i }));
    if (mark.length) {
      const x = [], y = [], z = [], sz = [], col = [];
      mark.forEach(m => {
        const c = xyz(lay, [m.i], m.kind === "overlay" ? "overlay" : "atlas");
        x.push(c.x[0]); y.push(c.y[0]); z.push(c.z[0]);
        sz.push(m.kind === "nb" ? S.size * 3 : S.size * 5); col.push(m.kind === "nb" ? "#e6007e" : "#000");
      });
      traces.push({ type: "scatter3d", mode: "markers", name: "selected", showlegend: false, hoverinfo: "skip", x, y, z,
        marker: { size: sz, symbol: "circle-open", color: col, line: { width: 3, color: col } } });
    }
    return traces;
  }

  function sceneLayout() {
    const lay = L[S.layout];
    const ax = t => ({ title: { text: t, font: { size: 11 } }, showspikes: false, zeroline: false, gridcolor: "#e6e6e6", backgroundcolor: "#fafaf8" });
    return {
      uirevision: S.layout, margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "#fafaf8",
      legend: { x: 0, y: 1, bgcolor: "rgba(255,255,255,0.75)", font: { size: 11 }, itemsizing: "constant" },
      scene: { xaxis: ax(lay.axes[colsOf(lay)[0]]), yaxis: ax(lay.axes[colsOf(lay)[1]]), zaxis: ax(lay.axes[colsOf(lay)[2]]), aspectmode: "cube" },
      hovermode: "closest"
    };
  }
  const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["resetCameraLastSave3d"] };

  // Motion. A change of layout or axes moves the points from their old places to the new ones; every other
  // redraw is immediate. Each draw gets a number, so a later draw ends a running move.
  const MOVE_MS = 650, MAX_MOVING = 60000;
  const still = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let drawNo = 0, lastDraw = Promise.resolve(), moving = false;
  const positions = () => buildTraces().map(t => ({ x: t.x, y: t.y, z: t.z }));
  const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

  function draw(from) {                // from: positions() taken before the change of layout, to move from
    const no = ++drawNo;
    const traces = buildTraces();
    const n = traces.reduce((a, t) => a + t.x.length, 0);
    const can = from && !still() && n <= MAX_MOVING && from.length === traces.length && from.every((f, k) => f.x.length === traces[k].x.length);
    if (!can) { moving = false; lastDraw = Plotly.react("plot", traces, sceneLayout(), CONFIG); return; }
    const to = traces.map(t => ({ x: t.x, y: t.y, z: t.z }));
    traces.forEach((t, k) => { t.x = from[k].x; t.y = from[k].y; t.z = from[k].z; });
    moving = true;
    const mix = (a, b, w) => a.map((v, i) => (v == null || b[i] == null) ? b[i] : v + (b[i] - v) * w);
    lastDraw = Plotly.react("plot", traces, sceneLayout(), CONFIG).then(() => new Promise(done => {
      const t0 = performance.now();
      const step = () => {
        if (no !== drawNo) { done(); return; }                    // a later draw took over
        const w = Math.min(1, (performance.now() - t0) / MOVE_MS), e = ease(w);
        Plotly.restyle("plot", { x: to.map((t, k) => w < 1 ? mix(from[k].x, t.x, e) : t.x), y: to.map((t, k) => w < 1 ? mix(from[k].y, t.y, e) : t.y),
          z: to.map((t, k) => w < 1 ? mix(from[k].z, t.z, e) : t.z) }).then(() => {
          if (w < 1 && no === drawNo) requestAnimationFrame(step); else { if (no === drawNo) moving = false; done(); }
        });
      };
      requestAnimationFrame(step);
    }));
  }

  // Camera. plotly's camera lives in the scene's own units: the axis ranges fill a cube of side 1 around 0.
  const HOME = { center: { x: 0, y: 0, z: 0 }, eye: { x: 1.25, y: 1.25, z: 1.25 }, up: { x: 0, y: 0, z: 1 } };
  const CAM_MS = 700;
  let camNo = 0;
  const camera = () => { const c = (plot._fullLayout.scene || {}).camera || HOME; return JSON.parse(JSON.stringify({ center: c.center, eye: c.eye, up: c.up })); };
  function sceneUnits(c) {             // data coordinates -> scene units
    const sc = plot._fullLayout.scene, out = {};
    ["x", "y", "z"].forEach(a => { const r = sc[a + "axis"].range; out[a] = (c[a] - (r[0] + r[1]) / 2) / (r[1] - r[0]); });
    return out;
  }
  function moveCamera(target, ms) {
    const no = ++camNo;
    setRotate(false);
    lastDraw.then(() => {
      if (no !== camNo) return;
      if (still() || !ms) { Plotly.relayout("plot", { "scene.camera": target }); return; }
      const from = camera(), t0 = performance.now();
      const mixP = (a, b, w) => ({ x: a.x + (b.x - a.x) * w, y: a.y + (b.y - a.y) * w, z: a.z + (b.z - a.z) * w });
      const step = () => {
        if (no !== camNo) return;
        const w = Math.min(1, (performance.now() - t0) / ms), e = ease(w);
        Plotly.relayout("plot", { "scene.camera": { center: mixP(from.center, target.center, e), eye: mixP(from.eye, target.eye, e), up: target.up } })
          .then(() => { if (w < 1) requestAnimationFrame(step); });
      };
      requestAnimationFrame(step);
    });
  }
  function flyTo(idx, key) {           // turn the camera to samples idx: their centre in the middle, as close as their spread allows
    if (!idx.length) return;
    lastDraw.then(() => {
      const c = xyz(L[S.layout], idx, key || "atlas"), u = { x: [], y: [], z: [] };
      idx.forEach((_, k) => { const p = sceneUnits({ x: c.x[k], y: c.y[k], z: c.z[k] }); u.x.push(p.x); u.y.push(p.y); u.z.push(p.z); });
      const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
      const center = { x: mean(u.x), y: mean(u.y), z: mean(u.z) };
      const radius = Math.max(...idx.map((_, k) => Math.hypot(u.x[k] - center.x, u.y[k] - center.y, u.z[k] - center.z)));
      const now = camera(), d = { x: now.eye.x - now.center.x, y: now.eye.y - now.center.y, z: now.eye.z - now.center.z };
      const len = Math.hypot(d.x, d.y, d.z) || 1, dist = Math.min(2.2, Math.max(0.9, 0.7 + 2.6 * radius));
      moveCamera({ center, eye: { x: center.x + d.x / len * dist, y: center.y + d.y / len * dist, z: center.z + d.z / len * dist }, up: now.up }, CAM_MS);
    });
  }
  function flyHome() { moveCamera(HOME, CAM_MS); }

  function setRotate(on) {             // turn the scene slowly about the vertical axis until the next mouse press on the plot
    if (on === S.rotate) return;
    S.rotate = on;
    $("rotate").classList.toggle("on", on);
    if (!on) return;
    camNo++;
    let last = performance.now();
    const step = () => {
      if (!S.rotate) return;
      const t = performance.now(), a = (t - last) / 1000 * 0.35; last = t;      // radians per second
      const c = camera(), dx = c.eye.x - c.center.x, dy = c.eye.y - c.center.y;
      c.eye.x = c.center.x + dx * Math.cos(a) - dy * Math.sin(a); c.eye.y = c.center.y + dx * Math.sin(a) + dy * Math.cos(a);
      Plotly.relayout("plot", { "scene.camera": c }).then(() => { if (S.rotate) requestAnimationFrame(step); });
    };
    lastDraw.then(() => requestAnimationFrame(step));
  }

  // ------------------------------------------------------------------ panels
  function link(url, label) { return `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`; }
  const geo = acc => link(`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${acc}`, acc);

  const sortedCache = {};
  function percentile(name, v) {      // share of atlas samples with a lower value (numeric field)
    if (v === "" || v == null || !F[name] || F[name].kind !== "num") return "";
    let a = sortedCache[name];
    if (!a) { a = F[name].values.filter(x => x != null).sort((x, y) => x - y); sortedCache[name] = a; }
    let lo = 0, hi = a.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
    return a.length ? Math.round(100 * lo / a.length) : "";
  }
  const fmtNum = v => {
    const x = Number(v);
    if (Number.isInteger(x)) return String(x);
    return Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(3);
  };
  const ordinal = n => n + ((n % 100 >= 11 && n % 100 <= 13) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"));

  function qcTable(get) {             // get(fieldName) -> value of the sample ('' when missing)
    const names = D.meta.qc_fields || [];
    if (!names.length) return "";
    let rows = "";
    names.forEach(n => {
      const v = get(n);
      if (v === "" || v == null) return;
      const num = F[n] && F[n].kind === "num";
      const pc = num ? percentile(n, v) : "";
      rows += `<tr><td class="k">${esc(nice(n))}</td><td>${esc(num ? fmtNum(v) : v)}</td><td class="basis">${pc === "" ? "" : ordinal(pc) + " pct"}</td></tr>`;
    });
    return rows ? `<h3>QC covariates</h3><table>${rows}</table><p class="muted">percentile among the atlas samples; these values enter the model as covariates</p>` : "";
  }

  function select(sel) {
    S.selected = sel; S.hl = null;
    draw(); renderSample(); showTab("sample"); writeHash();
  }

  function callsTable(i) {
    let rows = "";
    Object.keys(HEAD_POS).forEach(h => {
      rows += `<tr><td class="k">${esc(h.replace("_", " "))}</td><td>${esc(val(h + "_call", i))}</td><td>${fmtP(val("p_" + HEAD_POS[h], i))}</td>${D.meta.n_seeds ? `<td>${seedsOf(h, i)}</td>` : ""}<td>${esc(val(h + "_label", i)) || "<span class='muted'>–</span>"}</td></tr>`;
    });
    return `<table><tr><th>call</th><th>model</th><th>p(${Object.values(HEAD_POS).join("/")})</th>${D.meta.n_seeds ? "<th title='seeds of the model that make the same call'>seeds</th>" : ""}<th>curated label</th></tr>${rows}</table>`;
  }

  function renderSample() {
    const el = $("tab-sample");
    const sel = S.selected;
    if (!sel) { el.innerHTML = `<p class="muted">Hover over a point to see the sample; click it to open it here.</p>`; return; }
    if (sel.kind === "overlay") { renderOverlaySample(sel.i); return; }
    const i = sel.i, g = gseOf(i), st = D.studies[g] || {};
    let h = `<h2>${geo(D.ids[i])}</h2><p>${esc(text("title", i))}</p>`;
    h += `<p class="muted">${geo(g)} · ${esc(st.title || "")}</p>`;
    h += `<div class="row-actions"><button class="primary" id="go-study">Show this study (${st.n || ""} samples)</button>`;
    if (F.split) h += `<span class="pill">${esc(val("split", i))}</span>`;
    if (val("study_flag", i)) h += `<span class="pill warn">${esc(val("study_flag", i))}</span>`;
    h += `</div>`;
    h += `<h3>Model</h3>${callsTable(i)}`;
    const extra = [];
    if (val("phase_logit", i) !== "") extra.push(`phase logit ${Number(val("phase_logit", i)).toFixed(2)}`);
    if (val("n_missing_genes", i) !== "") extra.push(`${val("n_missing_genes", i)} genes called missing`);
    if (extra.length) h += `<p class="muted">${esc(extra.join(" · "))}</p>`;
    if (val("condition", i)) h += `<p class="muted">condition within the study: ${esc(val("condition", i))}</p>`;
    if (regNow) h += `<p><b>${esc(R.sets[S.regulon].name)}</b>: ${regNow[i] == null ? "none of the set's genes" : fmtReg(regNow[i])}</p>`;
    h += `<h3>Curated metadata</h3><table>`;
    D.meta.label_fields.forEach(c => {
      const v = val(c, i);
      if (v === "" || c === "study_flag") return;
      const b = val(c + "_basis", i);
      h += `<tr><td class="k">${esc(c.replace(/_/g, " "))}</td><td>${esc(v)}</td><td class="basis">${esc(b)}</td></tr>`;
    });
    h += `</table>`;
    h += qcTable(n => val(n, i));
    if (D.meta.has_text) {
      h += `<h3>GEO sample record</h3><table>`;
      [["source", "source_name"], ["characteristics", "characteristics"], ["description", "description"], ["layout", "layout"], ["instrument", "instrument"]].forEach(([k, f]) => {
        const v = text(f, i); if (v) h += `<tr><td class="k">${k}</td><td>${esc(v)}</td></tr>`;
      });
      const runs = text("runs", i);
      if (runs) h += `<tr><td class="k">runs</td><td>${runs.split(/[,;\s]+/).filter(Boolean).map(r => link(`https://www.ncbi.nlm.nih.gov/sra/${r}`, r)).join(" ")}</td></tr>`;
      h += `</table>`;
    }
    el.innerHTML = h;
    $("go-study").addEventListener("click", () => enterStudy([g], null, true));
  }

  function renderOverlaySample(j) {
    const el = $("tab-sample");
    const sa = OV.seed_agreement || null;
    let h = `<h2>${esc(OV.sample[j])}</h2><p class="muted">your samples: ${esc(OV.set[j])}</p>`;
    const sheet = Object.keys(OVF).filter(c => ovVal(c, j) !== "");
    if (sheet.length) h += `<h3>Sample sheet</h3><table>` + sheet.map(c => `<tr><td class="k">${esc(c.replace(/_/g, " "))}</td><td>${esc(ovVal(c, j))}</td></tr>`).join("") + `</table>`;
    h += `<h3>Model</h3><table><tr><th>call</th><th>model</th><th>p</th>${sa ? "<th title='seeds of the model that make the same call'>seeds</th>" : ""}</tr>`;
    Object.keys(OV.calls || {}).forEach(k => {
      const c = OV.calls[k];
      h += `<tr><td class="k">${esc(k.replace("_", " "))}</td><td>${esc(c.call[j])}</td><td>${fmtP(c.p[j])}</td>${sa ? `<td>${sa[k] && sa[k][j] != null ? sa[k][j] + "/" + OV.n_seeds[j] : ""}</td>` : ""}</tr>`;
    });
    h += `</table>`;
    const rv = S.regulon !== null ? regulonValues(S.regulon, true) : null;
    if (rv) h += `<p><b>${esc(R.sets[S.regulon].name)}</b>: ${rv[j] == null ? "no value" : fmtReg(rv[j])}</p>`;
    if (OV.n_missing_genes) h += `<p class="muted">${OV.n_missing_genes[j]} genes called missing</p>`;
    if (OV.covariates) h += qcTable(n => { const c = OV.covariates[n]; return c ? (c[j] == null ? "" : c[j]) : ""; });
    h += `<h3>Nearest atlas samples (${esc(OV.metric)} distance, whole embedding)</h3><table><tr><th>#</th><th>sample</th><th>study</th><th>dist</th><th>phase / medium</th></tr>`;
    OV.neighbors[j].forEach(([i, d], r) => {
      h += `<tr class="click" data-i="${i}"><td>${r + 1}</td><td>${esc(D.ids[i])}</td><td>${esc(gseOf(i))}</td><td>${d.toFixed(3)}</td><td>${esc(val("growth_phase", i) || val("phase_label", i))} / ${esc(val("base_medium", i) || val("medium_label", i))}</td></tr>`;
    });
    h += `</table>`;
    const studies = {};
    OV.neighbors[j].forEach(([i]) => { const g = gseOf(i); studies[g] = (studies[g] || 0) + 1; });
    h += `<h3>Their studies</h3>` + Object.entries(studies).sort((a, b) => b[1] - a[1]).map(([g, n]) =>
      `<div class="results"><div data-gse="${esc(g)}"><b>${esc(g)}</b> ×${n} ${esc((D.studies[g] || {}).title || "")}</div></div>`).join("");
    el.innerHTML = h;
    el.querySelectorAll("tr[data-i]").forEach(r => r.addEventListener("click", () => select({ kind: "atlas", i: +r.dataset.i })));
    el.querySelectorAll("[data-gse]").forEach(d => d.addEventListener("click", () => enterStudy([d.dataset.gse], null, true)));
  }

  function studyCard(g) {
    const s = D.studies[g] || {};
    let h = `<h2>${geo(g)} <span class="muted">${s.n} samples in the atlas</span></h2>`;
    if (s.title) h += `<p><b>${esc(s.title)}</b></p>`;
    const meta = [];
    if (s.pubmed_ids) meta.push("PubMed " + s.pubmed_ids.split(",").map(p => link(`https://pubmed.ncbi.nlm.nih.gov/${p.trim()}/`, p.trim())).join(", "));
    if (s.bioproject) meta.push(link(`https://www.ncbi.nlm.nih.gov/bioproject/${s.bioproject}`, s.bioproject));
    if (s.sra) meta.push(link(`https://www.ncbi.nlm.nih.gov/sra?term=${s.sra}`, s.sra));
    if (s.first_public || s.submission_date) meta.push(`public ${esc(s.first_public || s.submission_date)}`);
    if (meta.length) h += `<p class="muted">${meta.join(" · ")}</p>`;
    if (s.study_flag) h += `<p><span class="pill warn">${esc(s.study_flag)}</span></p>`;
    if (s.contributors) h += `<p class="muted">${esc(s.contributors)}</p>`;
    if (s.summary) h += `<details open><summary>Summary</summary><p>${esc(s.summary)}</p></details>`;
    if (s.overall_design) h += `<details open><summary>Overall design</summary><p>${esc(s.overall_design)}</p></details>`;
    if (s.curator_notes) h += `<details><summary>Curation notes (metadata v2)</summary><p>${esc(s.curator_notes.replace(/ \| /g, "\n\n"))}</p></details>`;
    if (!D.meta.has_studies) h += `<p class="muted">No study text in this page (build it with --studies).</p>`;
    if ((s.varying || []).length) h += `<p class="muted">labels that vary within the study: ${s.varying.map(c => `<span class="pill">${esc(c.replace(/_/g, " "))}</span>`).join("")}</p>`;
    return h;
  }

  function renderStudy() {
    const el = $("tab-study");
    if (!S.study) { el.innerHTML = `<p class="muted">Pick a study with the search box, an example, or the “Show this study” button of a sample.</p>`; return; }
    let h = S.study.map(studyCard).join("<hr>");
    const opts = studyColorOptions(S.study);
    h += `<h3>Samples</h3><div class="row-actions"><label class="muted">group and colour by <select id="study-color">` +
      opts.map(o => `<option value="${o}" ${o === S.studyColor ? "selected" : ""}>${esc(nice(o))}</option>`).join("") +
      `</select></label><label class="muted"><input type="checkbox" id="hide-others" ${S.hideOthers ? "checked" : ""}> hide other samples</label></div>`;
    const idx = studyIndices(S.study);
    const field = S.studyColor;
    const groups = new Map();
    idx.forEach(i => {
      let v = val(field, i);
      if (F[field] && F[field].kind === "num") v = v === "" ? "" : Number(v).toFixed(2);
      const k = v === "" ? "not given" : String(v);
      if (!groups.has(k)) groups.set(k, []); groups.get(k).push(i);
    });
    const keys = [...groups.keys()].sort(natural);
    h += `<p class="muted">${keys.length} groups · click a group to show it alone, ▸ to list its samples; hover a sample to find it in the plot</p>`;
    keys.forEach(k => {
      const col = groupColors[k] || (k === "not given" ? MISSING : OTHER);
      const only = S.onlyGroup === k;
      const off = S.onlyGroup !== null && !only;
      h += `<div class="group ${only ? "only" : ""} ${off ? "off" : ""}" data-k="${esc(k)}"><div class="ghead"><span class="tog">▸</span><span class="swatch" style="background:${col}"></span>${esc(k)}<span class="gcount">${groups.get(k).length}</span></div><table>` +
        groups.get(k).sort((a, b) => natural(text("title", a) || D.ids[a], text("title", b) || D.ids[b])).map(i =>
          `<tr class="click" data-i="${i}"><td>${esc(D.ids[i])}</td><td>${esc(text("title", i))}</td><td>${esc(val("phase_call", i))}${val("p_stationary", i) !== "" ? " " + fmtP(val("p_stationary", i)) : ""}</td></tr>`).join("") +
        `</table></div>`;
    });
    el.innerHTML = h;
    $("study-color").addEventListener("change", e => { S.studyColor = e.target.value; S.onlyGroup = null; draw(); renderStudy(); });
    $("hide-others").addEventListener("change", e => { S.hideOthers = e.target.checked; draw(); });
    el.querySelectorAll(".group").forEach(gEl => {
      const k = gEl.dataset.k;
      gEl.querySelector(".tog").addEventListener("click", ev => { ev.stopPropagation(); gEl.classList.toggle("open"); ev.target.textContent = gEl.classList.contains("open") ? "▾" : "▸"; });
      gEl.querySelector(".ghead").addEventListener("click", () => {
        const isCat = F[field] && F[field].kind === "cat";
        S.onlyGroup = (!isCat || S.onlyGroup === k) ? null : k;
        draw(); renderStudy();
      });
      gEl.querySelectorAll("tr[data-i]").forEach(r => {
        r.addEventListener("mouseenter", () => { S.hl = +r.dataset.i; draw(); });
        r.addEventListener("mouseleave", () => { S.hl = null; draw(); });
        r.addEventListener("click", () => { S.selected = { kind: "atlas", i: +r.dataset.i }; S.hl = null; draw(); renderSample(); showTab("sample"); writeHash(); });
      });
    });
  }

  // ------------------------------------------------------------------ regulons
  function setRegulon(k, keepColor) {
    S.regulon = k;
    $("regulon").value = k === null ? "" : R.sets[k].name;
    draw(); renderRegulon(); renderSample(); if (S.study) renderStudy();
    if (k !== null) showTab("regulon");
    writeHash();
  }
  function regulonSearch(q) {          // a set by its name, or the sets that hold a gene
    q = (q || "").trim();
    if (!R || !q) return;
    const lo = q.toLowerCase();
    const exact = R.sets.findIndex(t => t.name.toLowerCase() === lo || t.term.toLowerCase() === lo);
    if (exact >= 0) { setRegulon(exact); return; }
    const hits = [];
    R.sets.forEach((t, k) => {
      const gene = t.genes.concat(t.absent).find(g => g.toLowerCase() === lo);
      if (gene) hits.push([k, `holds ${gene}`]);
      else if (t.name.toLowerCase().includes(lo)) hits.push([k, ""]);
    });
    hits.sort((x, y) => (x[1] !== "") - (y[1] !== ""));          // sets of that name first, then the sets that hold the gene
    if (hits.length === 1) { setRegulon(hits[0][0]); return; }
    renderRegulon({ q, hits }); showTab("regulon");
  }
  function renderRegulon(found) {
    const el = $("tab-regulon");
    let h = "";
    if (found) {
      h += `<h2>Regulon sets matching “${esc(found.q)}”</h2><p class="muted">${found.hits.length} sets (by name, or holding a gene of that name)</p><div class="results">` +
        found.hits.map(([k, why]) => `<div data-k="${k}"><b>${esc(R.sets[k].name)}</b> · ${R.sets[k].genes.length} genes ${why ? `<span class="muted">${esc(why)}</span>` : ""}</div>`).join("") + `</div>`;
    }
    if (S.regulon !== null) {
      const t = R.sets[S.regulon], n = t.genes.length;
      h += `<h2>${esc(t.name)}</h2><p class="muted">${esc(R.source || "RegulonDB")}${t.category ? " · " + esc(t.category) : ""}</p>`;
      h += `<p>The map is coloured by the mean expression of the ${n} gene${n === 1 ? "" : "s"} of this set: each gene's log expression as SDs from its mean over the atlas, averaged over the genes the sample has, compared with the sample's other genes. Red is above the atlas mean, blue below; the scale ends at ±${t.limit} SD.</p>`;
      if (n < 5) h += `<p><span class="pill warn">${n} gene${n === 1 ? "" : "s"}</span> A set this small is the expression of ${n === 1 ? "one gene" : "a few genes"}, not a regulon-wide signal.</p>`;
      h += `<p class="muted">This is the expression of the regulator's targets, not its activity; regulons that share targets rise and fall together.</p>`;
      h += `<div class="row-actions"><button id="regulon-off">Back to the colour menu</button></div>`;
      h += `<h3>Genes (${n})</h3><p class="genes">${t.genes.map(esc).join(" ") || "<span class='muted'>none in the model</span>"}</p>`;
      if (t.absent.length) h += `<h3>Not in the model (${t.absent.length})</h3><p class="genes muted">${t.absent.map(esc).join(" ")}</p>`;
    } else if (!found) {
      h += `<p class="muted">Type a regulator (Fur, CRP, RpoS …) or a gene in the Regulon box to colour the map by the expression of a RegulonDB set. ${R.sets.length} sets: the genes each regulator activates and those it represses.</p>`;
    }
    el.innerHTML = h;
    el.querySelectorAll("[data-k]").forEach(d => d.addEventListener("click", () => setRegulon(+d.dataset.k)));
    if ($("regulon-off")) $("regulon-off").addEventListener("click", () => setRegulon(null));
  }

  // ------------------------------------------------------------------ your samples
  function renderMine() {
    const el = $("tab-mine");
    const opts = [["", "set"]].concat(Object.keys(OVF).map(c => [c, c.replace(/_/g, " ")]), Object.keys(OV.calls || {}).map(k => ["call:" + k, k.replace("_", " ") + " call"]));
    let h = `<h2>Your samples <span class="muted">${NO}</span></h2>`;
    h += `<div class="row-actions"><label class="muted">colour by <select id="mine-color">` + opts.map(([v, t]) => `<option value="${esc(v)}" ${v === (S.ovColor || "") ? "selected" : ""}>${esc(t)}</option>`).join("") +
      `</select></label><button id="mine-fly">Turn the camera to them</button></div>`;
    if (!Object.keys(OVF).length) h += `<p class="muted">No sample sheet in this page (build it with --overlay-metadata to colour and group your samples by condition).</p>`;
    const heads = Object.keys(OV.calls || {});
    h += `<table><tr><th></th><th>sample</th>${S.ovColor ? `<th>${esc(S.ovColor.replace(/^call:/, "").replace(/_/g, " "))}</th>` : ""}` + heads.map(k => `<th>p(${esc(OV.calls[k].positive)})</th>`).join("") + `</tr>`;
    OV.ids.forEach((_, j) => {
      const g = S.ovColor ? (ovVal(S.ovColor, j) === "" ? "not given" : String(ovVal(S.ovColor, j))) : OV.set[j];
      h += `<tr class="click" data-j="${j}"><td><span class="swatch" style="background:${ovColors[g] || MISSING}"></span></td><td>${esc(OV.sample[j])}</td>${S.ovColor ? `<td>${esc(g)}</td>` : ""}` +
        heads.map(k => `<td>${fmtP(OV.calls[k].p[j])}</td>`).join("") + `</tr>`;
    });
    h += `</table>`;
    if ((OV.series || []).length) h += `<p class="muted">${OV.series.length} series joined by lines (the cross marks the last sample of each).</p>`;
    el.innerHTML = h;
    $("mine-color").addEventListener("change", e => { S.ovColor = e.target.value || null; draw(); renderMine(); });
    $("mine-fly").addEventListener("click", () => flyTo(OV.ids.map((_, j) => j), "overlay"));
    el.querySelectorAll("tr[data-j]").forEach(r => r.addEventListener("click", () => select({ kind: "overlay", i: +r.dataset.j })));
  }

  function renderAbout() {
    const m = D.meta;
    let h = `<h2>About this map</h2><p>${m.n_atlas.toLocaleString()} RNA-seq samples of <i>E. coli</i> in ${m.n_studies} GEO studies, embedded by the model <b>${esc(m.bundle)}</b> (ecoli_atlas ${esc(m.version)}, page built ${esc(m.built)}).</p>`;
    h += `<p>The embedding has ${m.embedding_dim} dimensions in blocks: ` + m.blocks.map(b => `<span class="pill">${esc(b.block)} ${b.end - b.start}-d</span>`).join("") + `. Each block of the first three is trained to carry one call (growth phase, medium, lab strain); the residual block carries the rest of the expression profile.</p>`;
    h += `<h3>Layouts</h3><table>` + D.layouts.map(l => `<tr><td class="k">${esc(l.label)}</td><td>${l.axes.slice(0, 3).map(esc).join(" · ")}${l.choose ? ` · … (${l.axes.length} columns; pick any three with X / Y / Z)` : ""}</td></tr>`).join("") + `</table>`;
    h += `<p class="muted">PCA and UMAP are fitted on the atlas samples; your samples are projected onto the same fit. Distances between neighbours are computed in the full embedding, not in the 3-D map.</p>`;
    h += `<h3>Using the map</h3><p>Drag to rotate, scroll to zoom, right-drag to pan. Click a legend entry to hide a group, double-click to show it alone. Click a point to open the sample; “Show this study” draws its study on top of the rest of the atlas, grouped by the labels that differ between its samples. The split filters show training samples, the held-out TEST studies and projected samples (in the atlas but not fitted).</p>`;
    if (OV) h += `<p>Your samples (${NO}) are drawn as diamonds; clicking one lists its nearest atlas samples. The “Your samples” tab lists them and colours them by a column of your sample sheet.</p>`;
    if (R) h += `<h3>Regulons</h3><p>The Regulon box colours the map by the expression of a RegulonDB set (${R.sets.length} sets from ${esc(R.source || "RegulonDB")}): the mean over the set's genes of their log expression, each gene as SDs from its mean over the atlas, minus the sample's median over all genes. Genes a strain lacks are left out for its samples.</p>`;
    if ((D.series || []).length) h += `<h3>Time courses</h3><p>In a study with time courses (${D.series.length} in the atlas) a line joins the samples of each course in their order; a cross marks the last one. “Lines” switches them off.</p>`;
    if (m.n_seeds) h += `<p class="muted">Seeds: the model was fitted ${m.n_seeds} times from different random starts; “seeds that agree” counts the fits that make the same call as the model of this page.</p>`;
    h += `<p class="muted">“Rotate” turns the scene until the next mouse press. A change of layout moves the points to their new places.</p>`;
    $("tab-about").innerHTML = h;
  }

  // ------------------------------------------------------------------ URL hash (study=, sample=, layout=)
  function writeHash() {
    const p = new URLSearchParams();
    if (S.study) p.set("study", S.study.join(","));
    if (S.selected && S.selected.kind === "atlas") p.set("sample", D.ids[S.selected.i]);
    if (S.layout) p.set("layout", S.layout);
    if (S.cols[S.layout]) p.set("pcs", S.cols[S.layout].join(","));
    if (S.regulon !== null) p.set("regulon", R.sets[S.regulon].term);
    try { history.replaceState(null, "", "#" + p.toString()); } catch (e) { /* file:// in some browsers */ }
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("layout") && L[p.get("layout")]) {
      S.layout = p.get("layout"); $("layout").value = S.layout;
      const l = L[S.layout], c = (p.get("pcs") || "").split(",").map(Number);
      if (l.choose && c.length === 3 && c.every((k, d) => l.choose[d].includes(k))) S.cols[l.name] = c;
      fillPcs();
    }
    if (R && p.get("regulon")) {
      const k = R.sets.findIndex(t => t.term === p.get("regulon"));
      if (k >= 0) { S.regulon = k; $("regulon").value = R.sets[k].name; renderRegulon(); }
    }
    const s = p.get("sample");
    if (s && idIndex[s] !== undefined) { S.selected = { kind: "atlas", i: idIndex[s] }; renderSample(); }
    if (p.get("study")) { enterStudy(p.get("study").split(","), null, true); return true; }
    return false;
  }

  // ------------------------------------------------------------------ start
  fillControls();
  if (!readHash()) draw();
  if (OV) renderMine();
  // plotly_click is not used: a 3-D scene emits it from inside its render loop (and again on every
  // re-render), so redrawing in its handler re-enters the handler until the stack overflows, and a
  // plain click often emits nothing. Instead the point under the mouse is tracked with hover events
  // and a real mouse click on the plot (not the end of a rotation) selects it, after the click event.
  let hovered = null, downAt = null;
  plot.on("plotly_hover", ev => {
    const pt = ev.points && ev.points[0];
    hovered = (pt && pt.customdata !== undefined && pt.customdata !== null) ? pt.customdata : null;
  });
  plot.on("plotly_unhover", () => { hovered = null; });
  plot.addEventListener("pointerdown", e => { downAt = [e.clientX, e.clientY]; setRotate(false); camNo++; }, true);
  plot.addEventListener("wheel", () => { setRotate(false); camNo++; }, { capture: true, passive: true });
  plot.addEventListener("pointerup", e => {
    const still = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5;
    downAt = null;
    if (!still || hovered === null || e.button !== 0) return;
    if (e.target && e.target.closest && e.target.closest(".modebar, .legend")) return;
    const c = hovered;
    setTimeout(() => select(c < 0 ? { kind: "overlay", i: -1 - c } : { kind: "atlas", i: c }), 0);
  }, true);
  window.atlasViewer = { data: D, state: S, draw, enterStudy, select, applyPreset, setRegulon, regulonSearch, regulonValues, flyTo, flyHome,
    setRotate, camera, isMoving: () => moving, ready: () => lastDraw };
})();
