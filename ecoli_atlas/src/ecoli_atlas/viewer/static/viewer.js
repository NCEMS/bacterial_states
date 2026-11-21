/* ecoli_atlas view: interactive 3-D map of the atlas. Data come from the JSON block #atlas-data
   (written by ecoli_atlas/viewer/build.py); drawing uses plotly.js scatter3d. */
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("atlas-data").textContent);
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
  const HEAD_POS = {};                // head -> positive class
  Object.entries(D.meta.heads).forEach(([h, v]) => { HEAD_POS[h] = v[1]; });
  const NICE = {
    split: "split", gse: "study (GSE)", condition: "condition (labels that vary in the study)", title_group: "GEO sample title (without replicate)",
    phase_label: "phase label (curated)", medium_label: "medium label (curated)", lab_strain_label: "lab-strain label (curated)",
    phase_call: "phase call (model)", medium_call: "medium call (model)", lab_strain_call: "lab-strain call (model)",
    p_stationary: "p(stationary)", p_minimal: "p(minimal medium)", p_lab: "p(lab strain)", phase_logit: "phase logit",
    n_missing_genes: "genes called missing"
  };
  const nice = n => NICE[n] || n.replace(/_/g, " ");
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
    caption: null
  };

  // ------------------------------------------------------------------ controls
  function fillControls() {
    $("subtitle").textContent = `${D.meta.n_atlas.toLocaleString()} samples · ${D.meta.n_studies} studies · model ${D.meta.bundle} · ${D.meta.embedding_dim}-d embedding`;
    const lay = $("layout");
    D.layouts.forEach(l => { const o = document.createElement("option"); o.value = l.name; o.textContent = l.label; lay.appendChild(o); });
    const col = $("color");
    const groups = [["Model", ["split", "phase_call", "medium_call", "lab_strain_call", "p_stationary", "p_minimal", "p_lab", "phase_logit", "n_missing_genes"]],
      ["Curated labels", ["phase_label", "medium_label", "lab_strain_label"].concat(D.meta.label_fields)], ["Study", ["gse"]]];
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
    lay.addEventListener("change", () => { S.layout = lay.value; draw(); writeHash(); });
    col.addEventListener("change", () => { S.color = col.value; draw(); });
    $("size").addEventListener("input", e => { S.size = +e.target.value; draw(); });
    $("reset").addEventListener("click", () => { S.study = null; S.selected = null; S.hl = null; S.onlyGroup = null; setCaption(null); draw(); renderSample(); renderStudy(); writeHash(); });
    pre.addEventListener("change", () => { if (pre.value !== "") applyPreset(D.presets[+pre.value]); });
    $("search").addEventListener("keydown", e => { if (e.key === "Enter") search($("search").value); });
    $("search").addEventListener("change", e => { if (D.studies[e.target.value.trim()]) search(e.target.value); });
    document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
    renderAbout();
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

  function xyz(lay, idx, key) {
    const a = lay[key || "atlas"]; const x = [], y = [], z = [];
    idx.forEach(i => { x.push(a[3 * i]); y.push(a[3 * i + 1]); z.push(a[3 * i + 2]); });
    return { x, y, z };
  }

  function pointTrace(lay, idx, opts) {
    const c = xyz(lay, idx);
    return Object.assign({
      type: "scatter3d", mode: "markers", x: c.x, y: c.y, z: c.z, customdata: idx, text: idx.map(i => hover[i]),
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

  function categoricalTraces(lay, idx, field, size, legendgroup) {
    if (field === "gse" && new Set(idx.map(gseOf)).size > MAX_LEVELS) {
      return [pointTrace(lay, idx, { name: "studies", showlegend: false, marker: { size, opacity: 0.85, color: idx.map(i => hashColor(gseOf(i))) } })];
    }
    return colorGroups(field, idx).map(g => pointTrace(lay, g.idx, {
      name: `${g.name} (${g.idx.length})`, legendgroup, marker: { size, color: g.color, opacity: 0.9, line: { width: 0 } }
    }));
  }

  let groupColors = {};                // study mode: group value -> colour (for the study panel)

  function draw() {
    const lay = L[S.layout];
    const traces = [];
    const all = [];
    for (let i = 0; i < N; i++) if (visible(i)) all.push(i);
    groupColors = {};
    if (S.study) {
      const inStudy = new Set(studyIndices(S.study));
      const field = S.studyColor;
      let idx = all.filter(i => inStudy.has(i));
      if (!S.hideOthers) {
        const bg = all.filter(i => !inStudy.has(i));
        traces.push(pointTrace(lay, bg, { name: "other atlas samples", marker: { size: Math.max(1, S.size * 0.7), color: FADED, opacity: 0.25 } }));
      }
      const big = S.size * 2.2;
      if (F[field] && F[field].kind === "num") traces.push(...numericTrace(lay, idx, field, nice(field), big));
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
      if (F[field].kind === "num") traces.push(...numericTrace(lay, all, field, nice(field), S.size));
      else traces.push(...categoricalTraces(lay, all, field, S.size));
    }
    if (OV) {
      const sets = [...new Set(OV.set)];
      sets.forEach((s, k) => {
        const jj = []; OV.set.forEach((v, j) => { if (v === s) jj.push(j); });
        const a = lay.overlay || [];
        traces.push({
          type: "scatter3d", mode: "markers", name: `your samples: ${s} (${jj.length})`, x: jj.map(j => a[3 * j]), y: jj.map(j => a[3 * j + 1]),
          z: jj.map(j => a[3 * j + 2]), customdata: jj.map(j => -1 - j), text: jj.map(j => ovHover[j]), hoverinfo: "text",
          hoverlabel: { align: "left" }, marker: { size: S.size * 2.6, symbol: "diamond", color: OVERLAY[k % OVERLAY.length], line: { width: 1, color: "#000" } }
        });
      });
    }
    const mark = [];
    if (S.selected) mark.push(S.selected);
    if (S.hl !== null) mark.push({ kind: "atlas", i: S.hl });
    if (S.selected && S.selected.kind === "overlay") OV.neighbors[S.selected.i].forEach(([i]) => mark.push({ kind: "nb", i }));
    if (mark.length) {
      const x = [], y = [], z = [], sz = [], col = [];
      mark.forEach(m => {
        const a = m.kind === "overlay" ? lay.overlay : lay.atlas; const i = m.i;
        x.push(a[3 * i]); y.push(a[3 * i + 1]); z.push(a[3 * i + 2]);
        sz.push(m.kind === "nb" ? S.size * 3 : S.size * 5); col.push(m.kind === "nb" ? "#e6007e" : "#000");
      });
      traces.push({ type: "scatter3d", mode: "markers", name: "selected", showlegend: false, hoverinfo: "skip", x, y, z,
        marker: { size: sz, symbol: "circle-open", color: col, line: { width: 3, color: col } } });
    }
    const ax = t => ({ title: { text: t, font: { size: 11 } }, showspikes: false, zeroline: false, gridcolor: "#e6e6e6", backgroundcolor: "#fafaf8" });
    const layout = {
      uirevision: S.layout, margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "#fafaf8",
      legend: { x: 0, y: 1, bgcolor: "rgba(255,255,255,0.75)", font: { size: 11 }, itemsizing: "constant" },
      scene: { xaxis: ax(lay.axes[0]), yaxis: ax(lay.axes[1]), zaxis: ax(lay.axes[2]), aspectmode: "cube" },
      hovermode: "closest"
    };
    Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["resetCameraLastSave3d"] });
  }

  // ------------------------------------------------------------------ panels
  function link(url, label) { return `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`; }
  const geo = acc => link(`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${acc}`, acc);

  function select(sel) {
    S.selected = sel; S.hl = null;
    draw(); renderSample(); showTab("sample"); writeHash();
  }

  function callsTable(i) {
    let rows = "";
    Object.keys(HEAD_POS).forEach(h => {
      rows += `<tr><td class="k">${esc(h.replace("_", " "))}</td><td>${esc(val(h + "_call", i))}</td><td>${fmtP(val("p_" + HEAD_POS[h], i))}</td><td>${esc(val(h + "_label", i)) || "<span class='muted'>–</span>"}</td></tr>`;
    });
    return `<table><tr><th>call</th><th>model</th><th>p(${Object.values(HEAD_POS).join("/")})</th><th>curated label</th></tr>${rows}</table>`;
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
    h += `<h3>Curated metadata</h3><table>`;
    D.meta.label_fields.forEach(c => {
      const v = val(c, i);
      if (v === "" || c === "study_flag") return;
      const b = val(c + "_basis", i);
      h += `<tr><td class="k">${esc(c.replace(/_/g, " "))}</td><td>${esc(v)}</td><td class="basis">${esc(b)}</td></tr>`;
    });
    h += `</table>`;
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
    let h = `<h2>${esc(OV.sample[j])}</h2><p class="muted">your samples: ${esc(OV.set[j])}</p><h3>Model</h3><table><tr><th>call</th><th>model</th><th>p</th></tr>`;
    Object.keys(OV.calls || {}).forEach(k => { const c = OV.calls[k]; h += `<tr><td class="k">${esc(k.replace("_", " "))}</td><td>${esc(c.call[j])}</td><td>${fmtP(c.p[j])}</td></tr>`; });
    h += `</table>`;
    if (OV.n_missing_genes) h += `<p class="muted">${OV.n_missing_genes[j]} genes called missing</p>`;
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

  function renderAbout() {
    const m = D.meta;
    let h = `<h2>About this map</h2><p>${m.n_atlas.toLocaleString()} RNA-seq samples of <i>E. coli</i> in ${m.n_studies} GEO studies, embedded by the model <b>${esc(m.bundle)}</b> (ecoli_atlas ${esc(m.version)}, page built ${esc(m.built)}).</p>`;
    h += `<p>The embedding has ${m.embedding_dim} dimensions in blocks: ` + m.blocks.map(b => `<span class="pill">${esc(b.block)} ${b.end - b.start}-d</span>`).join("") + `. Each block of the first three is trained to carry one call (growth phase, medium, lab strain); the residual block carries the rest of the expression profile.</p>`;
    h += `<h3>Layouts</h3><table>` + D.layouts.map(l => `<tr><td class="k">${esc(l.label)}</td><td>${l.axes.map(esc).join(" · ")}</td></tr>`).join("") + `</table>`;
    h += `<p class="muted">PCA and UMAP are fitted on the atlas samples; your samples are projected onto the same fit. Distances between neighbours are computed in the full embedding, not in the 3-D map.</p>`;
    h += `<h3>Using the map</h3><p>Drag to rotate, scroll to zoom, right-drag to pan. Click a legend entry to hide a group, double-click to show it alone. Click a point to open the sample; “Show this study” draws its study on top of the rest of the atlas, grouped by the labels that differ between its samples. The split filters show training samples, the held-out TEST studies and projected samples (in the atlas but not fitted).</p>`;
    if (OV) h += `<p>Your samples (${NO}) are drawn as diamonds; clicking one lists its nearest atlas samples.</p>`;
    $("tab-about").innerHTML = h;
  }

  // ------------------------------------------------------------------ URL hash (study=, sample=, layout=)
  function writeHash() {
    const p = new URLSearchParams();
    if (S.study) p.set("study", S.study.join(","));
    if (S.selected && S.selected.kind === "atlas") p.set("sample", D.ids[S.selected.i]);
    if (S.layout) p.set("layout", S.layout);
    try { history.replaceState(null, "", "#" + p.toString()); } catch (e) { /* file:// in some browsers */ }
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("layout") && L[p.get("layout")]) { S.layout = p.get("layout"); $("layout").value = S.layout; }
    const s = p.get("sample");
    if (s && idIndex[s] !== undefined) { S.selected = { kind: "atlas", i: idIndex[s] }; renderSample(); }
    if (p.get("study")) { enterStudy(p.get("study").split(","), null, true); return true; }
    return false;
  }

  // ------------------------------------------------------------------ start
  fillControls();
  if (!readHash()) draw();
  const plot = $("plot");
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
  plot.addEventListener("pointerdown", e => { downAt = [e.clientX, e.clientY]; }, true);
  plot.addEventListener("pointerup", e => {
    const still = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5;
    downAt = null;
    if (!still || hovered === null || e.button !== 0) return;
    if (e.target && e.target.closest && e.target.closest(".modebar, .legend")) return;
    const c = hovered;
    setTimeout(() => select(c < 0 ? { kind: "overlay", i: -1 - c } : { kind: "atlas", i: c }), 0);
  }, true);
  window.atlasViewer = { data: D, state: S, draw, enterStudy, select, applyPreset };
})();
