/* Rosetta — run browser, layer x head grid, component detail, chat. */
(function () {
"use strict";

// --------------------------------------------------------------- palette
// Documented sequential blue ramp (100 -> 700), light -> dark.
const SEQ = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7",
             "#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281",
             "#0d366b"];
// Diverging: blue <-> neutral gray <-> red, equal steps per arm.
const DIV_NEG = "#1c5cab", DIV_MID = "#f0efec", DIV_POS = "#c0322f";
const NO_DATA = "#f2f1ed";

// Categorical slots, documented order. Used only for label chip dots,
// which always ship with their text label beside them.
const CAT = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300",
             "#4a3aa7","#e34948"];

const PREFERRED_METRICS = ["induction","prev_token","sink_pos0","same_token",
                           "self_attn","entropy","dla_logit_diff"];

// ----------------------------------------------------------------- state
const S = {
  runs: [], run: null, metric: null,
  activeLabels: new Set(), focus: new Set(),
  selected: null, ws: null, tableView: false, labelColor: {},
};

const $ = (id) => document.getElementById(id);
const el$ = {
  runSelect: $("run-select"), runMeta: $("run-meta"),
  metricSelect: $("metric-select"), legend: $("scale-legend"),
  labelBar: $("label-bar"), gridHost: $("grid-host"),
  gridScroll: $("grid-scroll"), tableHost: $("table-host"),
  emptyState: $("empty-state"), viewToggle: $("view-toggle"),
  clearFocus: $("clear-focus"), tooltip: $("tooltip"),
  modelStatus: $("model-status"),
  detailEmpty: $("detail-empty"), detailContent: $("detail-content"),
  messages: $("messages"), chatForm: $("chat-form"), chatInput: $("chat-input"),
};

// ------------------------------------------------------------ colour fns
function hexToRgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16),
          parseInt(h.slice(5,7),16)];
}
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return `rgb(${Math.round(A[0]+(B[0]-A[0])*t)},` +
         `${Math.round(A[1]+(B[1]-A[1])*t)},` +
         `${Math.round(A[2]+(B[2]-A[2])*t)})`;
}
function seqColor(t) {                       // t in 0..1 -> magnitude ramp
  t = Math.max(0, Math.min(1, t));
  const x = t * (SEQ.length - 1);
  const i = Math.min(Math.floor(x), SEQ.length - 2);
  return mix(SEQ[i], SEQ[i+1], x - i);
}
function divColor(t) {                       // t in -1..1 -> polarity ramp
  t = Math.max(-1, Math.min(1, t));
  return t < 0 ? mix(DIV_MID, DIV_NEG, -t) : mix(DIV_MID, DIV_POS, t);
}

function scaleFor(metricKey) {
  const m = S.run && S.run.metrics[metricKey];
  if (!m) return null;
  if (m.signed) {
    const bound = Math.max(Math.abs(m.min), Math.abs(m.max)) || 1;
    return { signed: true, bound,
             color: (v) => divColor(v / bound),
             lo: -bound, hi: bound };
  }
  const lo = m.min, hi = m.max, span = (hi - lo) || 1;
  return { signed: false, lo, hi,
           color: (v) => seqColor((v - lo) / span) };
}

const fmt = (v) => {
  if (v === null || v === undefined) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(a < 1 ? 4 : 3);
};

// Legend ticks: coarser, and anything negligible reads as plain 0.
const fmtScale = (v, span) => {
  if (v === null || v === undefined) return "—";
  if (span && Math.abs(v) < span * 1e-4) return "0";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 10000)) return v.toPrecision(2);
  return String(Math.round(v * 100) / 100);
};

// Square cells sized to fit the viewport: a 3-layer toy run gets large
// cells, a 32-layer run shrinks until the whole model is visible at once.
function cellSize(n_layers, n_heads) {
  const box = el$.gridScroll.getBoundingClientRect();
  const availH = Math.max(160, box.height - 64);   // caption + column header
  const availW = Math.max(160, box.width - 36 - 52 - 36); // labels, MLP, pad
  const fitH = availH / n_layers - 2;
  const fitW = availW / n_heads - 2;
  return Math.round(Math.max(13, Math.min(54, fitH, fitW)));
}

// ------------------------------------------------------------------ init
async function init() {
  const r = await fetch("/api/runs").then(x => x.json());
  S.runs = r.runs;
  if (!S.runs.length) {
    el$.emptyState.textContent =
      "No runs found under runs/. Generate one with: python -m rosetta.cli demo";
    el$.emptyState.classList.remove("hidden");
    return;
  }
  el$.runSelect.innerHTML = S.runs
    .map(x => `<option value="${x.id}">${x.id}</option>`).join("");
  connectWs();
  await selectRun(S.runs[0].id);
}

// After an intervention appends evidence, pull the run again but keep the
// current metric, filters and selection in place.
async function refreshRun() {
  if (!S.run) return;
  const keep = {
    metric: S.metric, labels: new Set(S.activeLabels),
    focus: new Set(S.focus), selected: S.selected,
  };
  const [data, runs] = await Promise.all([
    fetch(`/api/runs/${S.run.id}`).then(x => x.json()),
    fetch("/api/runs").then(x => x.json()),
  ]);
  S.run = data;
  S.runs = runs.runs;
  buildMetricSelect();
  buildLabelBar();
  if (keep.metric && keep.metric in data.metrics) {
    S.metric = keep.metric;
    el$.metricSelect.value = keep.metric;
  }
  S.activeLabels = keep.labels;
  S.focus = keep.focus;
  S.selected = keep.selected;
  el$.labelBar.querySelectorAll(".chip").forEach(c =>
    c.setAttribute("aria-pressed", S.activeLabels.has(c.dataset.label)));
  setRunMeta();
  renderModelStatus();
  render();
  if (S.selected) selectComponent(S.selected);
}

function setRunMeta() {
  const s = S.runs.find(x => x.id === S.run.id) || {};
  el$.runMeta.textContent =
    `${S.run.n_layers} layers x ${S.run.n_heads} heads · ` +
    `${Object.keys(S.run.components).length} components · ` +
    `${(s.n_evidence || 0).toLocaleString()} evidence records` +
    (s.n_explanations ? ` · ${s.n_explanations} explanations` : "");
}

async function selectRun(id) {
  const data = await fetch(`/api/runs/${id}`).then(x => x.json());
  S.run = data;
  S.activeLabels.clear();
  S.focus.clear();
  S.selected = null;
  S.modelLoaded = false;   // selecting a run drops any loaded adapter
  el$.runSelect.value = id;
  el$.clearFocus.classList.add("hidden");
  showDetailEmpty();

  setRunMeta();
  buildMetricSelect();
  buildLabelBar();
  renderModelStatus();
  render();
  if (S.ws && S.ws.readyState === WebSocket.OPEN) {
    S.ws.send(JSON.stringify({ type: "select_run", run_id: id }));
  }
}

el$.runSelect.addEventListener("change", e => selectRun(e.target.value));

// Model is only needed for new interventions, so say plainly whether it is
// loaded and which checkpoint the run points at.
function renderModelStatus(state) {
  const m = (S.run && S.run.manifest && S.run.manifest.model) || {};
  const id = m.id || "unknown checkpoint";
  const short = id.includes("/") ? id.split("/").pop() : id;
  const cls = state || (S.modelLoaded ? "loaded" : "");
  const verb = state === "loading" ? "loading"
             : cls === "loaded" ? "loaded" : "not loaded";
  el$.modelStatus.className = "model-status " + cls;
  el$.modelStatus.innerHTML = `<span class="dot"></span>${esc(short)} · ${verb}`;
  el$.modelStatus.title = m.inferred
    ? `${id} (inferred from the run's shape; confirmed once it loads)`
    : `${id} (recorded in run.json)`;
}

// --------------------------------------------------------- metric select
function buildMetricSelect() {
  const keys = Object.keys(S.run.metrics);
  const pref = PREFERRED_METRICS.filter(k => keys.includes(k));
  const rest = keys.filter(k => !pref.includes(k)).sort();
  const opt = (k) => {
    const m = S.run.metrics[k];
    return `<option value="${k}">${k}${m.signed ? " (signed)" : ""}</option>`;
  };
  el$.metricSelect.innerHTML =
    (pref.length ? `<optgroup label="Attention & readout">${
        pref.map(opt).join("")}</optgroup>` : "") +
    (rest.length ? `<optgroup label="Other">${
        rest.map(opt).join("")}</optgroup>` : "");
  S.metric = pref[0] || rest[0] || null;
  if (S.metric) el$.metricSelect.value = S.metric;
}

el$.metricSelect.addEventListener("change", e => {
  S.metric = e.target.value;
  render();
});

// ------------------------------------------------------------- label bar
function buildLabelBar() {
  const labels = Object.entries(S.run.labels)
    .sort((a, b) => b[1] - a[1]);
  S.labelColor = {};
  labels.forEach(([name], i) => { S.labelColor[name] = CAT[i % CAT.length]; });

  if (!labels.length) { el$.labelBar.innerHTML = ""; return; }
  el$.labelBar.innerHTML =
    `<span class="field"><span>Detected</span></span>` +
    labels.map(([name, n]) =>
      `<button class="chip" type="button" data-label="${name}"
               aria-pressed="false">
         <span class="dot" style="background:${S.labelColor[name]}"></span>
         ${name.replace(/_/g, " ")}
         <span class="count">${n}</span>
       </button>`).join("");

  el$.labelBar.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const name = chip.dataset.label;
      if (S.activeLabels.has(name)) S.activeLabels.delete(name);
      else S.activeLabels.add(name);
      chip.setAttribute("aria-pressed", S.activeLabels.has(name));
      render();
    });
  });
}

// ----------------------------------------------------------------- render
function render() { S.tableView ? renderTable() : renderGrid(); renderLegend(); }

function matchesLabels(c) {
  if (!S.activeLabels.size) return true;
  return c.labels.some(l => S.activeLabels.has(l));
}

function renderGrid() {
  el$.gridScroll.classList.remove("hidden");
  el$.tableHost.classList.add("hidden");

  const { n_layers, n_heads, components } = S.run;
  const scale = S.metric ? scaleFor(S.metric) : null;
  const filtering = S.activeLabels.size > 0 || S.focus.size > 0;
  const sz = cellSize(n_layers, n_heads);
  const showEveryCol = sz >= 18 ? 1 : 2;   // thin out tick labels when tiny

  let html = `<div class="axis-caption">Attention heads &rarr;` +
             `&nbsp;&nbsp;&nbsp;(rows: layers, top = last)</div>`;
  html += `<table class="grid" style="--cell:${sz}px">` +
          `<thead><tr><th class="row-head"></th>`;
  for (let h = 0; h < n_heads; h++) {
    html += `<th class="col-head">${
      h % showEveryCol === 0 ? h : ""}</th>`;
  }
  html += `<th class="gap"></th><th class="col-head">MLP</th></tr></thead><tbody>`;

  for (let L = n_layers - 1; L >= 0; L--) {
    html += `<tr><th class="row-head">${
      n_layers <= 40 || L % 2 === 0 ? "L" + L : ""}</th>`;
    for (let h = 0; h < n_heads; h++) {
      html += cellHtml(`L${L}.H${h}`, components, scale, filtering, "");
    }
    html += `<td class="gap"></td>`;
    html += cellHtml(`L${L}.MLP`, components, scale, filtering, " mlp-cell");
    html += `</tr>`;
  }
  html += `</tbody></table>`;
  el$.gridHost.innerHTML = html;

  el$.gridHost.querySelectorAll(".cell[data-comp]").forEach(cell => {
    cell.addEventListener("click", () => selectComponent(cell.dataset.comp));
    cell.addEventListener("mouseenter", e => showTooltip(e, cell.dataset.comp));
    cell.addEventListener("mousemove", moveTooltip);
    cell.addEventListener("mouseleave", hideTooltip);
  });
}

function cellHtml(name, components, scale, filtering, extraCls) {
  const c = components[name];
  if (!c) return `<td class="cell empty-cell${extraCls}"></td>`;

  let bg = NO_DATA;
  if (scale && S.metric in c.stats) bg = scale.color(c.stats[S.metric]);

  const cls = ["cell", ...(extraCls ? [extraCls.trim()] : [])];
  if (S.activeLabels.size && matchesLabels(c)) cls.push("flagged");
  if (S.focus.has(name)) cls.push("flagged");
  if (filtering && !(matchesLabels(c) && (!S.focus.size || S.focus.has(name))))
    cls.push("dimmed");
  if (S.selected === name) cls.push("selected");

  return `<td class="${cls.join(" ")}" data-comp="${name}"
              style="background:${bg}"></td>`;
}

function renderLegend() {
  if (!S.metric) { el$.legend.innerHTML = ""; return; }
  const sc = scaleFor(S.metric);
  if (!sc) { el$.legend.innerHTML = ""; return; }
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    stops.push(sc.signed ? divColor(t * 2 - 1) : seqColor(t));
  }
  const span = sc.hi - sc.lo;
  el$.legend.innerHTML =
    `<span>${fmtScale(sc.lo, span)}</span>` +
    `<span class="scale-bar" style="background:linear-gradient(to right,${
      stops.join(",")})"></span>` +
    `<span>${fmtScale(sc.hi, span)}</span>` +
    (sc.signed ? `<span>&nbsp;0 at centre</span>` : "");
}

// Accessible equivalent of the grid.
function renderTable() {
  el$.gridScroll.classList.add("hidden");
  el$.tableHost.classList.remove("hidden");

  const metrics = PREFERRED_METRICS.filter(k => k in S.run.metrics);
  const rows = Object.entries(S.run.components)
    .filter(([, c]) => matchesLabels(c))
    .filter(([n]) => !S.focus.size || S.focus.has(n))
    .sort((a, b) => (S.metric && (b[1].stats[S.metric] ?? -Infinity) -
                                 (a[1].stats[S.metric] ?? -Infinity)) ||
                    a[1].layer - b[1].layer);

  el$.tableHost.innerHTML =
    `<table class="data"><thead><tr><th>Component</th><th>Kind</th>` +
    metrics.map(m => `<th>${m}</th>`).join("") +
    `<th>Labels</th><th>Evidence</th></tr></thead><tbody>` +
    rows.map(([n, c]) =>
      `<tr><td class="name">${n}</td><td>${c.kind}</td>` +
      metrics.map(m => `<td class="num">${
        m in c.stats ? fmt(c.stats[m]) : "—"}</td>`).join("") +
      `<td>${c.labels.map(l => l.replace(/_/g," ")).join(", ") || "—"}</td>` +
      `<td class="num">${c.n_evidence}</td></tr>`).join("") +
    `</tbody></table>`;
}

el$.viewToggle.addEventListener("click", () => {
  S.tableView = !S.tableView;
  el$.viewToggle.textContent = S.tableView ? "Grid view" : "Table view";
  render();
});

el$.clearFocus.addEventListener("click", () => {
  S.focus.clear();
  el$.clearFocus.classList.add("hidden");
  render();
});

// ---------------------------------------------------------------- tooltip
function showTooltip(e, name) {
  const c = S.run.components[name];
  if (!c) return;
  const shown = PREFERRED_METRICS.filter(m => m in c.stats).slice(0, 5);
  const extra = (S.metric && !shown.includes(S.metric) && S.metric in c.stats)
    ? [S.metric] : [];
  el$.tooltip.innerHTML =
    `<div class="t-name">${name}</div>` +
    [...extra, ...shown].map(m =>
      `<div class="t-row"><span>${m}</span>` +
      `<span class="v">${fmt(c.stats[m])}</span></div>`).join("") +
    (c.labels.length
      ? `<div class="t-labels">${c.labels.map(l=>l.replace(/_/g," ")).join(", ")}</div>`
      : "");
  el$.tooltip.classList.remove("hidden");
  moveTooltip(e);
}
function moveTooltip(e) {
  const t = el$.tooltip, pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = t.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  t.style.left = x + "px";
  t.style.top = y + "px";
}
function hideTooltip() { el$.tooltip.classList.add("hidden"); }

// ----------------------------------------------------------------- detail
function showDetailEmpty() {
  el$.detailEmpty.classList.remove("hidden");
  el$.detailContent.innerHTML = "";
}

async function selectComponent(name) {
  S.selected = name;
  switchTab("detail");
  render();

  const c = S.run.components[name];
  const exp = S.run.explanations[name];
  el$.detailEmpty.classList.add("hidden");

  let html =
    `<div class="detail-title">${name}</div>` +
    `<div class="detail-sub">${c.kind} · layer ${c.layer}` +
    (c.index !== null ? ` · index ${c.index}` : "") +
    (c.cluster !== null && c.cluster !== undefined
      ? ` · cluster ${c.cluster}` : "") + `</div>`;

  if (c.labels.length) {
    html += `<div class="sect"><h4>Detected patterns</h4>` +
      c.labels.map(l =>
        `<span class="tag"><span class="dot" style="background:${
          S.labelColor[l] || CAT[0]}"></span>${l.replace(/_/g," ")}</span>`
      ).join("") + `</div>`;
  }

  if (exp) {
    html += `<div class="sect"><h4>Proposed function</h4>` +
      `<div class="prose"><strong>${esc(exp.candidate_function)}</strong>` +
      (exp.confidence != null
        ? ` <span style="color:var(--text-muted)">(confidence ${
            exp.confidence.toFixed(2)})</span>` : "") + `</div>`;
    (exp.predictions || []).forEach(p => {
      if (p.passed === null || p.passed === undefined) return;
      html += `<div class="callout ${p.passed ? "pass" : "fail"}"
                 style="margin-top:8px">` +
        `${p.passed ? "PASSED" : "FAILED"} — ${esc(p.method || "")} ` +
        `expected ${esc(p.expected_direction || "")}, ` +
        `measured ${fmt(p.measured)}</div>`;
    });
    if (exp.next_experiment) {
      html += `<div class="prose" style="margin-top:9px">` +
        `<span style="color:var(--text-muted)">Next:</span> ` +
        `${esc(exp.next_experiment)}</div>`;
    }
    html += `</div>`;
  }

  (c.disagreements || []).forEach(d => {
    html += `<div class="sect"><h4>Method disagreement</h4>` +
            `<div class="callout">${esc(d)}</div></div>`;
  });

  const groups = [
    ["Attention", ["prev_token","induction","sink_pos0","same_token",
                   "self_attn","entropy"]],
    ["Activation", ["mean","std","kurtosis","abs_max","sparsity",
                    "dla_logit_diff"]],
  ];
  for (const [title, keys] of groups) {
    const present = keys.filter(k => k in c.stats);
    if (!present.length) continue;
    html += `<div class="sect"><h4>${title}</h4><div class="kv">` +
      present.map(k => `<div>${k.replace(/_/g," ")}</div>` +
                       `<div>${fmt(c.stats[k])}</div>`).join("") +
      `</div></div>`;
  }

  html += `<div class="sect"><h4>Evidence</h4>` +
    `<div class="prose" id="ev-slot">Loading ${c.n_evidence} records…</div>` +
    `</div>`;
  el$.detailContent.innerHTML = html;
  el$.detailContent.scrollTop = 0;

  const detail = await fetch(
    `/api/runs/${S.run.id}/component/${encodeURIComponent(name)}`
  ).then(x => x.json());
  const slot = document.getElementById("ev-slot");
  if (!slot) return;
  const recs = detail.evidence || [];
  if (!recs.length) { slot.textContent = "No evidence records."; return; }

  const byMethod = {};
  recs.forEach(r => {
    const m = r.spec.method;
    (byMethod[m] = byMethod[m] || []).push(r);
  });
  slot.innerHTML = `<table class="data"><thead><tr><th>Method</th>` +
    `<th>Spec</th><th>Key value</th></tr></thead><tbody>` +
    Object.entries(byMethod).map(([m, rs]) => {
      const r = rs[rs.length - 1];
      const spec = [r.spec.baseline, r.spec.direction, r.spec.metric]
        .filter(Boolean).join(" / ") || "—";
      const k = Object.keys(r.values)[0];
      return `<tr><td>${esc(m)}${rs.length>1?` <span style="color:var(--text-muted)">x${rs.length}</span>`:""}</td>` +
             `<td style="color:var(--text-secondary)">${esc(spec)}</td>` +
             `<td class="num">${k ? `${esc(k)} ${fmt(r.values[k])}` : "—"}</td></tr>`;
    }).join("") + `</tbody></table>`;
}

// -------------------------------------------------------------- chat / ws
function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  S.ws = new WebSocket(`${proto}//${location.host}/ws`);
  S.ws.onmessage = e => onWsMessage(JSON.parse(e.data));
  S.ws.onclose = () => setTimeout(connectWs, 2000);
}

let statusEl = null;

function onWsMessage(m) {
  if (m.type === "chat") {
    clearStatus();
    addMsg(m.message, m.sender, m.mentions);
  } else if (m.type === "status") {
    showStatus(m.message);
    if (/^Loading |^Rebuilding /.test(m.message)) renderModelStatus("loading");
    if (/^Model ready/.test(m.message)) {
      S.modelLoaded = true;
      renderModelStatus("loaded");
    }
  } else if (m.type === "run_updated") {
    refreshRun();
  } else if (m.type === "focus") {
    S.focus = new Set(m.components || []);
    if (m.metric && m.metric in S.run.metrics) {
      S.metric = m.metric;
      el$.metricSelect.value = m.metric;
    }
    el$.clearFocus.classList.toggle("hidden", S.focus.size === 0);
    render();
  }
}

function addMsg(text, sender, mentions) {
  const d = document.createElement("div");
  d.className = `msg ${sender}`;
  d.innerHTML = renderText(text, mentions);
  d.querySelectorAll(".comp-ref").forEach(ref => {
    ref.addEventListener("click", () => selectComponent(ref.dataset.comp));
  });
  el$.messages.appendChild(d);
  el$.messages.scrollTop = el$.messages.scrollHeight;
}

function showStatus(text) {
  if (!statusEl) {
    statusEl = document.createElement("div");
    statusEl.className = "msg status";
    el$.messages.appendChild(statusEl);
  }
  statusEl.textContent = text + "…";
  el$.messages.scrollTop = el$.messages.scrollHeight;
}
function clearStatus() {
  if (statusEl) { statusEl.remove(); statusEl = null; }
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function renderText(text, mentions) {
  let h = esc(text);
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, l, c) => `<pre>${c.trim()}</pre>`);
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  if (mentions && mentions.length) {
    const re = new RegExp(`\\b(${mentions.map(escRe).join("|")})\\b`, "g");
    h = h.replace(re, m => `<span class="comp-ref" data-comp="${m}">${m}</span>`);
  }
  // Group runs of "- item" lines into a real list; a pipe table would not
  // fit this panel, so fold one into rows of plain text instead.
  const out = [];
  let list = null;
  for (let line of h.split("\n")) {
    const t = line.trim();
    if (/^\|.*\|$/.test(t)) {
      if (/^\|[\s|:-]+\|$/.test(t)) continue;         // separator row
      line = t.slice(1, -1).split("|").map(c => c.trim())
              .filter(Boolean).join(" · ");
    }
    const m = line.match(/^\s*[-*]\s+(.*)$/);
    if (m) {
      (list = list || []).push(`<li>${m[1]}</li>`);
      continue;
    }
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
    out.push(line);
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("\n")
            .replace(/\n(?=<ul>)/g, "")
            .replace(/(<\/ul>)\n/g, "$1")
            .replace(/\n/g, "<br>");
}
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

el$.chatForm.addEventListener("submit", e => {
  e.preventDefault();
  const t = el$.chatInput.value.trim();
  if (!t) return;
  addMsg(t, "user");
  S.ws.send(JSON.stringify({ type: "chat", message: t }));
  el$.chatInput.value = "";
});

// ----------------------------------------------------------------- tabs
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.tab === name));
  $("tab-detail").classList.toggle("hidden", name !== "detail");
  $("tab-chat").classList.toggle("hidden", name !== "chat");
}
document.querySelectorAll(".tab").forEach(t =>
  t.addEventListener("click", () => switchTab(t.dataset.tab)));

let resizeTimer = null;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (S.run && !S.tableView) renderGrid(); }, 120);
});

init();
})();
