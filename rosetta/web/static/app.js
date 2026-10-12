/* Rosetta web frontend — chat + interactive model visualization. */

(function () {
  "use strict";

  // ------------------------------------------------------------------ state
  const state = {
    ws: null,
    model: null,         // model info from server
    highlights: {},      // component -> {metric: value, ...}
    selectedComp: null,  // currently selected component name
    activeMetric: "none",
    metrics: new Set(),  // discovered metric names
  };

  // ------------------------------------------------------------------ DOM refs
  const $messages   = document.getElementById("messages");
  const $chatForm   = document.getElementById("chat-form");
  const $chatInput  = document.getElementById("chat-input");
  const $modelSvg   = document.getElementById("model-svg");
  const $metricSel  = document.getElementById("metric-select");
  const $modelBadge = document.getElementById("model-badge");
  const $statusDot  = document.getElementById("status-indicator");
  const $detailPanel  = document.getElementById("detail-panel");
  const $detailTitle  = document.getElementById("detail-title");
  const $detailContent = document.getElementById("detail-content");
  const $detailClose   = document.getElementById("detail-close");

  // --------------------------------------------------------- WebSocket
  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    state.ws = new WebSocket(`${proto}//${location.host}/ws`);
    state.ws.onopen = () => { $statusDot.className = "status-dot connected"; };
    state.ws.onclose = () => {
      $statusDot.className = "status-dot";
      setTimeout(connect, 2000);
    };
    state.ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  }

  function send(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
    }
  }

  // --------------------------------------------------------- message handling
  function handleMessage(msg) {
    switch (msg.type) {
      case "chat":
        addChatMessage(msg.message, msg.sender);
        break;
      case "status":
        showStatus(msg.message);
        $statusDot.className = "status-dot busy";
        break;
      case "model_info":
        onModelInfo(msg.data);
        break;
      case "results":
        onResults(msg);
        $statusDot.className = "status-dot connected";
        break;
      case "error":
        addChatMessage(msg.message, "system");
        break;
    }
  }

  // --------------------------------------------------------- chat UI
  let lastStatusEl = null;

  function addChatMessage(text, sender) {
    lastStatusEl = null;
    const el = document.createElement("div");
    el.className = `msg ${sender}`;
    el.innerHTML = renderMarkdown(text);
    $messages.appendChild(el);
    $messages.scrollTop = $messages.scrollHeight;
  }

  function showStatus(text) {
    if (lastStatusEl) {
      lastStatusEl.textContent = text;
      return;
    }
    const el = document.createElement("div");
    el.className = "msg status";
    el.textContent = text;
    $messages.appendChild(el);
    $messages.scrollTop = $messages.scrollHeight;
    lastStatusEl = el;
  }

  function renderMarkdown(text) {
    // Minimal markdown: bold, code blocks, inline code, line breaks
    let html = escapeHtml(text);
    // code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g,
      (_, lang, code) => `<pre>${code.trim()}</pre>`);
    // inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // bold
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // italic
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    // line breaks
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  $chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $chatInput.value.trim();
    if (!text) return;
    addChatMessage(text, "user");
    send({ type: "chat", message: text });
    $chatInput.value = "";
  });

  // --------------------------------------------------------- model viz
  function onModelInfo(data) {
    state.model = data;
    if (data.loaded) {
      $modelBadge.textContent = data.name;
      $modelBadge.classList.add("active");
      buildModelSvg(data);
    } else {
      $modelBadge.textContent = "No model";
      $modelBadge.classList.remove("active");
    }
  }

  function buildModelSvg(m) {
    const { n_layers, n_heads } = m;
    $modelSvg.innerHTML = "";

    // layout constants
    const pad      = 24;
    const headW    = 36;
    const headH    = 32;
    const headGap  = 5;
    const mlpH     = 22;
    const layerGap = 16;
    const labelW   = 32;
    const rowW     = n_heads * (headW + headGap) - headGap;
    const mlpW     = rowW;
    const blockH   = headH + 8 + mlpH;
    const sectionH = 36;
    const totalW   = labelW + rowW + pad * 2;
    const totalH   = pad + sectionH + n_layers * (blockH + layerGap)
                     + sectionH + pad;

    $modelSvg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);

    let y = pad;

    // LM Head
    addRect($modelSvg, pad, y, totalW - pad * 2, sectionH - 8,
            "mlp", "lm_head");
    addText($modelSvg, totalW / 2, y + (sectionH - 8) / 2 + 4,
            "LM Head", "section-label");
    y += sectionH;

    // Layers (top = last layer)
    for (let layer = n_layers - 1; layer >= 0; layer--) {
      // layer label
      addText($modelSvg, 6, y + blockH / 2 + 3,
              `L${layer}`, "layer-label");

      // attention heads
      let x = labelW + pad - 8;
      for (let head = 0; head < n_heads; head++) {
        const id = `L${layer}.H${head}`;
        const rect = addRect($modelSvg, x, y, headW, headH, "head", id);
        addText($modelSvg, x + headW / 2, y + headH / 2 + 3,
                `H${head}`, "component-label");
        rect.addEventListener("click", () => selectComponent(id));
        x += headW + headGap;
      }

      // MLP
      const mlpId = `L${layer}.MLP`;
      const mlpX = labelW + pad - 8;
      const mlpRect = addRect($modelSvg, mlpX, y + headH + 8, mlpW, mlpH,
                              "mlp", mlpId);
      addText($modelSvg, mlpX + mlpW / 2, y + headH + 8 + mlpH / 2 + 3,
              "MLP", "component-label");
      mlpRect.addEventListener("click", () => selectComponent(mlpId));

      y += blockH + layerGap;
    }

    // Embedding
    addRect($modelSvg, pad, y, totalW - pad * 2, sectionH - 8,
            "mlp", "embed");
    addText($modelSvg, totalW / 2, y + (sectionH - 8) / 2 + 4,
            "Embedding", "section-label");
  }

  function addRect(parent, x, y, w, h, kind, id) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("rx", 4);
    rect.setAttribute("ry", 4);
    rect.setAttribute("data-comp", id);
    rect.classList.add("comp-rect", kind);
    parent.appendChild(rect);
    return rect;
  }

  function addText(parent, x, y, text, cls) {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x);
    t.setAttribute("y", y);
    t.textContent = text;
    if (cls) t.classList.add(cls);
    parent.appendChild(t);
    return t;
  }

  // --------------------------------------------------------- results handling
  function onResults(msg) {
    const hl = msg.highlights || {};
    // Merge into state
    for (const [comp, vals] of Object.entries(hl)) {
      if (!state.highlights[comp]) state.highlights[comp] = {};
      for (const [k, v] of Object.entries(vals)) {
        if (k === "labels") continue; // handled separately
        if (typeof v === "number") {
          state.highlights[comp][k] = v;
          state.metrics.add(k);
        }
      }
      // store labels
      if (vals.labels) {
        state.highlights[comp]._labels = vals.labels;
      }
    }
    updateMetricSelect();
    applyOverlay();
  }

  function updateMetricSelect() {
    const current = $metricSel.value;
    const existing = new Set();
    for (const opt of $metricSel.options) existing.add(opt.value);
    for (const m of state.metrics) {
      if (!existing.has(m)) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m.replace(/_/g, " ");
        $metricSel.appendChild(opt);
      }
    }
    // auto-select first real metric if still on "none"
    if (current === "none" && state.metrics.size > 0) {
      const first = state.metrics.values().next().value;
      $metricSel.value = first;
      state.activeMetric = first;
    }
  }

  $metricSel.addEventListener("change", () => {
    state.activeMetric = $metricSel.value;
    applyOverlay();
  });

  function applyOverlay() {
    const metric = state.activeMetric;
    const rects = $modelSvg.querySelectorAll(".comp-rect");

    if (metric === "none") {
      rects.forEach(r => {
        r.style.fill = "";
      });
      return;
    }

    // gather values for normalization
    let maxVal = 0;
    const vals = {};
    for (const [comp, data] of Object.entries(state.highlights)) {
      if (metric in data) {
        const v = Math.abs(data[metric]);
        vals[comp] = v;
        if (v > maxVal) maxVal = v;
      }
    }
    if (maxVal === 0) maxVal = 1;

    rects.forEach(r => {
      const comp = r.getAttribute("data-comp");
      if (comp in vals) {
        const t = vals[comp] / maxVal; // 0..1
        r.style.fill = heatColor(t);
      } else {
        r.style.fill = "";
      }
    });
  }

  function heatColor(t) {
    // 0 = cool dark blue, 0.5 = teal, 1 = warm orange
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.5) {
      const s = t * 2;
      r = Math.round(30 + s * 20);
      g = Math.round(40 + s * 140);
      b = Math.round(100 + s * 60);
    } else {
      const s = (t - 0.5) * 2;
      r = Math.round(50 + s * 210);
      g = Math.round(180 - s * 80);
      b = Math.round(160 - s * 120);
    }
    return `rgb(${r},${g},${b})`;
  }

  // --------------------------------------------------------- component select
  function selectComponent(name) {
    // toggle selection
    if (state.selectedComp === name) {
      deselectComponent();
      return;
    }
    state.selectedComp = name;

    // highlight SVG
    $modelSvg.querySelectorAll(".comp-rect").forEach(r => {
      r.classList.toggle("selected", r.getAttribute("data-comp") === name);
    });

    // show detail panel
    $detailTitle.textContent = name;
    $detailPanel.classList.remove("hidden");

    const data = state.highlights[name];
    if (!data) {
      $detailContent.innerHTML =
        '<p style="color:var(--text-dim)">No analysis data yet. ' +
        'Run profiling or ablation first.</p>';
      return;
    }

    let html = '';

    // labels
    if (data._labels && data._labels.length) {
      html += '<div class="detail-section"><h4>Detected patterns</h4>';
      html += data._labels.map(l => `<span class="label-tag">${escapeHtml(l)}</span>`).join("");
      html += '</div>';
    }

    // numeric stats
    const numericKeys = Object.keys(data).filter(k => !k.startsWith("_") && typeof data[k] === "number");
    if (numericKeys.length) {
      html += '<div class="detail-section"><h4>Metrics</h4><div class="stat-grid">';
      for (const k of numericKeys) {
        const v = data[k];
        const display = Number.isInteger(v) ? v : v.toFixed(4);
        html += `<div class="stat-item"><span class="key">${escapeHtml(k.replace(/_/g, " "))}</span>` +
                `<span class="val">${display}</span></div>`;
      }
      html += '</div></div>';
    }

    $detailContent.innerHTML = html;
  }

  function deselectComponent() {
    state.selectedComp = null;
    $modelSvg.querySelectorAll(".comp-rect.selected").forEach(r =>
      r.classList.remove("selected"));
    $detailPanel.classList.add("hidden");
  }

  $detailClose.addEventListener("click", deselectComponent);

  // --------------------------------------------------------- init
  connect();
})();
