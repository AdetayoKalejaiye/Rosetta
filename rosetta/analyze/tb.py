"""Logging & inspection.

TensorBoard handles the logging side (scalars, histograms, attention-score
heat images, text summaries) when the `tensorboard` package is installed;
otherwise the same payload is written as JSON so nothing is lost.

`write_inspection_html` builds the Rosetta inspection view described in the
brief: select a component -> its stats, top examples with activation
highlights, intervention results (with methodological spec), disagreements,
related components, and the proposed explanation with its prediction record -
so a heatmap number stays connected to the evidence behind it. It is a single
static HTML file with no dependencies.
"""
from __future__ import annotations

import html
import json
import os
from typing import Dict, Optional

from ..schema import Explanation
from ..store import EvidenceStore


class Exporter:
    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self._json: Dict[str, Dict[str, float]] = {}
        try:
            from torch.utils.tensorboard import SummaryWriter  # needs tensorboard pkg
            self.writer = SummaryWriter(logdir)
        except Exception:
            self.writer = None

    def scalar(self, tag: str, value: float, step: int = 0):
        self._json.setdefault(tag, {})[str(step)] = value
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def matrix(self, tag: str, mat, xlabel="", ylabel=""):
        data = [[float(v) for v in row] for row in mat]
        self._json[tag] = {"matrix": data, "xlabel": xlabel, "ylabel": ylabel}
        if self.writer:
            try:
                import torch
                t = torch.tensor(data)
                t = (t - t.min()) / (t.max() - t.min() + 1e-9)
                self.writer.add_image(tag, t.unsqueeze(0))
            except Exception:
                pass

    def text(self, tag: str, s: str):
        self._json[tag] = {"text": s}
        if self.writer:
            self.writer.add_text(tag, s)

    def flush(self):
        with open(os.path.join(self.logdir, "summaries.json"), "w") as f:
            json.dump(self._json, f, indent=2)
        if self.writer:
            self.writer.flush()


def export_store(store: EvidenceStore, exporter: Exporter, cfg=None):
    # per-component scalars
    for name, prof in store.profiles.items():
        for k, v in prof.feature_vector_fields().items():
            exporter.scalar(f"profile/{name}/{k}", v)
    for rec in store.all_records():
        for k, v in rec.values.items():
            if isinstance(v, (int, float)):
                exporter.scalar(
                    f"evidence/{rec.spec.method}/{'+'.join(c.name for c in rec.components)}/{k}",
                    float(v))
    # layer x head matrices for the headline numbers
    if cfg is not None:
        for field, source in (("attn.induction", "profile"),
                              ("attn.prev_token", "profile"),
                              ("act.dla_logit_diff", "profile")):
            mat = [[store.profile(f"L{l}.H{h}").feature_vector_fields().get(field, 0.0)
                    for h in range(cfg.n_heads)] for l in range(cfg.n_layers)]
            exporter.matrix(f"grid/{field}", mat, xlabel="head", ylabel="layer")
    exporter.flush()


# ------------------------------------------------------------- inspection view

_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Rosetta inspection</title><style>
 body{{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}}
 #list{{width:220px;overflow:auto;border-right:1px solid #ccc;padding:8px}}
 #list a{{display:block;padding:3px 6px;text-decoration:none;color:#222;border-radius:4px}}
 #list a:hover{{background:#eee}} #main{{flex:1;overflow:auto;padding:16px 24px}}
 .card{{border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}}
 .warn{{background:#fff3e0}} table{{border-collapse:collapse;font-size:13px}}
 td,th{{border:1px solid #ddd;padding:3px 8px;text-align:right}}
 th{{background:#f5f5f5}} .tag{{background:#e3f2fd;border-radius:10px;padding:1px 8px;
 margin-right:4px;font-size:12px}} h2{{margin-top:4px}}
 .hl{{background:#ffe082;border-radius:3px;padding:0 2px}}
 pre{{background:#f7f7f7;padding:8px;border-radius:6px;overflow:auto;font-size:12px}}
</style></head><body>
<div id="list"><h3>Components</h3>{links}</div>
<div id="main">{sections}</div>
<script>
 function show(id){{
   document.querySelectorAll('.comp').forEach(e=>e.style.display='none');
   document.getElementById(id).style.display='block';
 }}
 show('{first}');
</script></body></html>"""


def _stats_table(d: Dict[str, float]) -> str:
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{v:.4f}</td></tr>"
                   for k, v in sorted(d.items()) if isinstance(v, (int, float)))
    return f"<table>{rows}</table>" if rows else "<em>none</em>"


def write_inspection_html(store: EvidenceStore,
                          explanations: Dict[str, Explanation],
                          path: str,
                          related: Optional[Dict[str, list]] = None):
    related = related or {}
    names = sorted(store.profiles.keys(),
                   key=lambda n: (n.split(".")[0], n))
    links, sections = [], []
    for name in names:
        prof = store.profiles[name]
        sid = name.replace(".", "_")
        links.append(f'<a href="#" onclick="show(\'{sid}\');return false">{name}</a>')
        tags = "".join(f'<span class="tag">{html.escape(t)}</span>'
                       for t in prof.detector_labels)
        cluster = f'<span class="tag">cluster {prof.cluster}</span>' if prof.cluster is not None else ""
        warn = ""
        if prof.disagreements:
            items = "".join(f"<li>{html.escape(d)}</li>" for d in prof.disagreements)
            warn = f'<div class="card warn"><b>Method disagreements</b><ul>{items}</ul></div>'
        examples = ""
        for ex in prof.top_examples:
            toks = ex.get("tokens", [])
            pos = ex.get("position", -1)
            shown = " ".join(
                f'<span class="hl">{t}</span>' if i == pos else str(t)
                for i, t in enumerate(toks))
            examples += (f'<div class="card"><b>act {ex.get("activation")}</b> '
                         f'@ pos {pos} ({html.escape(str(ex.get("family")))})'
                         f'<br><code>{shown}</code></div>')
        ev_rows = ""
        for rec in store.for_component(name)[-30:]:
            vals = ", ".join(f"{k}={v:.3f}" for k, v in rec.values.items()
                             if isinstance(v, (int, float)))
            spec = {k: v for k, v in rec.spec.to_dict().items() if v}
            ev_rows += (f"<tr><td style='text-align:left'>{rec.spec.method}</td>"
                        f"<td style='text-align:left'>{html.escape(rec.family)}</td>"
                        f"<td style='text-align:left'>{html.escape(vals)}</td>"
                        f"<td style='text-align:left'><small>{html.escape(json.dumps(spec))}</small></td></tr>")
        expl_html = "<em>no explanation yet</em>"
        if name in explanations:
            e = explanations[name]
            preds = ""
            for p in e.predictions:
                status = ("&#10003; passed" if p.passed else
                          "&#10007; failed" if p.passed is False else "untested")
                preds += (f"<li>{p.method} on {'+'.join(p.components)} @ {p.family}: "
                          f"expect {p.metric} to {p.expected_direction} "
                          f"(&ge;{p.min_abs_effect:.2f}) - <b>{status}</b>"
                          + (f", measured {p.measured:+.3f}" if p.measured is not None else "")
                          + "</li>")
            contra = "".join(f"<li>{html.escape(c)}</li>" for c in e.contradictions) or "<li>none recorded</li>"
            expl_html = (f"<p><b>{html.escape(e.candidate_function)}</b> "
                         f"<small>({e.source})</small></p>"
                         f"<p>Contexts: {html.escape(', '.join(e.relevant_contexts))}</p>"
                         f"<p>Contradictions:</p><ul>{contra}</ul>"
                         f"<p>Predictions:</p><ul>{preds}</ul>"
                         f"<p>Next experiment: {html.escape(e.next_experiment)}</p>")
        rel = ", ".join(related.get(name, [])) or "none found"
        sections.append(f"""
<div class="comp" id="{sid}" style="display:none">
 <h2>{name}</h2><p>{tags}{cluster}</p>{warn}
 <div class="card"><b>Proposed explanation</b>{expl_html}</div>
 <div class="card"><b>Related components</b><p>{rel}</p></div>
 <div class="card"><b>Weight stats</b>{_stats_table(prof.weight_stats)}</div>
 <div class="card"><b>Attention stats</b>{_stats_table(prof.attention_stats)}</div>
 <div class="card"><b>Activation stats</b>{_stats_table(prof.activation_stats)}</div>
 <div class="card"><b>Top activating examples</b>{examples or '<em>none</em>'}</div>
 <div class="card"><b>Evidence (with method spec)</b>
   <table><tr><th>method</th><th>family</th><th>values</th><th>spec</th></tr>{ev_rows}</table>
 </div>
</div>""")
    first = names[0].replace(".", "_") if names else ""
    with open(path, "w") as f:
        f.write(_PAGE.format(links="".join(links), sections="".join(sections),
                             first=first))
    return path
