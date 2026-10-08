"""Logging & inspection.

TensorBoard handles the logging side (scalars, histograms, attention-score
heat images, text summaries, embeddings, and hparams) when the `tensorboard` 
package is installed; otherwise the same payload is written as JSON so nothing 
is lost.

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
from typing import Dict, Optional, List, Any, Union

from ..schema import Explanation
from ..store import EvidenceStore


class Exporter:
    """Rich TensorBoard exporter with graceful JSON degradation."""
    
    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        # Remove stale event files so TensorBoard doesn't see multiple runs
        # in the same directory and merge/confuse them.
        for fname in os.listdir(logdir):
            if fname.startswith("events.out.tfevents"):
                try:
                    os.remove(os.path.join(logdir, fname))
                except OSError:
                    pass
        self._json: Dict[str, Any] = {}
        try:
            from torch.utils.tensorboard import SummaryWriter  # needs tensorboard pkg
            self.writer = SummaryWriter(logdir)
        except Exception:
            self.writer = None

    def scalar(self, tag: str, value: float, step: int = 0):
        self._json.setdefault("scalars", {}).setdefault(tag, {})[str(step)] = value
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def histogram(self, tag: str, values: List[float], step: int = 0):
        self._json.setdefault("histograms", {}).setdefault(tag, {})[str(step)] = values
        if self.writer:
            try:
                import numpy as np
                self.writer.add_histogram(tag, np.array(values), step)
            except Exception:
                pass

    def matrix(self, tag: str, mat: List[List[float]], xlabel: str = "", ylabel: str = "", step: int = 0):
        """Logs a 2D matrix as a colored heatmap figure, falling back to a grayscale image."""
        data = [[float(v) for v in row] for row in mat]
        self._json.setdefault("matrices", {})[tag] = {"matrix": data, "xlabel": xlabel, "ylabel": ylabel}
        
        if not self.writer:
            return
            
        try:
            # Prefer rich matplotlib heatmaps if available
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest')
            fig.colorbar(cax)
            if xlabel: 
                ax.set_xlabel(xlabel)
            if ylabel: 
                ax.set_ylabel(ylabel)
            ax.set_title(tag)
            
            self.writer.add_figure(tag, fig, global_step=step)
            plt.close(fig)
        except Exception:
            # Fallback to basic grayscale image tensor
            try:
                import torch
                t = torch.tensor(data, dtype=torch.float32)
                t_min, t_max = t.min(), t.max()
                t = (t - t_min) / (t_max - t_min + 1e-9)
                self.writer.add_image(tag, t.unsqueeze(0), global_step=step)
            except Exception:
                pass

    def embedding(self, tag: str, mat: List[List[float]], metadata: Optional[List[str]] = None, step: int = 0):
        """Logs high-dimensional data for TensorBoard's Projector (PCA/UMAP)."""
        self._json.setdefault("embeddings", {})[tag] = {"matrix": mat, "metadata": metadata}
        if self.writer:
            try:
                import torch
                t = torch.tensor(mat, dtype=torch.float32)
                self.writer.add_embedding(t, metadata=metadata, tag=tag, global_step=step)
            except Exception:
                pass

    def hparams(self, hparam_dict: Dict[str, Union[int, float, str, bool]], metric_dict: Dict[str, float]):
        """Logs hyperparameter configurations and their resulting metrics."""
        self._json.setdefault("hparams", []).append({"hparams": hparam_dict, "metrics": metric_dict})
        if self.writer:
            try:
                self.writer.add_hparams(hparam_dict, metric_dict)
            except Exception:
                pass

    def text(self, tag: str, s: str, step: int = 0):
        """Logs Markdown-formatted text to TensorBoard."""
        self._json.setdefault("texts", {}).setdefault(tag, {})[str(step)] = s
        if self.writer:
            self.writer.add_text(tag, s, step)

    def flush(self):
        with open(os.path.join(self.logdir, "summaries.json"), "w") as f:
            json.dump(self._json, f, indent=2)
        if self.writer:
            self.writer.flush()

    def close(self):
        """Safely close the TensorBoard writer."""
        if self.writer:
            self.writer.close()


def export_store(store: EvidenceStore, exporter: Exporter, cfg=None, explanations: Optional[Dict[str, Explanation]] = None):
    """Pipes EvidenceStore contents directly into TensorBoard's rich visualizers."""
    explanations = explanations or {}
    
    # 1. Per-component scalars and text summaries
    for name, prof in store.profiles.items():
        # Scalars
        for k, v in prof.feature_vector_fields().items():
            exporter.scalar(f"profile/{name}/{k}", v)
            
        # Rich Markdown summaries for TensorBoard Text tab
        md_lines = [f"### {name}"]
        
        if name in explanations:
            e = explanations[name]
            md_lines.append(f"**Explanation:** {e.candidate_function} *(source: {e.source})*")
            if e.relevant_contexts:
                md_lines.append(f"**Contexts:** {', '.join(e.relevant_contexts)}")
        
        if prof.detector_labels:
            md_lines.append(f"**Tags:** {', '.join(prof.detector_labels)}")
            
        if prof.top_examples:
            md_lines.append("#### Top Activating Examples")
            for i, ex in enumerate(prof.top_examples[:5]):
                toks = ex.get("tokens", [])
                pos = ex.get("position", -1)
                act = ex.get("activation", 0.0)
                # Markdown highlighting format for the specific token
                highlighted = " ".join(f"**[{t}]**" if idx == pos else str(t) for idx, t in enumerate(toks))
                md_lines.append(f"* **Act {act:.3f}**: `{highlighted}`")
                
        exporter.text(f"components/{name}", "\n\n".join(md_lines))

    # 2. Evidence Record logging
    for rec in store.all_records():
        for k, v in rec.values.items():
            if isinstance(v, (int, float)):
                exporter.scalar(
                    f"evidence/{rec.spec.method}/{'+'.join(c.name for c in rec.components)}/{k}",
                    float(v)
                )
                
    # 3. Structural grids (Layer x Head heatmaps)
    if cfg is not None:
        for field, source in (("attn.induction", "profile"),
                              ("attn.prev_token", "profile"),
                              ("act.dla_logit_diff", "profile")):
            mat = [[store.profile(f"L{l}.H{h}").feature_vector_fields().get(field, 0.0)
                    for h in range(cfg.n_heads)] for l in range(cfg.n_layers)]
            # Uses matplotlib rendering internally for gorgeous heatmaps
            exporter.matrix(f"grid/{field}", mat, xlabel="Head", ylabel="Layer")
            


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
