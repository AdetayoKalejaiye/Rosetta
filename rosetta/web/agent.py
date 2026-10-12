"""Rosetta agent — LLM-driven interpretability orchestrator.

Most questions are answered by querying the finished run's evidence, which
needs no model. Only new interventions load torch, and only on request.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Callable, Coroutine, Dict, List

from .runs import component_detail

SYSTEM_PROMPT = """\
You are Rosetta, an assistant for mechanistic interpretability of transformers.
A finished analysis run is loaded. Your job is to help the user find and \
understand specific components (attention heads, MLP blocks) inside it.

## How to work
1. Start from the evidence already in the run — `query_components` is your \
workhorse. It filters by detector label and ranks by any recorded metric.
2. Quote concrete component names (e.g. "L24.H11") and numbers. The UI \
highlights every component you name, so name them explicitly.
3. Explain what a number means, don't just report it.
4. Suggest the next experiment.

## Metrics you can rank by
- `induction` — attends to the previous occurrence of the current token and \
predicts what followed. The signature of an induction head.
- `prev_token` — attends to the immediately preceding token.
- `sink_pos0` — attention dumped on token 0. Usually an artifact, not a function.
- `same_token`, `self_attn`, `entropy` (low entropy = focused attention).
- `dla_logit_diff` — direct logit attribution; signed. Positive writes toward \
the answer. This is a readout contribution, NOT a removal effect.
- Activation stats: `mean`, `std`, `kurtosis`, `abs_max`, `sparsity`.
- Weight stats are prefixed `w.` (e.g. `w.q_norm`, `w.ov_eff_rank`).

## Detector labels
`induction_head`, `prev_token_head`, `attention_sink`, `same_token_head`, \
`self_attending`, `focused_attention`, `direct_logit_writer_positive`.

## Honesty rules
- Small models do not have rich semantic features. If the user asks for a \
"color head" or similar in a model trained only on sequence repetition, say \
plainly that the run contains no evidence that could support such a claim, \
and offer what the evidence *can* answer instead. Do not invent a story.
- A detector label is a heuristic threshold, not proof. Say so when it matters.
- Ablation baselines disagreeing is a real finding — surface it, don't hide it.
- Interventions need the original checkpoint. Only the toy run can be rebuilt \
locally; for other runs explain that and stick to the recorded evidence.

## Formatting
Your reply renders in a narrow side panel. Never use markdown tables — they \
do not fit. For ranked results use one short line per item, e.g.
`- L24.H10 — induction 0.437 (attention sink, focused)`
Keep it to a few lines plus a short takeaway. No preamble.
"""

TOOLS = [
    {
        "name": "get_run_summary",
        "description": "Architecture, component counts, detector-label distribution, and every metric available in the loaded run with its value range. Call this first if unsure what the run contains.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "query_components",
        "description": "Filter and rank components from the loaded run. The main way to answer 'which head does X'. Filter by detector label and/or component kind, rank by any metric.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string",
                          "description": "Only components with this detector label"},
                "kind": {"type": "string", "enum": ["head", "mlp", "kv_head", "resid"]},
                "sort_by": {"type": "string",
                            "description": "Metric to rank by, e.g. 'induction'"},
                "descending": {"type": "boolean", "description": "Default true"},
                "min_value": {"type": "number",
                              "description": "Drop components below this sort_by value"},
                "layer": {"type": "integer", "description": "Restrict to one layer"},
                "limit": {"type": "integer", "description": "Max rows (default 20)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_component",
        "description": "Full detail for one component: all stats, detector labels, recorded evidence records with their exact method specs, disagreements, and any explanation entry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {"type": "string", "description": "e.g. 'L24.H11'"},
                "include_evidence": {"type": "boolean",
                                     "description": "Include raw records (default true)"},
            },
            "required": ["component"],
        },
    },
    {
        "name": "compare_components",
        "description": "Side-by-side metric comparison of several components.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"},
                            "description": "Omit for the standard attention set"},
            },
            "required": ["components"],
        },
    },
    {
        "name": "run_ablation",
        "description": "NEW intervention: ablate components and measure the effect. Requires rebuilding the model — only works on the toy run. Use recorded evidence instead where possible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {"type": "array", "items": {"type": "string"}},
                "baseline": {"type": "string", "enum": ["zero", "mean", "resample"]},
            },
            "required": ["components"],
        },
    },
    {
        "name": "run_activation_patching",
        "description": "NEW intervention: patch activations between clean and corrupt runs. Requires rebuilding the model — toy run only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {"type": "array", "items": {"type": "string"}},
                "direction": {"type": "string", "enum": ["denoising", "noising"]},
            },
            "required": ["components"],
        },
    },
]

ATTENTION_METRICS = ["induction", "prev_token", "sink_pos0", "same_token",
                     "self_attn", "entropy"]

SendFn = Callable[[Dict], Coroutine]


def _call_api(messages: List[Dict], system: str) -> Dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "system": system,
        "tools": TOOLS,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={"Content-Type": "application/json",
                 "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


class RosettaAgent:
    def __init__(self, session):
        self.session = session
        self.history: List[Dict] = []
        self._has_api = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

    def reset(self):
        self.history = []

    async def handle_message(self, user_msg: str, send: SendFn):
        if not self._has_api:
            await send({"type": "chat", "sender": "system",
                        "message": "No `ANTHROPIC_API_KEY` found, so chat is "
                                   "disabled. The run data and visualization "
                                   "work without it."})
            return
        if not self.session.run_data:
            await send({"type": "chat", "sender": "system",
                        "message": "Select a run first."})
            return

        self.history.append({"role": "user", "content": user_msg})
        loop = asyncio.get_event_loop()

        while True:
            await send({"type": "status", "message": "Thinking"})
            try:
                response = await loop.run_in_executor(
                    None, lambda: _call_api(self.history, SYSTEM_PROMPT))
            except Exception as e:
                await send({"type": "chat", "sender": "system",
                            "message": f"API error: {e}"})
                return

            content = response.get("content", [])
            self.history.append({"role": "assistant", "content": content})

            for block in content:
                if block["type"] == "text" and block["text"].strip():
                    await send({"type": "chat", "sender": "agent",
                                "message": block["text"],
                                "mentions": self._mentions(block["text"])})

            if response.get("stop_reason") != "tool_use":
                break

            results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                await send({"type": "status",
                            "message": _label(block["name"])})
                try:
                    out = await loop.run_in_executor(
                        None, lambda b=block: self._exec(b["name"], b["input"]))
                except Exception as e:
                    out = {"error": str(e)}
                if out.get("_focus"):
                    await send({"type": "focus",
                                "components": out.pop("_focus"),
                                "metric": out.pop("_metric", None)})
                results.append({"type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": json.dumps(out, default=str)[:60000]})
            self.history.append({"role": "user", "content": results})

    def _mentions(self, text: str) -> List[str]:
        import re
        comps = self.session.run_data["components"]
        return sorted({m for m in re.findall(r"\bL\d+\.(?:H\d+|KV\d+|MLP|RESID)\b",
                                             text) if m in comps})

    # ------------------------------------------------------------------ tools

    def _exec(self, name: str, inp: Dict) -> Dict:
        fn = {
            "get_run_summary": lambda: self._summary(),
            "query_components": lambda: self._query(inp),
            "get_component": lambda: self._component(
                inp["component"], inp.get("include_evidence", True)),
            "compare_components": lambda: self._compare(
                inp["components"], inp.get("metrics")),
            "run_ablation": lambda: self._ablation(
                inp["components"], inp.get("baseline", "mean")),
            "run_activation_patching": lambda: self._patch(
                inp["components"], inp.get("direction", "denoising")),
        }.get(name)
        return fn() if fn else {"error": f"Unknown tool: {name}"}

    def _summary(self) -> Dict:
        r = self.session.run_data
        kinds: Dict[str, int] = {}
        for c in r["components"].values():
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        return {
            "run": r["id"],
            "n_layers": r["n_layers"],
            "n_heads_per_layer": r["n_heads"],
            "component_kinds": kinds,
            "detector_labels": r["labels"],
            "metrics": {k: {"min": round(v["min"], 4),
                            "max": round(v["max"], 4),
                            "n": v["n"], "signed": v["signed"]}
                        for k, v in r["metrics"].items()},
            "explained_components": list(r["explanations"].keys()),
        }

    def _query(self, inp: Dict) -> Dict:
        r = self.session.run_data
        sort_by = inp.get("sort_by")
        rows = []
        for name, c in r["components"].items():
            if inp.get("label") and inp["label"] not in c["labels"]:
                continue
            if inp.get("kind") and c["kind"] != inp["kind"]:
                continue
            if inp.get("layer") is not None and c["layer"] != inp["layer"]:
                continue
            val = c["stats"].get(sort_by) if sort_by else None
            if sort_by and val is None:
                continue
            if inp.get("min_value") is not None and (
                    val is None or val < inp["min_value"]):
                continue
            row = {"component": name, "layer": c["layer"],
                   "labels": c["labels"], "n_evidence": c["n_evidence"]}
            if sort_by:
                row[sort_by] = round(val, 5)
            else:
                row.update({m: round(c["stats"][m], 4)
                            for m in ATTENTION_METRICS if m in c["stats"]})
            rows.append(row)

        if sort_by:
            rows.sort(key=lambda x: x[sort_by],
                      reverse=inp.get("descending", True))
        else:
            rows.sort(key=lambda x: (x["layer"], x["component"]))
        limit = inp.get("limit", 20)
        out = {"n_matched": len(rows), "rows": rows[:limit]}
        if rows[:limit]:
            out["_focus"] = [x["component"] for x in rows[:limit]]
            out["_metric"] = sort_by
        return out

    def _component(self, comp: str, include_evidence: bool) -> Dict:
        r = self.session.run_data
        c = r["components"].get(comp)
        if not c:
            return {"error": f"{comp} not in run {r['id']}"}
        out = {"component": comp, "kind": c["kind"], "layer": c["layer"],
               "index": c["index"], "labels": c["labels"],
               "cluster": c["cluster"], "disagreements": c["disagreements"],
               "stats": {k: round(v, 5) for k, v in c["stats"].items()}}
        if comp in r["explanations"]:
            out["explanation"] = r["explanations"][comp]
        if include_evidence:
            detail = component_detail(r["path"], comp)
            out["evidence"] = [
                {"method": rec["spec"]["method"],
                 "baseline": rec["spec"].get("baseline"),
                 "direction": rec["spec"].get("direction"),
                 "metric": rec["spec"].get("metric"),
                 "family": rec.get("family"),
                 "values": rec["values"]}
                for rec in detail["evidence"]
            ]
        out["_focus"] = [comp]
        return out

    def _compare(self, comps: List[str], metrics) -> Dict:
        r = self.session.run_data
        metrics = metrics or ATTENTION_METRICS
        table = {}
        for name in comps:
            c = r["components"].get(name)
            if not c:
                table[name] = {"error": "not found"}
                continue
            table[name] = {m: round(c["stats"][m], 4)
                           for m in metrics if m in c["stats"]}
            table[name]["labels"] = c["labels"]
        return {"comparison": table,
                "_focus": [c for c in comps if c in r["components"]]}

    def _ablation(self, comps: List[str], baseline: str) -> Dict:
        from ..methods.ablation import ablate
        from ..schema import Component
        s = self.session
        s.ensure_model()
        out = {}
        for n in comps:
            rec = ablate(s.adapter, s.families["train"], Component.parse(n),
                         baseline, ref_fam=s.families["reference"])
            s.store.add(rec)
            out[n] = {k: round(float(v), 5) for k, v in rec.values.items()}
        return {"baseline": baseline, "results": out, "_focus": list(out)}

    def _patch(self, comps: List[str], direction: str) -> Dict:
        from ..methods.patching import activation_patch
        from ..schema import Component
        s = self.session
        s.ensure_model()
        out = {}
        for n in comps:
            rec = activation_patch(s.adapter, s.families["train"],
                                   Component.parse(n), direction)
            s.store.add(rec)
            out[n] = {k: round(float(v), 5) for k, v in rec.values.items()}
        return {"direction": direction, "results": out, "_focus": list(out)}


def _label(tool: str) -> str:
    return {"get_run_summary": "Reading run summary",
            "query_components": "Querying components",
            "get_component": "Fetching component detail",
            "compare_components": "Comparing components",
            "run_ablation": "Running ablation",
            "run_activation_patching": "Running activation patching",
            }.get(tool, tool)
