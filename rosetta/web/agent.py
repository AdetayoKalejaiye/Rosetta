"""Rosetta agent — LLM-driven interpretability orchestrator.

Most questions are answered by querying the finished run's evidence, which
needs no model. Only new interventions load torch, and only on request.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Callable, Coroutine, Dict, List, Optional

from .runs import component_detail

SYSTEM_PROMPT = """\
You are Rosetta, an assistant for mechanistic interpretability of transformers.
A finished analysis run is loaded. Your job is to help the user find and \
understand specific components (attention heads, MLP blocks) inside it.

## How to work
1. Start from the evidence already in the run — `query_components` is your \
workhorse. It filters by detector label and ranks by any recorded metric.
2. When you do NOT already know which components matter — especially on a \
family you just built — run `screen_components` first. It ranks all ~480 \
heads in one gradient pass, then you verify the top few with a real \
intervention. Never loop an intervention over every head: that is many \
minutes of work for something screening does in seconds, and it is capped at \
32 components per call for exactly that reason.
3. Quote concrete component names (e.g. "L24.H11") and numbers. The UI \
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

These come from fixed thresholds and some are near-useless on a real model: \
`attention_sink` fires on ~72% of heads in SmolLM2, so it tells you almost \
nothing. Prefer the population-relative view: every metric also has a \
percentile and a z-score (selectable as `z.<metric>`). A head at z=5 on \
induction is a finding; one merely "above threshold" on a ubiquitous metric \
is not. Say which you are relying on.

## Semantic questions
The run's built-in prompts are random tokens, so they can only support \
positional/structural claims (induction, previous-token, copying). They say \
NOTHING about meaning. To investigate a semantic question — colour, \
sentiment, country-capital, anything about content — you must first build a \
real-text family with `create_prompt_family`, then run interventions against \
it. Do that rather than refusing, and rather than over-reading the synthetic \
prompts. Check the reported `separation` before trusting results: if the \
model does not clearly prefer the clean answer, say so.

## Honesty rules
- Small models do not have rich semantic features. If the user asks for a \
"color head" or similar in a model trained only on sequence repetition, say \
plainly that the run contains no evidence that could support such a claim, \
and offer what the evidence *can* answer instead. Do not invent a story.
- A detector label is a heuristic threshold, not proof. Say so when it matters.
- Ablation baselines disagreeing is a real finding — surface it, don't hide it.
- Interventions load the run's model and take roughly 10s per component on a \
360M model. Run them when they answer the question; say what you are about to \
do first. They append new evidence to the run.

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
        "name": "create_prompt_family",
        "description": (
            "Write a set of real sentences to probe a specific behaviour, then "
            "run interventions against them. This is how you investigate "
            "semantic questions ('which head knows colours?') that the "
            "run's default random-token prompts cannot answer.\n"
            "Each item is a minimal pair: `clean` and `corrupt` differ by ONE "
            "word, `answer` is the token the clean sentence should predict and "
            "`distractor` the one the corrupt sentence favours.\n"
            "Two hard constraints, both checked: clean and corrupt must "
            "tokenize to the SAME number of tokens (swap words of similar "
            "length), and answer/distractor must each be exactly ONE token "
            "(write them with a leading space, e.g. ' red', and prefer short "
            "common words). Rejected items come back with reasons — fix and "
            "resubmit. Aim for 10-30 items; more is better evidence."),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Short id, e.g. 'colour'"},
                "description": {"type": "string",
                                "description": "What behaviour this probes"},
                "items": {
                    "type": "array",
                    "description": "Minimal pairs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "clean": {"type": "string"},
                            "corrupt": {"type": "string"},
                            "answer": {"type": "string"},
                            "distractor": {"type": "string"},
                        },
                        "required": ["clean", "corrupt", "answer",
                                     "distractor"],
                    },
                },
            },
            "required": ["name", "items"],
        },
    },
    {
        "name": "list_families",
        "description": "Prompt families available to run interventions against: the run's built-in splits plus any you have created.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "screen_components",
        "description": (
            "Rank EVERY component by estimated causal importance in one "
            "gradient pass. Use this before any intervention when you do not "
            "already know which components to look at — it covers all ~480 "
            "heads in a few seconds, where patching them individually would "
            "take many minutes. Results are gradient approximations, so "
            "verify the top handful with run_ablation or "
            "run_activation_patching."),
        "input_schema": {
            "type": "object",
            "properties": {
                "family": {"type": "string",
                           "description": "Family to screen against; pass a "
                                          "family you created to find what "
                                          "matters for that behaviour."},
                "top_k": {"type": "integer", "description": "Rows to return (default 15)"},
                "kind": {"type": "string", "enum": ["head", "mlp"],
                         "description": "Restrict to one component kind"},
            },
            "required": [],
        },
    },
    {
        "name": "run_ablation",
        "description": "NEW intervention: ablate the named components and measure the effect on the metric. Loads the run's model on first use, then ~10s per component. Appends new evidence to the run. Prefer recorded evidence when it already answers the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {"type": "array", "items": {"type": "string"}},
                "baseline": {"type": "string", "enum": ["zero", "mean", "resample"]},
                "family": {"type": "string",
                           "description": "Family to measure on; defaults to "
                                          "the run's 'train' split. Pass a "
                                          "family you created to test a "
                                          "specific behaviour."},
            },
            "required": ["components"],
        },
    },
    {
        "name": "run_activation_patching",
        "description": "NEW intervention: patch activations between clean and corrupt runs. Loads the run's model on first use, then ~10s per component. Appends new evidence to the run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {"type": "array", "items": {"type": "string"}},
                "direction": {"type": "string", "enum": ["denoising", "noising"]},
                "family": {"type": "string",
                           "description": "Family to measure on; defaults to "
                                          "the run's 'train' split."},
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

            # Interventions take seconds to minutes; report from the worker
            # thread rather than going silent.
            def progress(text):
                asyncio.run_coroutine_threadsafe(
                    send({"type": "status", "message": text}), loop)

            results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                await send({"type": "status",
                            "message": _label(block["name"])})
                try:
                    out = await loop.run_in_executor(
                        None, lambda b=block: self._exec(
                            b["name"], b["input"], progress))
                except Exception as e:
                    out = {"error": str(e)}
                if out.get("_focus"):
                    await send({"type": "focus",
                                "components": out.pop("_focus"),
                                "metric": out.pop("_metric", None)})
                if out.pop("_store_changed", False):
                    await send({"type": "run_updated"})
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

    def _exec(self, name: str, inp: Dict, progress=None) -> Dict:
        say = progress or (lambda _m: None)
        fn = {
            "get_run_summary": lambda: self._summary(),
            "query_components": lambda: self._query(inp),
            "get_component": lambda: self._component(
                inp["component"], inp.get("include_evidence", True)),
            "compare_components": lambda: self._compare(
                inp["components"], inp.get("metrics")),
            "create_prompt_family": lambda: self._make_family(inp, say),
            "list_families": lambda: self._list_families(),
            "screen_components": lambda: self._screen(inp, say),
            "run_ablation": lambda: self._ablation(
                inp["components"], inp.get("baseline", "mean"),
                inp.get("family"), say),
            "run_activation_patching": lambda: self._patch(
                inp["components"], inp.get("direction", "denoising"),
                inp.get("family"), say),
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

    def _make_family(self, inp: Dict, say) -> Dict:
        """Turn the agent's sentences into a usable prompt family."""
        import torch
        from ..tasks import build_text_family, logit_diff
        s = self.session
        name = inp["name"]
        items = inp["items"]
        say(f"Tokenizing {len(items)} prompt pairs")
        tk = s.ensure_tokenizer()
        try:
            built = build_text_family(
                tk, items, name=name,
                description=inp.get("description", ""))
        except ValueError as e:
            return {"error": str(e), "hint": "Every pair failed validation. "
                    "Check token-length matching and single-token answers."}

        fam = built["family"]
        s.custom_families[name] = fam

        # Sanity-check the contrast before anything is measured against it:
        # if the clean prompts do not favour the answer, the family is not
        # testing what it claims and every downstream number is noise.
        say("Checking the family separates clean from corrupt")
        s.ensure_model(progress=say)
        with torch.no_grad():
            lc, _ = s.adapter.run(fam.clean)
            lk, _ = s.adapter.run(fam.corrupt)
        clean_ld = float(logit_diff(lc, fam))
        corrupt_ld = float(logit_diff(lk, fam))

        verdict = ("usable" if clean_ld > 0 and clean_ld > corrupt_ld else
                   "weak — the model does not clearly prefer the clean answer, "
                   "so results on this family will be hard to interpret")
        return {
            "family": name,
            "n_kept": built["n_kept"],
            "n_rejected": len(built["rejected"]),
            "rejected": built["rejected"][:10],
            "seq_len": built["seq_len"],
            "logit_diff_clean": round(clean_ld, 4),
            "logit_diff_corrupt": round(corrupt_ld, 4),
            "separation": round(clean_ld - corrupt_ld, 4),
            "verdict": verdict,
            "note": "logit_diff_clean > 0 means the model prefers the answer "
                    "over the distractor on the clean prompts.",
        }

    def _list_families(self) -> Dict:
        s = self.session
        out = []
        for n, f in (s.families or {}).items():
            out.append({"name": f.name, "alias": n, "n": f.n,
                        "seq_len": f.seq_len, "kind": "built-in (random tokens)"})
        for n, f in s.custom_families.items():
            out.append({"name": n, "alias": n, "n": f.n, "seq_len": f.seq_len,
                        "kind": "real text", "example": f.texts[0] if f.texts else ""})
        if not out:
            return {"families": [], "note": "Model not loaded yet; built-in "
                                            "families appear once it is."}
        return {"families": out}

    MAX_INTERVENTION = 32

    def _screen(self, inp: Dict, say) -> Dict:
        """One gradient pass over every component, instead of N forward passes."""
        from ..methods.attribution import attribution_patching
        s = self.session
        s.ensure_model(progress=say)
        fam = s.resolve_family(inp.get("family"))
        say(f"Screening all components against {fam.name}")
        recs = attribution_patching(s.adapter, fam, s.store)

        rows = []
        for rec in recs:
            comp = rec.components[0]
            if inp.get("kind") and comp.kind != inp["kind"]:
                continue
            rows.append({"component": comp.name, "kind": comp.kind,
                         "layer": comp.layer,
                         "approx_delta": round(rec.values["approx_delta"], 5),
                         "abs": round(rec.values["abs_approx_delta"], 5)})
        rows.sort(key=lambda r: r["abs"], reverse=True)
        top = rows[:inp.get("top_k", 15)]
        return {"family": fam.name, "n_screened": len(rows), "top": top,
                "caveat": "gradient approximation — verify the leaders with a "
                          "real intervention before concluding",
                "recorded_to": s.store.root,
                "_focus": [r["component"] for r in top],
                "_store_changed": True}

    def _check_batch(self, comps: List[str]) -> Optional[Dict]:
        if len(comps) > self.MAX_INTERVENTION:
            return {"error": f"{len(comps)} components is too many for one "
                             f"intervention (limit {self.MAX_INTERVENTION}); "
                             "each costs a forward pass. Use "
                             "screen_components to rank everything cheaply, "
                             "then verify the top few here."}
        return None

    def _ablation(self, comps: List[str], baseline: str, family, say) -> Dict:
        guard = self._check_batch(comps)
        if guard:
            return guard
        from ..methods.ablation import ablate
        from ..schema import Component
        s = self.session
        s.ensure_model(progress=say)
        fam = s.resolve_family(family)
        ref = (fam if family in s.custom_families
               else s.families["reference"])
        out = {}
        for i, n in enumerate(comps, 1):
            say(f"Ablating {n} ({baseline}) on {fam.name} — {i}/{len(comps)}")
            rec = ablate(s.adapter, fam, Component.parse(n),
                         baseline, ref_fam=ref)
            s.store.add(rec)
            out[n] = {k: round(float(v), 5) for k, v in rec.values.items()}
        return {"baseline": baseline, "family": fam.name, "results": out,
                "recorded_to": s.store.root,
                "_focus": list(out), "_store_changed": True}

    def _patch(self, comps: List[str], direction: str, family, say) -> Dict:
        guard = self._check_batch(comps)
        if guard:
            return guard
        from ..methods.patching import activation_patch
        from ..schema import Component
        s = self.session
        s.ensure_model(progress=say)
        fam = s.resolve_family(family)
        out = {}
        for i, n in enumerate(comps, 1):
            say(f"Patching {n} ({direction}) on {fam.name} — {i}/{len(comps)}")
            rec = activation_patch(s.adapter, fam, Component.parse(n),
                                   direction)
            s.store.add(rec)
            out[n] = {k: round(float(v), 5) for k, v in rec.values.items()}
        return {"direction": direction, "family": fam.name, "results": out,
                "recorded_to": s.store.root,
                "_focus": list(out), "_store_changed": True}


def _label(tool: str) -> str:
    return {"get_run_summary": "Reading run summary",
            "query_components": "Querying components",
            "get_component": "Fetching component detail",
            "compare_components": "Comparing components",
            "create_prompt_family": "Building prompt family",
            "list_families": "Listing prompt families",
            "screen_components": "Screening all components",
            "run_ablation": "Running ablation",
            "run_activation_patching": "Running activation patching",
            }.get(tool, tool)
