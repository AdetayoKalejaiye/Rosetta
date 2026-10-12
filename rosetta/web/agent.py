"""Rosetta agent — LLM-driven interpretability orchestrator.

Uses Claude API with tool use to interpret user requests and run the
appropriate pipeline methods against the loaded model. Falls back to
a simple command interface when no API key is available.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Any, Callable, Coroutine, Dict, List

SYSTEM_PROMPT = """\
You are Rosetta, an AI assistant for mechanistic interpretability of neural networks.
You help users understand what individual components (attention heads, MLP layers) \
do inside transformer models. You have tools to run various interpretability \
techniques against the currently loaded model.

## Approach
1. Identify what the user wants to find (which component, behavior, or circuit).
2. Pick the right technique(s) from your tools.
3. Run the analysis and interpret the results concisely.
4. Suggest a follow-up experiment.

## Technique guide
- **Profiling** — attention patterns (prev-token, induction, sink, entropy) and \
activation stats. Run this first to see what each head does.
- **DLA** — direct logit attribution: who writes toward the answer at the output. \
Readout contribution, not a removal effect.
- **Attribution screen** — fast gradient ranking of all components. Use to find \
candidates, then verify with ablation.
- **Ablation** — remove a component (zero/mean/resample baseline) and measure the \
damage. Gold standard for causal importance. Different baselines can disagree—\
that is informative.
- **Activation patching** — transplant activations between clean and corrupt runs. \
Denoising = measures sufficiency, noising = necessity.
- **Detectors** — rule-based labels (induction head, prev-token head, etc.).
- **Weight analysis** — structural view of weight norms, SVD, Q-composition.

## Important context
- The toy model is a 3-layer GQA transformer trained on sequence repetition \
(induction task). Its interesting components are induction heads and their \
supporting circuitry.
- GQA means multiple query heads share one KV head. Ablating a KV head hits \
the whole group.
- Always mention component names explicitly (e.g., "L2.H1") so the \
visualization can highlight them.
- Be concise. Report numbers, explain what they mean, suggest next steps.
- If a request does not make sense for this model, explain why honestly.
"""

TOOLS = [
    {
        "name": "get_model_info",
        "description": "Get architecture and config of the loaded model.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_weight_analysis",
        "description": "Analyze weight matrices: norms, SVD, Q-composition. Structural first look.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_profiling",
        "description": "Profile attention patterns (prev-token, induction, sink, entropy) and activation stats for every head and MLP. Essential first step.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_dla",
        "description": "Direct logit attribution: which components write toward the answer token. Positive = toward answer.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_attribution_screen",
        "description": "Attribution patching + EAP-IG: cheaply rank all components by causal importance. Returns top-K candidates for further investigation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "description": "How many candidates (default 6)"},
            },
            "required": [],
        },
    },
    {
        "name": "run_ablation",
        "description": "Ablate specific components (zero/mean/resample) and measure effect. Gold standard for causal importance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array", "items": {"type": "string"},
                    "description": "e.g. ['L2.H1', 'L1.MLP']",
                },
                "baseline": {
                    "type": "string", "enum": ["zero", "mean", "resample"],
                    "description": "Default: mean",
                },
            },
            "required": ["components"],
        },
    },
    {
        "name": "run_activation_patching",
        "description": "Transplant activations between clean/corrupt runs. 'denoising'=sufficiency, 'noising'=necessity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Component names to patch",
                },
                "direction": {
                    "type": "string", "enum": ["denoising", "noising"],
                    "description": "Default: denoising",
                },
            },
            "required": ["components"],
        },
    },
    {
        "name": "run_detectors",
        "description": "Rule-based pattern detectors: prev-token head, induction head, attention sink, etc. Requires profiling first.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_component_profile",
        "description": "Full profile for one component: stats, evidence, labels, top activating examples.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {"type": "string", "description": "e.g. 'L2.H1'"},
            },
            "required": ["component"],
        },
    },
    {
        "name": "list_components",
        "description": "List all model components with analysis status and detector labels.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_full_pipeline",
        "description": "Run the complete 12-stage Rosetta pipeline. Comprehensive but slower.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "description": "Candidates to investigate (default 6)"},
            },
            "required": [],
        },
    },
]

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
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


class RosettaAgent:
    def __init__(self, session):
        self.session = session
        self.history: List[Dict] = []
        self._has_api = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

    async def handle_message(self, user_msg: str, send: SendFn):
        if not self._has_api:
            await self._command_fallback(user_msg, send)
            return

        self.history.append({"role": "user", "content": user_msg})
        loop = asyncio.get_event_loop()

        while True:
            await send({"type": "status", "message": "Thinking..."})
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
                                "message": block["text"]})

            if response.get("stop_reason") != "tool_use":
                break

            tool_results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                name, inp, tid = block["name"], block["input"], block["id"]
                await send({"type": "status",
                            "message": f"Running {_tool_label(name)}..."})
                try:
                    result = await loop.run_in_executor(
                        None, lambda n=name, i=inp: self._exec(n, i))
                    if result.get("highlights"):
                        await send({"type": "results", "tool": name,
                                    "highlights": result["highlights"]})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tid,
                        "content": json.dumps(result, default=str),
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tid,
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True,
                    })
            self.history.append({"role": "user", "content": tool_results})

    # ---------------------------------------------------------------------- tools

    def _exec(self, name: str, inp: Dict) -> Dict:
        s = self.session
        if not s.adapter and name != "get_model_info":
            return {"error": "No model loaded. Ask the user to load one first."}
        dispatch = {
            "get_model_info": lambda: s.model_info,
            "run_weight_analysis": self._weight_analysis,
            "run_profiling": self._profiling,
            "run_dla": self._dla,
            "run_attribution_screen": lambda: self._attribution(inp.get("top_k", 6)),
            "run_ablation": lambda: self._ablation(
                inp["components"], inp.get("baseline", "mean")),
            "run_activation_patching": lambda: self._patching(
                inp["components"], inp.get("direction", "denoising")),
            "run_detectors": self._detectors,
            "get_component_profile": lambda: self._profile(inp["component"]),
            "list_components": self._list,
            "run_full_pipeline": lambda: self._full_pipeline(inp.get("top_k", 6)),
        }
        fn = dispatch.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        return fn()

    def _weight_analysis(self) -> Dict:
        from ..methods.weights import run_weight_analysis
        s = self.session
        run_weight_analysis(s.adapter, s.store)
        hl = {}
        for name, prof in s.store.profiles.items():
            if prof.weight_stats:
                hl[name] = {"weight_norm": prof.weight_stats.get("norm", 0)}
        return {"status": "complete", "highlights": hl}

    def _profiling(self) -> Dict:
        from ..methods.profiling import run_profiling
        s = self.session
        run_profiling(s.adapter, s.families["train"], s.store)
        hl = {}
        for name, prof in s.store.profiles.items():
            stats = {}
            if prof.attention_stats:
                stats.update(prof.attention_stats)
            if prof.activation_stats:
                stats.update(prof.activation_stats)
            if stats:
                hl[name] = stats
        return {"status": "complete", "highlights": hl}

    def _dla(self) -> Dict:
        from ..methods.dla import run_dla
        s = self.session
        run_dla(s.adapter, s.families["train"], s.store)
        hl = {}
        for rec in s.store.by_method("dla"):
            for comp in rec.components:
                hl[comp.name] = dict(rec.values)
        return {"status": "complete", "highlights": hl}

    def _attribution(self, top_k: int) -> Dict:
        from ..methods.attribution import attribution_patching, eap_ig, rank_candidates
        s = self.session
        attribution_patching(s.adapter, s.families["train"], s.store)
        eap_ig(s.adapter, s.families["train"], s.store, steps=5)
        cands = rank_candidates(s.store, method="eap_ig", top_k=top_k)
        if not cands:
            cands = rank_candidates(s.store, method="attribution_patching",
                                    top_k=top_k)
        s.candidates = cands
        hl = {}
        for rec in s.store.by_method("eap_ig"):
            for comp in rec.components:
                val = rec.values.get("attribution",
                                    rec.values.get("mean_grad", 0))
                hl[comp.name] = {"attribution": abs(float(val))}
        return {"status": "complete",
                "candidates": [c.name for c in cands],
                "highlights": hl}

    def _ablation(self, names: List[str], baseline: str) -> Dict:
        from ..methods.ablation import ablate
        from ..schema import Component
        s = self.session
        hl = {}
        for n in names:
            comp = Component.parse(n)
            rec = ablate(s.adapter, s.families["train"], comp, baseline,
                         ref_fam=s.families["reference"])
            s.store.add(rec)
            hl[n] = {k: float(v) for k, v in rec.values.items()}
        return {"status": "complete", "highlights": hl}

    def _patching(self, names: List[str], direction: str) -> Dict:
        from ..methods.patching import activation_patch
        from ..schema import Component
        s = self.session
        hl = {}
        for n in names:
            comp = Component.parse(n)
            rec = activation_patch(s.adapter, s.families["train"],
                                   comp, direction)
            s.store.add(rec)
            hl[n] = {k: float(v) for k, v in rec.values.items()}
        return {"status": "complete", "highlights": hl}

    def _detectors(self) -> Dict:
        from ..analyze.classify import run_detectors
        s = self.session
        if not s.store.profiles:
            return {"error": "No profiles yet. Run profiling first."}
        hl = {}
        for name, prof in s.store.profiles.items():
            run_detectors(prof)
            if prof.detector_labels:
                hl[name] = {"labels": list(prof.detector_labels)}
        return {"status": "complete", "highlights": hl}

    def _profile(self, name: str) -> Dict:
        s = self.session
        prof = s.store.profiles.get(name)
        if not prof:
            return {"error": f"No profile for {name}. Run profiling first."}
        d = prof.to_dict()
        d["evidence"] = [r.to_dict() for r in s.store.for_component(name)]
        return d

    def _list(self) -> Dict:
        s = self.session
        cfg = s.adapter.cfg
        comps = []
        for layer in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                n = f"L{layer}.H{head}"
                p = s.store.profiles.get(n)
                comps.append({"name": n, "kind": "head",
                              "analyzed": p is not None,
                              "labels": p.detector_labels if p else [],
                              "n_evidence": len(p.evidence_ids) if p else 0})
            n = f"L{layer}.MLP"
            p = s.store.profiles.get(n)
            comps.append({"name": n, "kind": "mlp",
                          "analyzed": p is not None,
                          "labels": p.detector_labels if p else [],
                          "n_evidence": len(p.evidence_ids) if p else 0})
        return {"components": comps}

    def _full_pipeline(self, top_k: int) -> Dict:
        from ..pipeline import run_pipeline
        from ..store import EvidenceStore
        s = self.session
        result = run_pipeline(s.adapter, s.families, s.out_dir, top_k=top_k)
        s.store = EvidenceStore(os.path.join(s.out_dir, "store"))
        hl = {}
        for name, prof in s.store.profiles.items():
            entry = {}
            if prof.attention_stats:
                entry.update(prof.attention_stats)
            if prof.detector_labels:
                entry["labels"] = list(prof.detector_labels)
            if entry:
                hl[name] = entry
        return {"status": "complete", "pipeline_result": result,
                "highlights": hl}

    # ---------------------------------------------------------------------- fallback

    async def _command_fallback(self, msg: str, send: SendFn):
        msg = msg.strip()
        if not msg.startswith("/"):
            await send({"type": "chat", "sender": "system",
                        "message": "No API key set — using command mode. "
                                   "Type **/help** for commands."})
            return
        parts = msg[1:].split()
        cmd, args = (parts[0] if parts else ""), parts[1:]
        loop = asyncio.get_event_loop()
        if cmd == "help":
            await send({"type": "chat", "sender": "system",
                        "message": "**Commands:** /profile, /dla, /screen, "
                                   "/ablate L2.H1 [mean], /patch L2.H1 "
                                   "[denoising], /detect, /list, /pipeline, "
                                   "/info L2.H1"})
        elif cmd == "profile":
            r = await loop.run_in_executor(None, self._profiling)
            await send({"type": "results", "tool": "run_profiling",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": "Profiling complete."})
        elif cmd == "dla":
            r = await loop.run_in_executor(None, self._dla)
            await send({"type": "results", "tool": "run_dla",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": "DLA complete."})
        elif cmd == "screen":
            r = await loop.run_in_executor(None, lambda: self._attribution(6))
            await send({"type": "results", "tool": "run_attribution_screen",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": f"Candidates: {r.get('candidates', [])}"})
        elif cmd == "detect":
            r = await loop.run_in_executor(None, self._detectors)
            await send({"type": "results", "tool": "run_detectors",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": "Detectors complete."})
        elif cmd == "list":
            r = await loop.run_in_executor(None, self._list)
            lines = [f"  {c['name']}: {', '.join(c['labels']) or '—'}"
                     for c in r["components"]]
            await send({"type": "chat", "sender": "system",
                        "message": "**Components:**\n" + "\n".join(lines)})
        elif cmd == "pipeline":
            r = await loop.run_in_executor(None, lambda: self._full_pipeline(6))
            await send({"type": "results", "tool": "run_full_pipeline",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": "Pipeline complete."})
        elif cmd == "ablate" and args:
            bl = args[1] if len(args) > 1 else "mean"
            r = await loop.run_in_executor(
                None, lambda: self._ablation([args[0]], bl))
            await send({"type": "results", "tool": "run_ablation",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": f"Ablation result: {json.dumps(r.get('highlights', {}), indent=2)}"})
        elif cmd == "patch" and args:
            d = args[1] if len(args) > 1 else "denoising"
            r = await loop.run_in_executor(
                None, lambda: self._patching([args[0]], d))
            await send({"type": "results", "tool": "run_activation_patching",
                        "highlights": r.get("highlights", {})})
            await send({"type": "chat", "sender": "system",
                        "message": f"Patching result: {json.dumps(r.get('highlights', {}), indent=2)}"})
        elif cmd == "info" and args:
            r = await loop.run_in_executor(None, lambda: self._profile(args[0]))
            await send({"type": "chat", "sender": "system",
                        "message": f"```json\n{json.dumps(r, indent=2, default=str)}\n```"})
        else:
            await send({"type": "chat", "sender": "system",
                        "message": f"Unknown command: **{cmd}**. Type /help."})


def _tool_label(name: str) -> str:
    return name.replace("run_", "").replace("_", " ")
