"""The LLM stage: propose Rosetta Stone entries from structured evidence.

Each explainer gets an evidence PACKET - the profile's numbers, detector
labels, method disagreements, and the raw records including negative examples
(small measured effects are included on purpose) - and must return:

  candidate function, relevant contexts, supporting evidence, contradictions,
  PREDICTED INTERVENTION EFFECTS (machine-checkable, on a held-out family),
  and the next experiment.

Two implementations:
  TemplateExplainer  - deterministic, rule-based stand-in so the whole
                       pipeline runs offline. Its predictions are real and
                       testable (verify.py runs them), which keeps the
                       feedback loop closed even without an API key.
  APIExplainer       - calls an Anthropic-compatible /v1/messages endpoint if
                       ANTHROPIC_API_KEY is set, asks for strict JSON, and
                       falls back to the template on any failure. (Delphi-style
                       explanation scoring would slot in at this layer.)
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List, Optional

from ..schema import ComponentProfile, EvidenceRecord, Explanation, Prediction
from ..store import EvidenceStore


# ------------------------------------------------------------- evidence packet


def build_packet(store: EvidenceStore, component: str,
                 heldout_family: str) -> Dict:
    prof = store.profiles[component]
    records = []
    for rec in store.for_component(component):
        d = rec.to_dict()
        d.pop("created_at", None)
        records.append(d)
    return {
        "component": component,
        "profile": prof.to_dict(),
        "detector_labels": prof.detector_labels,
        "method_disagreements": prof.disagreements,
        "records_including_negative_results": records,
        "heldout_family_for_predictions": heldout_family,
        "instructions": (
            "Propose: candidate_function, relevant_contexts, "
            "supporting_evidence (record ids), contradictions, predictions "
            "(each: method, components, family=the held-out family, metric, "
            "expected_direction, min_abs_effect), next_experiment. "
            "Note that dla is a readout contribution, not a removal effect, "
            "and attribution methods are approximations."),
    }


# ---------------------------------------------------------------------- template


class TemplateExplainer:
    """Rule-based explanation writer. Grounded in the same evidence the LLM
    would see; every claim it makes is traceable to a record."""

    source = "template"

    def explain(self, store: EvidenceStore, component: str,
                heldout_family: str) -> Explanation:
        prof = store.profiles[component]
        labels = prof.detector_labels
        feats = prof.feature_vector_fields()

        # candidate function from strongest signals
        parts = []
        if "induction_head" in labels:
            parts.append("induction-style copying (attends to the token after a "
                         "previous occurrence of the current token)")
        if "prev_token_head" in labels:
            parts.append("previous-token attention (likely supplies the shifted "
                         "context an induction head reads)")
        if "attention_sink" in labels:
            parts.append("attention sink on position 0 (often a no-op/default)")
        if "direct_logit_writer_positive" in labels:
            parts.append("writes directly in the answer-token direction at the "
                         "final position")
        if "direct_logit_writer_negative" in labels:
            parts.append("suppresses the answer token at the final position")
        if not parts:
            parts.append("function unclear from current evidence")
        candidate = "; ".join(parts)

        # supporting evidence & contradictions from records
        supporting, contradictions = [], []
        zero = mean = None
        for rec in store.for_component(component):
            v = rec.values
            if rec.spec.method == "zero_ablation":
                zero = v.get("delta_logit_diff")
            if rec.spec.method == "mean_ablation":
                mean = v.get("delta_logit_diff")
            if rec.spec.method in ("zero_ablation", "mean_ablation",
                                   "resample_ablation", "activation_patching",
                                   "path_patching", "dla"):
                supporting.append(rec.record_id)
        important = any(abs(x) > 0.2 for x in (zero, mean) if x is not None)
        if not important and ("induction_head" in labels or
                              "direct_logit_writer_positive" in labels):
            contradictions.append(
                "attention/readout pattern suggests task relevance, but measured "
                "ablation effects are small - pattern may be present-but-unused")
        contradictions.extend(prof.disagreements)

        # machine-checkable predictions on the held-out family
        preds: List[Prediction] = []
        expected_drop = None
        for rec in store.for_component(component):
            if rec.spec.method == "mean_ablation":
                expected_drop = rec.values.get("delta_logit_diff")
        if expected_drop is not None and expected_drop < -0.1:
            preds.append(Prediction(
                method="mean_ablation", components=[component],
                family=heldout_family, metric="logit_diff",
                expected_direction="decrease",
                min_abs_effect=abs(expected_drop) * 0.4,  # generalization margin
            ))
        else:
            preds.append(Prediction(
                method="mean_ablation", components=[component],
                family=heldout_family, metric="logit_diff",
                expected_direction="no_change", tolerance=0.15,
            ))
        # denoising patching prediction for components with high recovery
        for rec in store.for_component(component):
            if (rec.spec.method == "activation_patching"
                    and rec.spec.direction == "denoising"
                    and rec.values.get("recovery", 0.0) > 0.3):
                preds.append(Prediction(
                    method="activation_patching", components=[component],
                    family=heldout_family, metric="logit_diff",
                    expected_direction="increase",
                    min_abs_effect=0.1,
                ))
                break

        contexts = ["repeated-sequence / copying contexts"] if (
            "induction_head" in labels or "same_token_head" in labels
        ) else ["(contexts not yet narrowed - see next_experiment)"]

        nxt = ("test on a prompt family with the repeat at a different offset "
               "to separate positional from content-based attention"
               if "induction_head" in labels else
               "run path patching against the strongest downstream head to "
               "test whether anything reads this component's output")

        return Explanation(
            component=component, candidate_function=candidate,
            relevant_contexts=contexts, supporting_evidence=supporting[:8],
            contradictions=contradictions, predictions=preds,
            next_experiment=nxt, source=self.source,
        )


# --------------------------------------------------------------------- via API


class APIExplainer:
    """Anthropic-compatible messages endpoint. Falls back to the template."""

    source = "llm"

    def __init__(self, model: str = "claude-sonnet-4-6",
                 endpoint: str = "https://api.anthropic.com/v1/messages"):
        self.model = model
        self.endpoint = endpoint
        self.fallback = TemplateExplainer()

    def _call(self, packet: Dict) -> Optional[str]:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        body = json.dumps({
            "model": self.model, "max_tokens": 1500,
            "system": ("You are an interpretability researcher. Respond ONLY "
                       "with a JSON object matching the requested fields. No "
                       "markdown fences, no preamble."),
            "messages": [{"role": "user", "content": json.dumps(packet)}],
        }).encode()
        req = urllib.request.Request(self.endpoint, data=body, headers={
            "content-type": "application/json", "x-api-key": key,
            "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return "".join(b.get("text", "") for b in data.get("content", [])
                           if b.get("type") == "text")
        except Exception:
            return None

    def explain(self, store: EvidenceStore, component: str,
                heldout_family: str) -> Explanation:
        packet = build_packet(store, component, heldout_family)
        text = self._call(packet)
        if text:
            try:
                d = json.loads(text.replace("```json", "").replace("```", "").strip())
                preds = [Prediction(
                    method=p["method"], components=p.get("components", [component]),
                    family=p.get("family", heldout_family),
                    metric=p.get("metric", "logit_diff"),
                    expected_direction=p["expected_direction"],
                    min_abs_effect=float(p.get("min_abs_effect", 0.0)),
                    tolerance=float(p.get("tolerance", 0.05)),
                ) for p in d.get("predictions", [])]
                return Explanation(
                    component=component,
                    candidate_function=d.get("candidate_function", ""),
                    relevant_contexts=d.get("relevant_contexts", []),
                    supporting_evidence=d.get("supporting_evidence", []),
                    contradictions=d.get("contradictions", []),
                    predictions=preds,
                    next_experiment=d.get("next_experiment", ""),
                    source=self.source,
                )
            except Exception:
                pass
        expl = self.fallback.explain(store, component, heldout_family)
        expl.source = "template(fallback)"
        return expl


def get_explainer(kind: str = "auto"):
    if kind == "template":
        return TemplateExplainer()
    if kind == "llm":
        return APIExplainer()
    return APIExplainer() if os.environ.get("ANTHROPIC_API_KEY") else TemplateExplainer()
