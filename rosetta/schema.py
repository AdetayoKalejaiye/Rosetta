"""Common evidence format for Rosetta.

Every discovery method writes EvidenceRecords in this schema so that humans,
TensorBoard, the classifier, and the explainer LLM all read the same thing.
Design rule from the project brief: every result carries the exact
methodological choices (baseline, corruption, metric) that produced it,
because patching/ablation results change with those choices.
"""
from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------- components


@dataclass(frozen=True)
class Component:
    """Addressable unit of the model.

    kind:
      head     - one attention (query) head's output contribution
      kv_head  - one shared key/value head (GQA: affects a whole query group!)
      mlp      - one MLP block's output
      resid    - residual stream after a block
    """

    kind: str  # "head" | "kv_head" | "mlp" | "resid"
    layer: int
    index: Optional[int] = None  # head index / kv-head index; None for mlp/resid

    def __post_init__(self):
        assert self.kind in {"head", "kv_head", "mlp", "resid"}, self.kind

    @property
    def name(self) -> str:
        if self.kind == "head":
            return f"L{self.layer}.H{self.index}"
        if self.kind == "kv_head":
            return f"L{self.layer}.KV{self.index}"
        if self.kind == "mlp":
            return f"L{self.layer}.MLP"
        return f"L{self.layer}.RESID"

    @staticmethod
    def parse(name: str) -> "Component":
        layer_part, rest = name.split(".", 1)
        layer = int(layer_part[1:])
        if rest.startswith("KV"):
            return Component("kv_head", layer, int(rest[2:]))
        if rest.startswith("H"):
            return Component("head", layer, int(rest[1:]))
        if rest == "MLP":
            return Component("mlp", layer)
        return Component("resid", layer)

    def to_dict(self):
        return {"kind": self.kind, "layer": self.layer, "index": self.index}


# ----------------------------------------------------------------------------- specs


@dataclass
class InterventionSpec:
    """Exact description of what was done, so results are reproducible.

    method: weight_analysis | profiling | dla | zero_ablation | mean_ablation |
            resample_ablation | joint_ablation | activation_patching |
            path_patching | attribution_patching | eap_ig | probe | faithfulness
    """

    method: str
    baseline: Optional[str] = None          # zero | mean | resample (for ablations)
    reference_family: Optional[str] = None  # where the mean/resample activations come from
    mean_scope: Optional[str] = None        # "per_position" | "global"
    direction: Optional[str] = None         # patching: "noising" (clean run gets corrupt act)
                                            #           "denoising" (corrupt run gets clean act)
    corruption: Optional[str] = None        # how the corrupt prompts were built
    metric: Optional[str] = None            # logit_diff | answer_logprob | kl_from_clean
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class EvidenceRecord:
    """One measured result about one or more components."""

    components: List[Component]
    family: str                      # prompt family the measurement ran on
    spec: InterventionSpec
    values: Dict[str, float]         # e.g. {"delta_logit_diff": -1.3, "recovery": 0.8}
    notes: str = ""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "record_id": self.record_id,
            "components": [c.name for c in self.components],
            "family": self.family,
            "spec": self.spec.to_dict(),
            "values": self.values,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d) -> "EvidenceRecord":
        return EvidenceRecord(
            components=[Component.parse(n) for n in d["components"]],
            family=d["family"],
            spec=InterventionSpec(**d["spec"]),
            values=d["values"],
            notes=d.get("notes", ""),
            record_id=d.get("record_id", uuid.uuid4().hex[:12]),
            created_at=d.get("created_at", time.time()),
        )


# ----------------------------------------------------------------------------- profiles


@dataclass
class ComponentProfile:
    """Everything Rosetta knows about one component, in one place.

    This is what the classifier receives (numerical fields) and what the
    inspection view / explainer renders.
    """

    component: Component
    weight_stats: Dict[str, float] = field(default_factory=dict)
    attention_stats: Dict[str, float] = field(default_factory=dict)   # prev_token, induction, ...
    activation_stats: Dict[str, float] = field(default_factory=dict)  # mean, std, kurtosis, ...
    top_examples: List[Dict[str, Any]] = field(default_factory=list)  # texts + positions + acts
    evidence_ids: List[str] = field(default_factory=list)             # links into the store
    detector_labels: List[str] = field(default_factory=list)          # rule-based candidates
    cluster: Optional[int] = None
    disagreements: List[str] = field(default_factory=list)            # e.g. zero vs mean ablation

    def feature_vector_fields(self) -> Dict[str, float]:
        """Flat numeric view for clustering / classification."""
        out: Dict[str, float] = {}
        for prefix, d in (
            ("w", self.weight_stats),
            ("attn", self.attention_stats),
            ("act", self.activation_stats),
        ):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    out[f"{prefix}.{k}"] = float(v)
        return out

    def to_dict(self):
        return {
            "component": self.component.name,
            "weight_stats": self.weight_stats,
            "attention_stats": self.attention_stats,
            "activation_stats": self.activation_stats,
            "top_examples": self.top_examples,
            "evidence_ids": self.evidence_ids,
            "detector_labels": self.detector_labels,
            "cluster": self.cluster,
            "disagreements": self.disagreements,
        }

    @staticmethod
    def from_dict(d) -> "ComponentProfile":
        return ComponentProfile(
            component=Component.parse(d["component"]),
            weight_stats=d.get("weight_stats", {}),
            attention_stats=d.get("attention_stats", {}),
            activation_stats=d.get("activation_stats", {}),
            top_examples=d.get("top_examples", []),
            evidence_ids=d.get("evidence_ids", []),
            detector_labels=d.get("detector_labels", []),
            cluster=d.get("cluster"),
            disagreements=d.get("disagreements", []),
        )


# ----------------------------------------------------------------------------- explanations


@dataclass
class Prediction:
    """A machine-checkable prediction attached to an explanation."""

    method: str                     # zero_ablation | mean_ablation | activation_patching | ...
    components: List[str]           # component names
    family: str                     # family to test on (should be held out)
    metric: str                     # logit_diff | answer_logprob
    expected_direction: str         # "decrease" | "increase" | "no_change"
    min_abs_effect: float = 0.0     # threshold for decrease/increase
    tolerance: float = 0.05         # threshold for no_change
    # filled in by verify:
    measured: Optional[float] = None
    passed: Optional[bool] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d) -> "Prediction":
        return Prediction(**d)


@dataclass
class Explanation:
    """A candidate 'Rosetta Stone' entry proposed by the explainer."""

    component: str
    candidate_function: str
    relevant_contexts: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)     # record ids or prose
    contradictions: List[str] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)
    next_experiment: str = ""
    source: str = "template"  # which explainer produced it

    def to_dict(self):
        d = dataclasses.asdict(self)
        return d

    @staticmethod
    def from_dict(d) -> "Explanation":
        preds = [Prediction.from_dict(p) for p in d.get("predictions", [])]
        d = dict(d)
        d["predictions"] = preds
        return Explanation(**d)


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False)
