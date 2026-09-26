"""The central artifact: an evolving map whose descriptions are backed by
experiments. Successful predictions raise an entry's confidence; failures
lower it and are kept in the history rather than deleted.

Confidence = (passes + 1) / (passes + fails + 2)  (Laplace-smoothed), so an
untested entry sits at 0.5 and moves with evidence.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schema import Explanation, Prediction


@dataclass
class MapEntry:
    component: str
    explanation: Explanation
    prediction_history: List[dict] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @property
    def passes(self):
        return sum(1 for p in self.prediction_history if p.get("passed") is True)

    @property
    def fails(self):
        return sum(1 for p in self.prediction_history if p.get("passed") is False)

    @property
    def confidence(self) -> float:
        return (self.passes + 1) / (self.passes + self.fails + 2)

    def record(self, pred: Prediction):
        self.prediction_history.append(pred.to_dict())
        self.updated_at = time.time()

    def to_dict(self):
        return {"component": self.component,
                "explanation": self.explanation.to_dict(),
                "prediction_history": self.prediction_history,
                "confidence": self.confidence,
                "updated_at": self.updated_at}

    @staticmethod
    def from_dict(d):
        return MapEntry(component=d["component"],
                        explanation=Explanation.from_dict(d["explanation"]),
                        prediction_history=d.get("prediction_history", []),
                        updated_at=d.get("updated_at", time.time()))


class RosettaMap:
    def __init__(self, path: str):
        self.path = path
        self.entries: Dict[str, MapEntry] = {}
        if os.path.exists(path):
            with open(path) as f:
                for d in json.load(f):
                    e = MapEntry.from_dict(d)
                    self.entries[e.component] = e

    def upsert(self, explanation: Explanation) -> MapEntry:
        if explanation.component in self.entries:
            entry = self.entries[explanation.component]
            entry.explanation = explanation   # history is kept across revisions
            entry.updated_at = time.time()
        else:
            entry = MapEntry(component=explanation.component,
                             explanation=explanation)
            self.entries[explanation.component] = entry
        return entry

    def record_outcome(self, component: str, pred: Prediction):
        self.entries[component].record(pred)

    def save(self):
        with open(self.path, "w") as f:
            json.dump([e.to_dict() for e in self.entries.values()], f, indent=2)

    # ------------------------------------------------------------- rendering
    def render_markdown(self, title="Rosetta Stone",
                        faithfulness: Optional[dict] = None) -> str:
        lines = [f"# {title}", "",
                 "Confidence = Laplace-smoothed prediction pass rate; entries "
                 "with failed predictions keep them in their history.", ""]
        if faithfulness:
            lines += ["## Proposed circuit",
                      f"- components: {', '.join(faithfulness['components'])}",
                      f"- baseline for outside-ablation: {faithfulness['baseline']}",
                      f"- behavior retention: {faithfulness['retention']:.2f}", ""]
        for entry in sorted(self.entries.values(), key=lambda e: -e.confidence):
            e = entry.explanation
            lines += [f"## {entry.component}  (confidence {entry.confidence:.2f}, "
                      f"{entry.passes} passed / {entry.fails} failed)",
                      f"**Candidate function:** {e.candidate_function}",
                      f"**Contexts:** {', '.join(e.relevant_contexts) or '-'}",
                      f"**Source:** {e.source}"]
            if e.contradictions:
                lines.append("**Contradictions / caveats:**")
                lines += [f"- {c}" for c in e.contradictions]
            if e.predictions:
                lines.append("**Predictions:**")
                for p in e.predictions:
                    status = ("PASSED" if p.passed else "FAILED"
                              if p.passed is False else "untested")
                    meas = f", measured {p.measured:+.3f}" if p.measured is not None else ""
                    lines.append(
                        f"- {p.method} on {'+'.join(p.components)} @ {p.family}: "
                        f"{p.metric} to {p.expected_direction} "
                        f"(threshold {p.min_abs_effect:.2f}) -> {status}{meas}")
            if e.next_experiment:
                lines.append(f"**Next experiment:** {e.next_experiment}")
            lines.append("")
        return "\n".join(lines)
