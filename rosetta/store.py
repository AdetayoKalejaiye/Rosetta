"""Append-only evidence store.

Records land in evidence.jsonl; profiles in profiles.json. Everything is plain
JSON so the inspection view, the classifier, and the explainer read one source
of truth, and so runs can be diffed and versioned.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .schema import Component, ComponentProfile, EvidenceRecord, Explanation


class EvidenceStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._records: List[EvidenceRecord] = []
        self._by_component: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self.profiles: Dict[str, ComponentProfile] = {}
        self._load()

    # ------------------------------------------------------------------ io
    @property
    def _evidence_path(self):
        return os.path.join(self.root, "evidence.jsonl")

    @property
    def _profiles_path(self):
        return os.path.join(self.root, "profiles.json")

    def _load(self):
        if os.path.exists(self._evidence_path):
            with open(self._evidence_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = EvidenceRecord.from_dict(json.loads(line))
                    self._index(rec)
        if os.path.exists(self._profiles_path):
            with open(self._profiles_path) as f:
                for d in json.load(f):
                    p = ComponentProfile.from_dict(d)
                    self.profiles[p.component.name] = p

    def _index(self, rec: EvidenceRecord):
        self._records.append(rec)
        for c in rec.components:
            self._by_component[c.name].append(rec)

    # ------------------------------------------------------------------ write
    def add(self, rec: EvidenceRecord) -> EvidenceRecord:
        self._index(rec)
        with open(self._evidence_path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
        return rec

    def add_many(self, recs: Iterable[EvidenceRecord]) -> List[EvidenceRecord]:
        return [self.add(r) for r in recs]

    def put_profile(self, profile: ComponentProfile):
        self.profiles[profile.component.name] = profile

    def save_profiles(self):
        with open(self._profiles_path, "w") as f:
            json.dump([p.to_dict() for p in self.profiles.values()], f, indent=2)

    # ------------------------------------------------------------------ read
    def all_records(self) -> List[EvidenceRecord]:
        return list(self._records)

    def for_component(self, component: Component | str) -> List[EvidenceRecord]:
        name = component if isinstance(component, str) else component.name
        return list(self._by_component.get(name, []))

    def by_method(self, method: str, component: Optional[str] = None) -> List[EvidenceRecord]:
        recs = self._records if component is None else self._by_component.get(component, [])
        return [r for r in recs if r.spec.method == method]

    def profile(self, component: Component | str) -> ComponentProfile:
        name = component if isinstance(component, str) else component.name
        if name not in self.profiles:
            self.profiles[name] = ComponentProfile(component=Component.parse(name))
        return self.profiles[name]

    # ------------------------------------------------------------------ derived
    def note_disagreements(self, component: str, threshold: float = 0.5):
        """Flag baseline disagreements (the brief: a head can look important
        under zero ablation but not under mean/resample). Stored on the profile
        so the explainer sees it."""
        prof = self.profile(component)
        deltas = {}
        for method in ("zero_ablation", "mean_ablation", "resample_ablation"):
            recs = self.by_method(method, component)
            if recs:
                deltas[method] = recs[-1].values.get("delta_logit_diff", 0.0)
        if len(deltas) >= 2:
            vals = list(deltas.values())
            if max(vals) - min(vals) > threshold:
                msg = "ablation baselines disagree: " + ", ".join(
                    f"{m}={v:+.3f}" for m, v in deltas.items()
                )
                if msg not in prof.disagreements:
                    prof.disagreements.append(msg)
        return prof.disagreements
