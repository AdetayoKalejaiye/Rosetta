"""Run discovery and loading.

Reading a finished run needs no torch and no checkpoint — the evidence store
is plain JSON on disk. A model only gets loaded when the user asks for a NEW
intervention, which is the expensive path and strictly opt-in.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

_COMP_RE = re.compile(r"^L(\d+)\.(H(\d+)|KV(\d+)|MLP|RESID)$")

# Signed metrics get a diverging scale; everything else is magnitude-only.
SIGNED_METRICS = {
    "dla_logit_diff", "dla_score", "delta_logit_diff", "delta_answer_logprob",
    "mean", "attribution", "recovery", "logit_diff",
}


def _parse_component(name: str) -> Optional[Dict]:
    m = _COMP_RE.match(name)
    if not m:
        return None
    layer = int(m.group(1))
    if m.group(3) is not None:
        return {"kind": "head", "layer": layer, "index": int(m.group(3))}
    if m.group(4) is not None:
        return {"kind": "kv_head", "layer": layer, "index": int(m.group(4))}
    if m.group(2) == "MLP":
        return {"kind": "mlp", "layer": layer, "index": None}
    return {"kind": "resid", "layer": layer, "index": None}


def discover_runs(root: str = "runs") -> List[Dict]:
    """Every directory under *root* holding a store/profiles.json."""
    out = []
    root_p = Path(root)
    if not root_p.exists():
        return out
    for d in sorted(root_p.iterdir()):
        prof_path = d / "store" / "profiles.json"
        if not prof_path.exists():
            continue
        try:
            summary = _summarize(d, prof_path)
        except Exception:
            continue
        out.append(summary)
    out.sort(key=lambda r: r["updated_at"], reverse=True)
    return out


def _summarize(d: Path, prof_path: Path) -> Dict:
    with open(prof_path) as f:
        profiles = json.load(f)
    n_layers = n_heads = 0
    for p in profiles:
        parsed = _parse_component(p["component"])
        if not parsed:
            continue
        n_layers = max(n_layers, parsed["layer"] + 1)
        if parsed["kind"] == "head":
            n_heads = max(n_heads, parsed["index"] + 1)

    ev_path = d / "store" / "evidence.jsonl"
    n_evidence = 0
    if ev_path.exists():
        with open(ev_path, "rb") as f:
            n_evidence = sum(1 for _ in f)

    map_path = d / "rosetta_map.json"
    n_entries = 0
    if map_path.exists():
        try:
            with open(map_path) as f:
                n_entries = len(json.load(f))
        except Exception:
            pass

    return {
        "id": d.name,
        "path": str(d),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_profiles": len(profiles),
        "n_evidence": n_evidence,
        "n_explanations": n_entries,
        "updated_at": prof_path.stat().st_mtime,
    }


def load_run(path: str) -> Dict:
    """Everything the frontend needs to render a run, minus the heavy parts.

    top_examples and raw evidence records are excluded here and fetched per
    component on demand — smol's evidence.jsonl alone is 8MB.
    """
    d = Path(path)
    with open(d / "store" / "profiles.json") as f:
        raw_profiles = json.load(f)

    n_layers = n_heads = 0
    components: Dict[str, Dict] = {}
    metrics: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}

    for p in raw_profiles:
        name = p["component"]
        parsed = _parse_component(name)
        if not parsed:
            continue
        n_layers = max(n_layers, parsed["layer"] + 1)
        if parsed["kind"] == "head":
            n_heads = max(n_heads, parsed["index"] + 1)

        stats: Dict[str, float] = {}
        for group, prefix in (("attention_stats", ""),
                              ("activation_stats", ""),
                              ("weight_stats", "w.")):
            for k, v in (p.get(group) or {}).items():
                if isinstance(v, (int, float)):
                    key = prefix + k
                    stats[key] = float(v)
                    slot = metrics.setdefault(
                        key, {"min": v, "max": v, "n": 0, "group": group})
                    slot["min"] = min(slot["min"], v)
                    slot["max"] = max(slot["max"], v)
                    slot["n"] += 1

        for lab in p.get("detector_labels") or []:
            labels[lab] = labels.get(lab, 0) + 1

        components[name] = {
            **parsed,
            "stats": stats,
            "labels": list(p.get("detector_labels") or []),
            "cluster": p.get("cluster"),
            "disagreements": list(p.get("disagreements") or []),
            "n_evidence": len(p.get("evidence_ids") or []),
            "n_examples": len(p.get("top_examples") or []),
        }

    for key, slot in metrics.items():
        slot["signed"] = (key.lstrip("w.") in SIGNED_METRICS) or slot["min"] < 0

    explanations = {}
    map_path = d / "rosetta_map.json"
    if map_path.exists():
        try:
            with open(map_path) as f:
                for entry in json.load(f):
                    explanations[entry["component"]] = {
                        "candidate_function":
                            entry["explanation"].get("candidate_function", ""),
                        "contradictions":
                            entry["explanation"].get("contradictions", []),
                        "next_experiment":
                            entry["explanation"].get("next_experiment", ""),
                        "source": entry["explanation"].get("source", ""),
                        "confidence": entry.get("confidence"),
                        "predictions": [
                            {"method": p.get("method"),
                             "metric": p.get("metric"),
                             "expected_direction": p.get("expected_direction"),
                             "measured": p.get("measured"),
                             "passed": p.get("passed")}
                            for p in entry["explanation"].get("predictions", [])
                        ],
                    }
        except Exception:
            pass

    return {
        "id": d.name,
        "path": str(d),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "components": components,
        "metrics": metrics,
        "labels": labels,
        "explanations": explanations,
    }


def component_detail(path: str, comp: str) -> Dict:
    """Full detail for one component: profile, examples, evidence records."""
    d = Path(path)
    detail: Dict = {"component": comp}

    with open(d / "store" / "profiles.json") as f:
        for p in json.load(f):
            if p["component"] == comp:
                detail["profile"] = p
                break

    records = []
    ev_path = d / "store" / "evidence.jsonl"
    if ev_path.exists():
        with open(ev_path) as f:
            for line in f:
                line = line.strip()
                if not line or comp not in line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if comp in rec.get("components", []):
                    records.append(rec)
    records.sort(key=lambda r: r.get("created_at", 0))
    detail["evidence"] = records
    return detail
