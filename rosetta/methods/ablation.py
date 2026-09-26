"""Ablations under three baselines: zero, mean, resample.

The brief is explicit: compare all three rather than treating one as
definitive, and record the reference distribution. Mean ablation here uses a
SEPARATE reference family with corresponding token positions (per_position
scope when lengths match, global otherwise); resample substitutes an actual
activation from another example (seeded).

kv_head ablation is implemented separately from head ablation: with GQA, one
shared value/key representation feeds a whole group of query heads, and the
record notes which query heads were affected.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..model import HEAD_OUT, KV_K, KV_V, MLP_OUT, ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec
from ..tasks import PromptFamily, get_metric


def _point_for(comp: Component) -> str:
    if comp.kind == "head":
        return HEAD_OUT
    if comp.kind == "mlp":
        return MLP_OUT
    if comp.kind == "kv_head":
        return KV_V  # value-side by default; key-side available via extra arg
    raise ValueError(f"cannot ablate {comp.kind}")


def _reference_cache(adapter: ModelAdapter, ref_fam: PromptFamily,
                     points) -> Dict:
    _, cache = adapter.run(ref_fam.clean, capture=list(points))
    return cache


def _baseline_value(baseline: str, x: torch.Tensor, ref: Optional[torch.Tensor],
                    same_len: bool, gen: torch.Generator):
    """x: current activation slice [B,S,dh...]. Returns replacement."""
    if baseline == "zero":
        return torch.zeros_like(x)
    if baseline == "mean":
        assert ref is not None
        if same_len:  # mean over reference examples, kept per position
            m = ref.mean(dim=0, keepdim=True)          # [1,S,...]
            return m.expand_as(x).clone()
        m = ref.mean(dim=(0, 1), keepdim=True)         # global fallback
        return m.expand_as(x).clone()
    if baseline == "resample":
        assert ref is not None
        idx = torch.randint(0, ref.shape[0], (x.shape[0],), generator=gen)
        return ref[idx, : x.shape[1]].clone()
    raise ValueError(baseline)


def ablate(adapter: ModelAdapter, fam: PromptFamily, comp: Component,
           baseline: str, ref_fam: Optional[PromptFamily] = None,
           metric_name: str = "logit_diff", seed: int = 0,
           kv_side: str = "v") -> EvidenceRecord:
    metric = get_metric(metric_name)
    gen = torch.Generator().manual_seed(seed)
    point = KV_K if (comp.kind == "kv_head" and kv_side == "k") else _point_for(comp)
    key = (point, comp.layer)

    ref_cache = None
    same_len = True
    if baseline in ("mean", "resample"):
        assert ref_fam is not None, "mean/resample need a reference family"
        ref_cache = _reference_cache(adapter, ref_fam, [key])
        same_len = ref_fam.seq_len == fam.seq_len

    clean_logits, _ = adapter.run(fam.clean)
    m_clean = metric(clean_logits, fam).item()

    def slice_of(t):
        if comp.kind == "head":
            return t[..., adapter.head_slice(comp.index)]
        if comp.kind == "kv_head":
            return t[..., adapter.kv_slice(comp.index)]
        return t  # mlp: whole output

    def edit(x):
        x = x.clone()
        ref = slice_of(ref_cache[key]) if ref_cache is not None else None
        repl = _baseline_value(baseline, slice_of(x), ref, same_len, gen)
        if comp.kind == "head":
            x[..., adapter.head_slice(comp.index)] = repl
        elif comp.kind == "kv_head":
            x[..., adapter.kv_slice(comp.index)] = repl
        else:
            x = repl
        return x

    abl_logits, _ = adapter.run(fam.clean, edits={key: edit})
    m_abl = metric(abl_logits, fam).item()

    notes = ""
    if comp.kind == "kv_head":
        affected = adapter.query_heads_for_kv(comp.index)
        notes = (f"GQA: shared {kv_side}-representation; this intervention hits "
                 f"query heads {affected} together, distinct from ablating a "
                 f"single head's output")

    spec = InterventionSpec(
        method=f"{baseline}_ablation", baseline=baseline,
        reference_family=ref_fam.name if ref_fam else None,
        mean_scope=("per_position" if same_len else "global") if baseline == "mean" else None,
        metric=metric_name, seed=seed if baseline == "resample" else None,
        extra={"kv_side": kv_side} if comp.kind == "kv_head" else {},
    )
    return EvidenceRecord(
        components=[comp], family=fam.name, spec=spec,
        values={"metric_clean": m_clean, "metric_ablated": m_abl,
                "delta_logit_diff" if metric_name == "logit_diff" else f"delta_{metric_name}":
                    m_abl - m_clean},
        notes=notes,
    )


def joint_ablation(adapter: ModelAdapter, fam: PromptFamily,
                   comps: List[Component], baseline: str,
                   ref_fam: Optional[PromptFamily] = None,
                   metric_name: str = "logit_diff", seed: int = 0) -> EvidenceRecord:
    """Ablate a set together: how components work in combination (moves from
    'these heads matter' toward circuit-level claims)."""
    metric = get_metric(metric_name)
    gen = torch.Generator().manual_seed(seed)
    keys = sorted({(_point_for(c), c.layer) for c in comps})
    ref_cache = None
    same_len = True
    if baseline in ("mean", "resample"):
        assert ref_fam is not None
        ref_cache = _reference_cache(adapter, ref_fam, keys)
        same_len = ref_fam.seq_len == fam.seq_len

    clean_logits, _ = adapter.run(fam.clean)
    m_clean = metric(clean_logits, fam).item()

    edits = {}
    for key in keys:
        point, layer = key
        layer_comps = [c for c in comps if c.layer == layer and _point_for(c) == point]

        def make_edit(key=key, layer_comps=layer_comps):
            def edit(x):
                x = x.clone()
                for c in layer_comps:
                    if c.kind == "head":
                        sl = adapter.head_slice(c.index)
                    elif c.kind == "kv_head":
                        sl = adapter.kv_slice(c.index)
                    else:
                        sl = slice(None)
                    ref = ref_cache[key][..., sl] if ref_cache is not None else None
                    x[..., sl] = _baseline_value(baseline, x[..., sl], ref,
                                                 same_len, gen)
                return x
            return edit
        edits[key] = make_edit()

    abl_logits, _ = adapter.run(fam.clean, edits=edits)
    m_abl = metric(abl_logits, fam).item()
    spec = InterventionSpec(
        method="joint_ablation", baseline=baseline,
        reference_family=ref_fam.name if ref_fam else None,
        metric=metric_name, seed=seed,
        extra={"set_size": len(comps)},
    )
    return EvidenceRecord(
        components=comps, family=fam.name, spec=spec,
        values={"metric_clean": m_clean, "metric_ablated": m_abl,
                "delta_logit_diff": m_abl - m_clean},
    )


def run_ablation_battery(adapter, fam, comps: List[Component], ref_fam,
                         store, metric_name="logit_diff", seed=0):
    """All three baselines per component + disagreement flagging."""
    for comp in comps:
        for baseline in ("zero", "mean", "resample"):
            rec = ablate(adapter, fam, comp, baseline,
                         ref_fam=ref_fam if baseline != "zero" else ref_fam,
                         metric_name=metric_name, seed=seed)
            store.add(rec)
            store.profile(comp.name).evidence_ids.append(rec.record_id)
        store.note_disagreements(comp.name)
