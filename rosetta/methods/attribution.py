"""Attribution patching and an integrated-gradients variant (EAP-IG style).

These are the fast SCREENS: gradient approximations of patching effects used
to rank many components cheaply. The pipeline then verifies selected
candidates with actual interventions - the brief's attribution-vs-intervention
distinction is kept in the record notes and in how the pipeline uses them.

attribution_patching:
  one clean forward+backward gives grads for every hook point at once;
  approx delta(metric) for noising component c  ~=  g_clean . (a_corrupt - a_clean)

eap_ig:
  average the gradient along the straight line between corrupt and clean
  activations (m interpolation steps, all screened points patched jointly -
  a joint-patch IG approximation, recorded in the spec), then dot with the
  activation difference. More faithful than the single-point linearization.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..model import HEAD_OUT, MLP_OUT, ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec
from ..tasks import PromptFamily, get_metric


def _screen_keys(cfg) -> List[Tuple[str, int]]:
    keys = [(HEAD_OUT, l) for l in range(cfg.n_layers)]
    keys += [(MLP_OUT, l) for l in range(cfg.n_layers)]
    return keys


def _per_component(adapter, key, tensor) -> Dict[Component, torch.Tensor]:
    point, layer = key
    if point == HEAD_OUT:
        return {Component("head", layer, h): tensor[..., adapter.head_slice(h)]
                for h in range(adapter.cfg.n_heads)}
    return {Component("mlp", layer): tensor}


def attribution_patching(adapter: ModelAdapter, fam: PromptFamily, store,
                         metric_name: str = "logit_diff") -> List[EvidenceRecord]:
    metric = get_metric(metric_name)
    keys = _screen_keys(adapter.cfg)

    logits, clean_cache = adapter.run(fam.clean, capture=keys, grad=True)
    m = metric(logits, fam)
    m.backward()
    grads = {k: clean_cache[k].grad.detach() for k in keys}
    clean_acts = {k: clean_cache[k].detach() for k in keys}
    _, corrupt_cache = adapter.run(fam.corrupt, capture=keys)
    adapter.model.zero_grad(set_to_none=True)

    recs = []
    for key in keys:
        diff = corrupt_cache[key] - clean_acts[key]
        attr_full = grads[key] * diff
        for comp, attr in _per_component(adapter, key, attr_full).items():
            val = attr.sum().item() / fam.n
            rec = EvidenceRecord(
                components=[comp], family=fam.name,
                spec=InterventionSpec(method="attribution_patching",
                                      direction="noising",
                                      corruption=fam.corruption,
                                      metric=metric_name),
                values={"approx_delta": val, "abs_approx_delta": abs(val)},
                notes="gradient approximation of a patching effect; verify with "
                      "actual interventions before trusting",
            )
            recs.append(store.add(rec))
            store.profile(comp.name).evidence_ids.append(rec.record_id)
    return recs


def eap_ig(adapter: ModelAdapter, fam: PromptFamily, store,
           steps: int = 5, metric_name: str = "logit_diff") -> List[EvidenceRecord]:
    metric = get_metric(metric_name)
    keys = _screen_keys(adapter.cfg)

    _, clean_cache = adapter.run(fam.clean, capture=keys)
    _, corrupt_cache = adapter.run(fam.corrupt, capture=keys)
    diff = {k: (corrupt_cache[k] - clean_cache[k]) for k in keys}

    avg_grad = {k: torch.zeros_like(clean_cache[k]) for k in keys}
    for i in range(steps):
        alpha = (i + 0.5) / steps  # midpoint rule from corrupt(a=1) to clean(a=0)
        interp_holders: Dict = {}
        edits = {}
        for key in keys:
            target = (corrupt_cache[key] + (1 - alpha) *
                      (clean_cache[key] - corrupt_cache[key]))

            def make_edit(key=key, target=target):
                def edit(x):
                    t = target.clone().requires_grad_(True)
                    interp_holders[key] = t
                    return t
                return edit
            edits[key] = make_edit()
        logits, _ = adapter.run(fam.clean, edits=edits, grad=True)
        m = metric(logits, fam)
        m.backward()
        for key in keys:
            g = interp_holders[key].grad
            if g is not None:
                avg_grad[key] += g.detach() / steps
        adapter.model.zero_grad(set_to_none=True)

    recs = []
    for key in keys:
        attr_full = avg_grad[key] * diff[key]
        for comp, attr in _per_component(adapter, key, attr_full).items():
            val = attr.sum().item() / fam.n
            rec = EvidenceRecord(
                components=[comp], family=fam.name,
                spec=InterventionSpec(method="eap_ig", direction="noising",
                                      corruption=fam.corruption,
                                      metric=metric_name,
                                      extra={"ig_steps": steps,
                                             "note": "joint-patch IG approximation"}),
                values={"approx_delta": val, "abs_approx_delta": abs(val)},
                notes="IG-averaged gradient approximation; verify with actual "
                      "interventions before trusting",
            )
            recs.append(store.add(rec))
            store.profile(comp.name).evidence_ids.append(rec.record_id)
    return recs


def rank_candidates(store, method: str = "eap_ig", top_k: int = 8) -> List[Component]:
    """Rank screened components by |approx effect| for verification."""
    scored = []
    for rec in store.by_method(method):
        scored.append((rec.values.get("abs_approx_delta", 0.0), rec.components[0]))
    scored.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for _, comp in scored:
        if comp.name not in seen:
            seen.add(comp.name)
            out.append(comp)
        if len(out) >= top_k:
            break
    return out
