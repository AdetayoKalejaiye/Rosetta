"""Activation patching and path patching.

Activation patching runs BOTH directions, as specified in the brief:
  noising    (clean -> corrupted): run the clean prompt, splice in the
             corrupt-run activation; does that disrupt the behavior?
  denoising  (corrupted -> clean): run the corrupt prompt, splice in the
             clean-run activation; does that restore the behavior?
Recovery is reported on the clean/corrupt metric gap.

Path patching (sender head -> receiver head) uses the standard three passes:
  1) cache clean + corrupt HEAD_OUT everywhere
  2) forward on clean with sender <- corrupt while every OTHER head output is
     frozen to its clean cache (MLPs left free to recompute; recorded in the
     spec); capture what the receiver now outputs
  3) forward on clean with only the receiver's output <- (2)'s capture;
     measure the metric change. Nonzero delta = evidence for a used edge.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..model import HEAD_OUT, KV_V, MLP_OUT, ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec
from ..tasks import PromptFamily, get_metric


def _key_and_slice(adapter: ModelAdapter, comp: Component):
    if comp.kind == "head":
        return (HEAD_OUT, comp.layer), adapter.head_slice(comp.index)
    if comp.kind == "kv_head":
        return (KV_V, comp.layer), adapter.kv_slice(comp.index)
    if comp.kind == "mlp":
        return (MLP_OUT, comp.layer), slice(None)
    raise ValueError(comp.kind)


def activation_patch(adapter: ModelAdapter, fam: PromptFamily, comp: Component,
                     direction: str = "denoising",
                     metric_name: str = "logit_diff") -> EvidenceRecord:
    assert direction in ("noising", "denoising")
    metric = get_metric(metric_name)
    key, sl = _key_and_slice(adapter, comp)

    clean_logits, clean_cache = adapter.run(fam.clean, capture=[key])
    corrupt_logits, corrupt_cache = adapter.run(fam.corrupt, capture=[key])
    m_clean = metric(clean_logits, fam).item()
    m_corrupt = metric(corrupt_logits, fam).item()
    gap = m_clean - m_corrupt

    if direction == "denoising":
        run_on, donor, base = fam.corrupt, clean_cache[key], m_corrupt
    else:
        run_on, donor, base = fam.clean, corrupt_cache[key], m_clean

    def edit(x):
        x = x.clone()
        x[..., sl] = donor[..., sl]
        return x

    patched_logits, _ = adapter.run(run_on, edits={key: edit})
    m_patched = metric(patched_logits, fam).item()
    # recovery: fraction of the gap moved toward the other run's metric
    recovery = (m_patched - base) / gap if abs(gap) > 1e-9 else 0.0

    spec = InterventionSpec(
        method="activation_patching", direction=direction,
        corruption=fam.corruption, metric=metric_name,
    )
    return EvidenceRecord(
        components=[comp], family=fam.name, spec=spec,
        values={"metric_clean": m_clean, "metric_corrupt": m_corrupt,
                "metric_patched": m_patched, "recovery": recovery},
        notes=("denoising recovery ~1 => this activation carries the clean signal; "
               "noising recovery ~-1... sign convention: recovery measured from the "
               "run being patched toward the donor run"),
    )


def path_patch(adapter: ModelAdapter, fam: PromptFamily, sender: Component,
               receiver: Component, metric_name: str = "logit_diff",
               freeze_mlps: bool = False) -> EvidenceRecord:
    assert sender.kind == "head" and receiver.kind == "head"
    assert (sender.layer, sender.index) != (receiver.layer, receiver.index)
    assert sender.layer <= receiver.layer, "sender must be at or before receiver"
    metric = get_metric(metric_name)
    cfg = adapter.cfg
    head_keys = [(HEAD_OUT, l) for l in range(cfg.n_layers)]
    mlp_keys = [(MLP_OUT, l) for l in range(cfg.n_layers)] if freeze_mlps else []

    clean_logits, clean_cache = adapter.run(fam.clean, capture=head_keys + mlp_keys)
    _, corrupt_cache = adapter.run(fam.corrupt, capture=head_keys)
    m_clean = metric(clean_logits, fam).item()

    s_key = (HEAD_OUT, sender.layer)
    r_key = (HEAD_OUT, receiver.layer)
    s_sl = adapter.head_slice(sender.index)
    r_sl = adapter.head_slice(receiver.index)

    # pass 2: freeze all head outputs to clean, except sender <- corrupt and
    # receiver left free; capture receiver's response.
    edits2 = {}
    for key in head_keys:
        def make_edit(key=key):
            def edit(x):
                frozen = clean_cache[key].clone()
                if key == s_key:
                    frozen[..., s_sl] = corrupt_cache[s_key][..., s_sl]
                if key == r_key:
                    frozen[..., r_sl] = x[..., r_sl]  # receiver recomputes
                return frozen
            return edit
        edits2[key] = make_edit()
    if freeze_mlps:
        for key in mlp_keys:
            if key[1] < receiver.layer:
                edits2[key] = (lambda key=key: (lambda x: clean_cache[key].clone()))()
    _, cache2 = adapter.run(fam.clean, edits=edits2, capture=[r_key])
    receiver_patched = cache2[r_key][..., r_sl]

    # pass 3: clean run, only receiver's output replaced.
    def edit3(x):
        x = x.clone()
        x[..., r_sl] = receiver_patched
        return x

    logits3, _ = adapter.run(fam.clean, edits={r_key: edit3})
    m_path = metric(logits3, fam).item()

    spec = InterventionSpec(
        method="path_patching", corruption=fam.corruption, metric=metric_name,
        extra={"sender": sender.name, "receiver": receiver.name,
               "mlps": "frozen" if freeze_mlps else "free"},
    )
    return EvidenceRecord(
        components=[sender, receiver], family=fam.name, spec=spec,
        values={"metric_clean": m_clean, "metric_path_patched": m_path,
                "delta_via_path": m_path - m_clean},
        notes=f"effect of corrupting ONLY the {sender.name}->{receiver.name} path",
    )
