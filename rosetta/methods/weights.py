"""Weight analysis: structural candidates before running anything.

Per query head: norms and singular-value summaries of its Q/O slices (and of
the K/V slices of its SHARED kv head - recorded under kv.* so it is obvious
that those numbers are shared across the whole query group).

Also computes Q- and V-composition scores between heads in different layers
(||W_qk_2 @ W_ov_1|| / (||W_qk_2|| ||W_ov_1||)), a cheap structural hint that
head 2 reads what head 1 writes.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..model import ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec


def _svd_stats(m: torch.Tensor, k: int = 3) -> Dict[str, float]:
    s = torch.linalg.svdvals(m.float())
    total = s.sum().clamp_min(1e-9)
    out = {f"sv{i}": s[i].item() if i < len(s) else 0.0 for i in range(k)}
    out["sv_top_frac"] = (s[:k].sum() / total).item()
    out["eff_rank"] = ((s.sum() ** 2) / (s.pow(2).sum().clamp_min(1e-9))).item()
    return out


def head_weight_matrices(adapter: ModelAdapter, layer: int, head: int):
    cfg = adapter.cfg
    dh = cfg.head_dim
    kv = head // cfg.group_size
    Wq = adapter._attn(layer).q_proj.weight[head * dh:(head + 1) * dh, :]   # [dh, D]
    Wk = adapter._attn(layer).k_proj.weight[kv * dh:(kv + 1) * dh, :]       # [dh, D] (shared)
    Wv = adapter._attn(layer).v_proj.weight[kv * dh:(kv + 1) * dh, :]       # shared
    Wo = adapter._o_proj(layer).weight[:, head * dh:(head + 1) * dh]        # [D, dh]
    return Wq, Wk, Wv, Wo, kv


def analyze_head(adapter: ModelAdapter, layer: int, head: int) -> Dict[str, float]:
    Wq, Wk, Wv, Wo, kv = head_weight_matrices(adapter, layer, head)
    stats: Dict[str, float] = {"shared_kv_head": float(kv)}
    stats["q_norm"] = Wq.norm().item()
    stats["o_norm"] = Wo.norm().item()
    stats["kv.k_norm"] = Wk.norm().item()
    stats["kv.v_norm"] = Wv.norm().item()
    W_qk = Wq.T @ Wk          # [D, D] bilinear form the head scores with
    W_ov = Wo @ Wv            # [D, D] what the head writes as fn of resid
    for prefix, m in (("qk", W_qk), ("ov", W_ov)):
        for k, v in _svd_stats(m).items():
            stats[f"{prefix}_{k}"] = v
    return stats


def analyze_mlp(adapter: ModelAdapter, layer: int) -> Dict[str, float]:
    down = adapter._mlp_down(layer).weight
    stats = {"down_norm": down.norm().item()}
    stats.update({f"down_{k}": v for k, v in _svd_stats(down).items()})
    return stats


def composition_scores(adapter: ModelAdapter, max_pairs: int = 200) -> List[EvidenceRecord]:
    """Structural evidence that later heads compose with earlier heads."""
    cfg = adapter.cfg
    recs: List[EvidenceRecord] = []
    ov, qk = {}, {}
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            Wq, Wk, Wv, Wo, _ = head_weight_matrices(adapter, l, h)
            ov[(l, h)] = Wo @ Wv
            qk[(l, h)] = Wq.T @ Wk
    pairs = []
    for (l1, h1) in ov:
        for (l2, h2) in qk:
            if l2 <= l1:
                continue
            a, b = qk[(l2, h2)], ov[(l1, h1)]
            score = (a @ b).norm() / (a.norm() * b.norm()).clamp_min(1e-9)
            pairs.append(((l1, h1), (l2, h2), score.item()))
    pairs.sort(key=lambda p: -p[2])
    for (l1, h1), (l2, h2), score in pairs[:max_pairs]:
        recs.append(EvidenceRecord(
            components=[Component("head", l1, h1), Component("head", l2, h2)],
            family="(weights)",
            spec=InterventionSpec(method="weight_analysis",
                                  extra={"kind": "q_composition"}),
            values={"q_composition": score},
            notes=f"L{l2}.H{h2} query-side reads what L{l1}.H{h1} writes (structural only)",
        ))
    return recs


def run_weight_analysis(adapter: ModelAdapter, store) -> None:
    for l in range(adapter.cfg.n_layers):
        for h in range(adapter.cfg.n_heads):
            comp = Component("head", l, h)
            stats = analyze_head(adapter, l, h)
            store.profile(comp.name).weight_stats.update(stats)
            store.add(EvidenceRecord(
                components=[comp], family="(weights)",
                spec=InterventionSpec(method="weight_analysis"),
                values={k: v for k, v in stats.items()},
            ))
        comp = Component("mlp", l)
        stats = analyze_mlp(adapter, l)
        store.profile(comp.name).weight_stats.update(stats)
        store.add(EvidenceRecord(
            components=[comp], family="(weights)",
            spec=InterventionSpec(method="weight_analysis"), values=stats,
        ))
    store.add_many(composition_scores(adapter, max_pairs=20))
