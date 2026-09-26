"""Activation & attention profiling.

For each head: where it attends (previous token, position 0 sink, same-token,
induction offset), attention entropy, and activation statistics of its output
slice, plus top-activating examples with positions so the inspection view can
highlight text. MLP and residual activations are profiled too, so the
investigation covers more than attention heads.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..model import HEAD_OUT, MLP_OUT, RESID, ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec
from ..tasks import PromptFamily


def _attention_pattern_scores(pats: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    """pats: [L,B,H,S,S] (rows=query pos, cols=key pos). Returns per-head [L,H]."""
    L, B, H, S, _ = pats.shape
    dev = pats.device
    q = torch.arange(S, device=dev)

    prev = pats[..., q[1:], q[:-1]].mean(dim=(-1, 1))                  # attn to i-1
    sink = pats[..., :, 0].mean(dim=(-1, 1))                           # attn to pos 0
    diag = pats[..., q, q].mean(dim=(-1, 1))                           # attn to self

    same = (tokens[:, :, None] == tokens[:, None, :])                  # [B,S,S]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev), -1)
    same_prev_occ = same & causal
    same_score = (pats * same_prev_occ[None, :, None]).sum(-1).mean(dim=(-1, 1))

    # induction: attend from current pos to (previous occurrence of SAME token)+1
    ind_mask = torch.zeros(B, S, S, dtype=torch.bool, device=dev)
    ind_mask[:, :, 1:] = same_prev_occ[:, :, :-1]
    ind_score = (pats * ind_mask[None, :, None]).sum(-1).mean(dim=(-1, 1))

    ent = -(pats.clamp_min(1e-9).log() * pats).sum(-1).mean(dim=(-1, 1))
    return {
        "prev_token": prev, "sink_pos0": sink, "self_attn": diag,
        "same_token": same_score, "induction": ind_score, "entropy": ent,
    }


def _act_stats(x: torch.Tensor, prefix="") -> Dict[str, float]:
    x = x.float().flatten()
    mean, std = x.mean(), x.std().clamp_min(1e-9)
    kurt = (((x - mean) / std) ** 4).mean() - 3.0
    return {
        f"{prefix}mean": mean.item(), f"{prefix}std": std.item(),
        f"{prefix}kurtosis": kurt.item(),
        f"{prefix}abs_max": x.abs().max().item(),
        f"{prefix}sparsity": (x.abs() < 0.05 * std).float().mean().item(),
    }


def _top_examples(norms: torch.Tensor, fam: PromptFamily, k=3) -> List[dict]:
    """norms: [B,S] activation magnitude. Returns readable top-k."""
    B, S = norms.shape
    flat = norms.flatten()
    top = torch.topk(flat, min(k, flat.numel()))
    out = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        b, s = divmod(idx, S)
        toks = fam.clean[b].tolist()
        out.append({
            "family": fam.name, "example": b, "position": s,
            "activation": round(val, 4),
            "tokens": toks,
            "token_at_pos": toks[s],
            "window": toks[max(0, s - 4): s + 1],
        })
    return out


def run_profiling(adapter: ModelAdapter, fam: PromptFamily, store) -> None:
    cfg = adapter.cfg
    tokens = fam.clean
    pats = adapter.attention_patterns(tokens)
    scores = _attention_pattern_scores(pats, tokens)

    capture = [(HEAD_OUT, l) for l in range(cfg.n_layers)]
    capture += [(MLP_OUT, l) for l in range(cfg.n_layers)]
    capture += [(RESID, l) for l in range(cfg.n_layers)]
    _, cache = adapter.run(tokens, capture=capture)

    for l in range(cfg.n_layers):
        head_out = cache[(HEAD_OUT, l)]  # [B,S,H*dh]
        for h in range(cfg.n_heads):
            comp = Component("head", l, h)
            prof = store.profile(comp.name)
            astats = {k: v[l, h].item() for k, v in scores.items()}
            prof.attention_stats.update(astats)
            slc = head_out[..., adapter.head_slice(h)]
            prof.activation_stats.update(_act_stats(slc))
            prof.top_examples = _top_examples(slc.norm(dim=-1), fam)
            rec = store.add(EvidenceRecord(
                components=[comp], family=fam.name,
                spec=InterventionSpec(method="profiling", metric=None),
                values={**astats, **_act_stats(slc)},
            ))
            prof.evidence_ids.append(rec.record_id)
        for kind, point in (("mlp", MLP_OUT), ("resid", RESID)):
            comp = Component(kind, l)
            prof = store.profile(comp.name)
            stats = _act_stats(cache[(point, l)])
            prof.activation_stats.update(stats)
            rec = store.add(EvidenceRecord(
                components=[comp], family=fam.name,
                spec=InterventionSpec(method="profiling"), values=stats,
            ))
            prof.evidence_ids.append(rec.record_id)
