"""Direct logit attribution (DLA).

Which components directly push the final-position output toward the answer
token vs the distractor. Each head's residual contribution is
head_out_slice @ W_O_slice; we pass it through the final RMSNorm's measured
scale (taken from the actual clean run, so the nonlinearity is folded at the
observed operating point) and read it against W_U[answer]-W_U[distractor].

Recorded explicitly as a READOUT contribution, not a removal effect - the
brief calls this distinction out, so every record's notes repeat it.
"""
from __future__ import annotations

from typing import List

import torch

from ..model import HEAD_OUT, MLP_OUT, RESID, ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec
from ..tasks import PromptFamily


def run_dla(adapter: ModelAdapter, fam: PromptFamily, store) -> List[EvidenceRecord]:
    cfg = adapter.cfg
    capture = [(HEAD_OUT, l) for l in range(cfg.n_layers)]
    capture += [(MLP_OUT, l) for l in range(cfg.n_layers)]
    capture += [(RESID, cfg.n_layers - 1)]
    _, cache = adapter.run(fam.clean, capture=capture)

    resid_final = cache[(RESID, cfg.n_layers - 1)][:, -1, :]          # [B,D]
    rms = torch.sqrt(resid_final.pow(2).mean(-1, keepdim=True) + 1e-6)  # [B,1]
    gamma = adapter._final_norm_weight()                               # [D]
    W_U = adapter._unembed()                                           # [V,D]
    idx = torch.arange(fam.n)
    direction = (W_U[fam.answer] - W_U[fam.distractor])                # [B,D]

    recs = []
    for l in range(cfg.n_layers):
        head_out = cache[(HEAD_OUT, l)][:, -1, :]                      # [B,H*dh]
        Wo = adapter._o_proj(l).weight                                 # [D,H*dh]
        for h in range(cfg.n_heads):
            sl = adapter.head_slice(h)
            contrib = head_out[:, sl] @ Wo[:, sl].T                    # [B,D]
            scaled = contrib * gamma[None, :] / rms
            dla = (scaled * direction).sum(-1).mean().item()
            comp = Component("head", l, h)
            rec = EvidenceRecord(
                components=[comp], family=fam.name,
                spec=InterventionSpec(method="dla", metric="logit_diff"),
                values={"dla_logit_diff": dla},
                notes="readout contribution at final position, NOT a removal effect",
            )
            recs.append(store.add(rec))
            store.profile(comp.name).evidence_ids.append(rec.record_id)
            store.profile(comp.name).activation_stats["dla_logit_diff"] = dla

        mlp_out = cache[(MLP_OUT, l)][:, -1, :]                        # [B,D]
        scaled = mlp_out * gamma[None, :] / rms
        dla = (scaled * direction).sum(-1).mean().item()
        comp = Component("mlp", l)
        rec = EvidenceRecord(
            components=[comp], family=fam.name,
            spec=InterventionSpec(method="dla", metric="logit_diff"),
            values={"dla_logit_diff": dla},
            notes="readout contribution at final position, NOT a removal effect",
        )
        recs.append(store.add(rec))
        store.profile(comp.name).evidence_ids.append(rec.record_id)
        store.profile(comp.name).activation_stats["dla_logit_diff"] = dla
    return recs
