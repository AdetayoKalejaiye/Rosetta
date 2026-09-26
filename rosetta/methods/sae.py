"""Sparse autoencoders (flagged in the brief as a later addition).

Trains an SAE on cached activations from a hook point so the map can gain
feature-level nodes alongside heads/MLPs. Includes the safeguard the brief
asks for: measure how much information the reconstruction loses (recon loss +
downstream metric change when activations are replaced by their
reconstructions) BEFORE trusting feature interventions. feature_intervention
refuses to run unless that check passed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec
from ..tasks import PromptFamily, get_metric


class SAE(nn.Module):
    def __init__(self, d_in: int, d_features: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_features)
        self.dec = nn.Linear(d_features, d_in)
        self.d_in, self.d_features = d_in, d_features

    def encode(self, x):
        return F.relu(self.enc(x))

    def forward(self, x):
        f = self.encode(x)
        return self.dec(f), f


def train_sae(acts: torch.Tensor, d_features: int, steps: int = 300,
              l1: float = 1e-3, lr: float = 1e-3, seed: int = 0) -> SAE:
    """acts: [N, d_in] flattened activations."""
    torch.manual_seed(seed)
    sae = SAE(acts.shape[-1], d_features)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        recon, f = sae(acts)
        loss = F.mse_loss(recon, acts) + l1 * f.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sae.eval()
    return sae


def reconstruction_check(adapter: ModelAdapter, sae: SAE, fam: PromptFamily,
                         key, store, metric_name: str = "logit_diff",
                         max_metric_shift: float = 0.25,
                         max_frac_var_unexplained: float = 0.2) -> Tuple[bool, EvidenceRecord]:
    """Gate: is the SAE faithful enough to intervene through?"""
    metric = get_metric(metric_name)
    logits, cache = adapter.run(fam.clean, capture=[key])
    m_clean = metric(logits, fam).item()
    acts = cache[key]
    with torch.no_grad():
        recon, f = sae(acts)
    fvu = ((acts - recon).pow(2).sum() / (acts - acts.mean()).pow(2).sum()).item()
    l0 = (f > 0).float().sum(-1).mean().item()

    logits_r, _ = adapter.run(fam.clean, edits={key: lambda x: sae(x)[0]})
    m_recon = metric(logits_r, fam).item()
    shift = abs(m_recon - m_clean)
    ok = (fvu <= max_frac_var_unexplained) and (shift <= max_metric_shift)

    rec = store.add(EvidenceRecord(
        components=[Component("resid", key[1])],
        family=fam.name,
        spec=InterventionSpec(method="sae_reconstruction_check",
                              metric=metric_name,
                              extra={"point": key[0], "d_features": sae.d_features}),
        values={"frac_var_unexplained": fvu, "mean_l0": l0,
                "metric_clean": m_clean, "metric_reconstructed": m_recon,
                "metric_shift": shift, "passed": float(ok)},
        notes="feature interventions are only trusted if this check passes",
    ))
    return ok, rec


def feature_intervention(adapter: ModelAdapter, sae: SAE, fam: PromptFamily,
                         key, feature_idx: int, store, check_passed: bool,
                         metric_name: str = "logit_diff") -> EvidenceRecord:
    if not check_passed:
        raise RuntimeError(
            "SAE reconstruction check has not passed for this hook point; "
            "refusing to run a feature intervention (see brief: measure "
            "reconstruction loss before trusting feature interventions)")
    metric = get_metric(metric_name)
    logits, _ = adapter.run(fam.clean)
    m_clean = metric(logits, fam).item()

    def edit(x):
        f = sae.encode(x)
        f = f.clone()
        f[..., feature_idx] = 0.0
        return sae.dec(f)

    logits_a, _ = adapter.run(fam.clean, edits={key: edit})
    m_abl = metric(logits_a, fam).item()
    return store.add(EvidenceRecord(
        components=[Component("resid", key[1])],
        family=fam.name,
        spec=InterventionSpec(method="sae_feature_ablation", metric=metric_name,
                              extra={"feature": feature_idx, "point": key[0]}),
        values={"metric_clean": m_clean, "metric_feature_ablated": m_abl,
                "delta": m_abl - m_clean},
    ))
