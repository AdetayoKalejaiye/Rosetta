"""Classifier stage.

The classifier receives the underlying numerical measurements (the profile
feature vectors), exactly as the brief specifies. Since a supervised
classifier has no labels at the start, the order of operations is:

  1. explicit pattern DETECTORS surface candidates (previous-token head,
     induction head, attention sink, high direct-logit writer, ...)
  2. unsupervised CLUSTERING groups components with similar measurement
     profiles (tiny numpy k-means; no sklearn dependency)
  3. LINEAR PROBES test whether a property is decodable from activations -
     evidence that information is PRESENT, with the caveat (kept in the
     record) that the model may not use it
  4. once entries are reviewed/experimentally supported, ProfileClassifier
     trains one-vs-rest logistic heads over the same features, supports
     multiple labels, and returns "unknown" below a confidence threshold.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..schema import Component, ComponentProfile, EvidenceRecord, InterventionSpec


# ------------------------------------------------------------------- detectors

DETECTOR_THRESHOLDS = {
    "prev_token_head": ("attn.prev_token", 0.4),
    "induction_head": ("attn.induction", 0.3),
    "attention_sink": ("attn.sink_pos0", 0.5),
    "same_token_head": ("attn.same_token", 0.4),
    "self_attending": ("attn.self_attn", 0.5),
}


def run_detectors(profile: ComponentProfile) -> List[str]:
    feats = profile.feature_vector_fields()
    labels = []
    for label, (field, thr) in DETECTOR_THRESHOLDS.items():
        if feats.get(field, 0.0) >= thr:
            labels.append(label)
    dla = feats.get("act.dla_logit_diff", 0.0)
    if dla > 0.5:
        labels.append("direct_logit_writer_positive")
    if dla < -0.5:
        labels.append("direct_logit_writer_negative")
    if feats.get("attn.entropy", 99.0) < 0.5:
        labels.append("focused_attention")
    profile.detector_labels = labels
    return labels


# ------------------------------------------------------------------ clustering


def _standardize(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    centers = X[rng.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None] - centers[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centers[j] = X[labels == j].mean(0)
    return labels


def feature_matrix(profiles: Sequence[ComponentProfile]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for p in profiles for k in p.feature_vector_fields()})
    X = np.array([[p.feature_vector_fields().get(k, 0.0) for k in keys]
                  for p in profiles])
    return X, keys


def cluster_profiles(profiles: Sequence[ComponentProfile], k: int = 4,
                     seed: int = 0) -> Dict[str, int]:
    if not profiles:
        return {}
    X, _ = feature_matrix(profiles)
    labels = kmeans(_standardize(X), k, seed=seed)
    out = {}
    for p, lab in zip(profiles, labels):
        p.cluster = int(lab)
        out[p.component.name] = int(lab)
    return out


# ---------------------------------------------------------------- linear probe


class LinearProbe:
    """Can a property be decoded from a component's activations?
    Positive result = information present; whether the model USES it still
    needs an intervention (the caveat travels with the record)."""

    def __init__(self, d_in: int, seed: int = 0):
        torch.manual_seed(seed)
        self.w = torch.nn.Linear(d_in, 1)

    def fit(self, X: torch.Tensor, y: torch.Tensor, steps=300, lr=1e-2):
        opt = torch.optim.Adam(self.w.parameters(), lr=lr)
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                self.w(X).squeeze(-1), y.float())
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def accuracy(self, X, y) -> float:
        with torch.no_grad():
            pred = (self.w(X).squeeze(-1) > 0).long()
        return (pred == y.long()).float().mean().item()


def probe_component(acts_train, y_train, acts_test, y_test, component: Component,
                    property_name: str, family: str, store) -> EvidenceRecord:
    probe = LinearProbe(acts_train.shape[-1]).fit(acts_train, y_train)
    acc = probe.accuracy(acts_test, y_test)
    rec = EvidenceRecord(
        components=[component], family=family,
        spec=InterventionSpec(method="probe", extra={"property": property_name}),
        values={"probe_accuracy": acc},
        notes="decodable != used; confirm with an intervention",
    )
    return store.add(rec)


# ------------------------------------------------- supervised (label bootstrap)


class ProfileClassifier:
    """Multi-label one-vs-rest logistic classifier over profile features,
    trained from reviewed entries. Below `unknown_threshold` max probability
    it answers ["unknown"]."""

    def __init__(self, unknown_threshold: float = 0.6):
        self.unknown_threshold = unknown_threshold
        self.keys: List[str] = []
        self.labels: List[str] = []
        self.heads: Dict[str, torch.nn.Linear] = {}
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, profiles: Sequence[ComponentProfile],
            labels: Dict[str, List[str]], steps=400, lr=5e-2, seed=0):
        torch.manual_seed(seed)
        X, self.keys = feature_matrix(profiles)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        self.labels = sorted({l for ls in labels.values() for l in ls})
        for label in self.labels:
            y = torch.tensor([1.0 if label in labels.get(p.component.name, [])
                              else 0.0 for p in profiles])
            head = torch.nn.Linear(Xt.shape[1], 1)
            opt = torch.optim.Adam(head.parameters(), lr=lr)
            for _ in range(steps):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    head(Xt).squeeze(-1), y)
                opt.zero_grad(); loss.backward(); opt.step()
            self.heads[label] = head
        return self

    def predict(self, profile: ComponentProfile) -> List[str]:
        x = np.array([profile.feature_vector_fields().get(k, 0.0)
                      for k in self.keys])
        xt = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)
        probs = {label: torch.sigmoid(head(xt)).item()
                 for label, head in self.heads.items()}
        chosen = [l for l, p in probs.items() if p >= self.unknown_threshold]
        return chosen if chosen else ["unknown"]
