"""Prompt families and metrics.

A PromptFamily is a batch of matched clean/corrupt pairs with token-level
answers. All patching directions and ablation baselines are defined relative
to a family, and every record names the family + corruption method used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class PromptFamily:
    name: str
    description: str
    corruption: str                      # how corrupt prompts were constructed
    clean: torch.Tensor                  # [N, S] token ids
    corrupt: torch.Tensor                # [N, S]
    answer: torch.Tensor                 # [N] correct next-token id at final pos
    distractor: torch.Tensor             # [N] the corrupt-consistent answer
    texts: List[str] = field(default_factory=list)  # optional readable forms

    @property
    def n(self):
        return self.clean.shape[0]

    @property
    def seq_len(self):
        return self.clean.shape[1]

    def split(self, frac=0.5, seed=0):
        """Train/held-out split so explanations are tested on unseen pairs."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self.n, generator=g)
        k = max(1, int(self.n * frac))
        a_idx, b_idx = perm[:k], perm[k:]

        def take(idx, suffix):
            return PromptFamily(
                name=f"{self.name}-{suffix}", description=self.description,
                corruption=self.corruption,
                clean=self.clean[idx], corrupt=self.corrupt[idx],
                answer=self.answer[idx], distractor=self.distractor[idx],
                texts=[self.texts[i] for i in idx.tolist()] if self.texts else [],
            )
        return take(a_idx, "train"), take(b_idx, "heldout")


# --------------------------------------------------------------------- metrics


def logit_diff(logits: torch.Tensor, fam: PromptFamily) -> torch.Tensor:
    """Mean (answer - distractor) logit at the final position. Higher = model
    prefers the clean-consistent answer."""
    final = logits[:, -1, :]
    idx = torch.arange(fam.n)
    return (final[idx, fam.answer] - final[idx, fam.distractor]).mean()

def answer_logprob(logits: torch.Tensor, fam: PromptFamily) -> torch.Tensor:
    final = logits[:, -1, :].log_softmax(-1)
    return final[torch.arange(fam.n), fam.answer].mean()

def kl_from(reference_logits: torch.Tensor):
    ref = reference_logits[:, -1, :].log_softmax(-1).detach()

    def _kl(logits: torch.Tensor, fam: PromptFamily) -> torch.Tensor:
        cur = logits[:, -1, :].log_softmax(-1)
        return torch.nn.functional.kl_div(cur, ref, log_target=True,
                                          reduction="batchmean")
    return _kl

METRICS = {"logit_diff": logit_diff, "answer_logprob": answer_logprob}


def get_metric(name: str):
    return METRICS[name]


# ---------------------------------------------------------------- synthetic task


def build_induction_family(vocab_size: int, n: int = 32, seq_len: int = 24,
                           seed: int = 0, name: str = "induction") -> PromptFamily:
    """Repeated-sequence induction task.

    clean   : [x1..xk, x1..x_{k-1}]                -> answer = x_k
    corrupt : same, but the FIRST occurrence of x_k is swapped for x_k', so an
              induction mechanism now points at x_k' -> distractor = x_k'.

    This is the toy stand-in for "prompt families"; on a real model you would
    add IOI-style, docstring, greater-than, etc. families in the same format.
    """
    g = torch.Generator().manual_seed(seed)
    half = seq_len // 2
    first = torch.randint(2, vocab_size, (n, half), generator=g)
    # make the answer token unique in the prefix so induction is unambiguous
    clean_full = torch.cat([first, first], dim=1)
    clean = clean_full[:, :-1]
    answer = clean_full[:, -1].clone()          # = first[:, -1]

    corrupt = clean.clone()
    distractor = torch.empty(n, dtype=torch.long)
    for i in range(n):
        while True:
            alt = torch.randint(2, vocab_size, (1,), generator=g).item()
            if alt != answer[i].item():
                break
        corrupt[i, half - 1] = alt              # swap first occurrence of x_k
        distractor[i] = alt
    return PromptFamily(
        name=name,
        description="predict the repeat of a random sequence (induction)",
        corruption="swap the first occurrence of the answer token for a random other token",
        clean=clean, corrupt=corrupt, answer=answer, distractor=distractor,
    )


def build_task_suite(vocab_size: int, seed: int = 0) -> Dict[str, PromptFamily]:
    """train family, held-out family (same distribution, new samples), and a
    reference family (for mean/resample baselines, per the brief: a separate
    relevant prompt set with corresponding token positions)."""
    fam = build_induction_family(vocab_size, n=48, seed=seed)
    train, heldout = fam.split(frac=0.5, seed=seed)
    reference = build_induction_family(vocab_size, n=32, seed=seed + 1,
                                       name="induction-reference")
    return {"train": train, "heldout": heldout, "reference": reference}
