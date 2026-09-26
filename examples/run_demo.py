"""Minimal end-to-end demo on the built-in toy GQA transformer.

Equivalent to `python -m rosetta.cli demo --out runs/demo`, kept as a script
so the pieces are visible.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from rosetta.model import ModelConfig, ToyAdapter, ToyGQATransformer, train_toy_on_induction
from rosetta.pipeline import run_pipeline
from rosetta.tasks import build_task_suite


def main():
    torch.manual_seed(0)

    # group_size = 6 // 2 = 3: each KV head serves three query heads, so the
    # KV-head vs head-output ablation distinction is actually exercised.
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=3,
                      n_heads=6, n_kv_heads=2)
    model = ToyGQATransformer(cfg)

    print("training toy model on the induction task (CPU, ~seconds)...")
    train_toy_on_induction(model, steps=400, seq_len=24, verbose=True)

    families = build_task_suite(cfg.vocab_size, seed=0)
    result = run_pipeline(ToyAdapter(model), families, out_dir="runs/demo",
                          top_k=6, explainer_kind="template", run_sae=True)

    print("\n== pipeline summary ==")
    print(json.dumps(result, indent=2))
    print(f"\nopen {result['inspection_html']} in a browser for the inspection view")
    print(f"read {result['map_markdown']} for the Rosetta Stone")


if __name__ == "__main__":
    main()
