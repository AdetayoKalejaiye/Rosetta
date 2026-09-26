"""End-to-end smoke test on a tiny toy model. Run with:
    python tests/test_smoke.py        (or pytest tests/)
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from rosetta.model import ModelConfig, ToyAdapter, ToyGQATransformer, train_toy_on_induction
from rosetta.pipeline import run_pipeline
from rosetta.tasks import build_task_suite


def test_end_to_end():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2,
                      n_heads=4, n_kv_heads=2)
    model = ToyGQATransformer(cfg)
    train_toy_on_induction(model, steps=150, batch=16, seq_len=16, verbose=False)

    families = build_task_suite(cfg.vocab_size, seed=0)
    with tempfile.TemporaryDirectory() as out:
        result = run_pipeline(ToyAdapter(model), families, out_dir=out,
                              top_k=4, explainer_kind="template",
                              run_sae=True, verbose=False)

        # artifacts exist
        for k in ("inspection_html", "map_json", "map_markdown"):
            assert os.path.exists(result[k]), f"missing {k}"
        assert os.path.exists(os.path.join(result["store_root"], "evidence.jsonl"))
        assert os.path.exists(os.path.join(result["store_root"], "profiles.json"))
        assert os.path.exists(os.path.join(result["tensorboard_logdir"],
                                           "summaries.json"))

        # candidates found and predictions actually ran
        assert result["candidates"], "no candidates surfaced"
        total = result["predictions"]["passed"] + result["predictions"]["failed"]
        assert total > 0, "no predictions were tested"

        # confidences in [0,1]
        with open(result["map_json"]) as f:
            entries = json.load(f)
        assert entries
        for e in entries:
            assert 0.0 <= e["confidence"] <= 1.0

        # every evidence record carries its methodological choices
        with open(os.path.join(result["store_root"], "evidence.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                assert "spec" in rec and "method" in rec["spec"]

    print("smoke test passed")


if __name__ == "__main__":
    test_end_to_end()
