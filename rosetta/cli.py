"""Rosetta command line.

  python -m rosetta.cli demo --out runs/demo
      Train the built-in toy GQA transformer on the induction task and run the
      full pipeline against it. Fully self-contained; runs on CPU in seconds.

  python -m rosetta.cli run --model HuggingFaceTB/SmolLM2-360M --out runs/smol
      Point the same pipeline at a HuggingFace Llama-family checkpoint through
      HFAdapter. Present per the design, but NOT exercised in this build (the
      brief: do not test with the actual model). Expect to iterate the first
      time it touches real weights.
"""

from __future__ import annotations

import argparse
import json


def _demo(args) -> int:
    import torch
    from .model import ModelConfig, ToyAdapter, ToyGQATransformer, train_toy_on_induction
    from .pipeline import run_pipeline
    from .tasks import build_task_suite

    torch.manual_seed(args.seed)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=3,
                      n_heads=6, n_kv_heads=2)  # group_size=3: real GQA sharing
    model = ToyGQATransformer(cfg)
    print(f"training toy GQA model ({cfg.n_layers}L, {cfg.n_heads}Q/{cfg.n_kv_heads}KV heads) "
          "on the induction task...")
    train_toy_on_induction(model, steps=args.steps, seq_len=24,
                           seed=args.seed, verbose=True)

    families = build_task_suite(cfg.vocab_size, seed=args.seed)
    result = run_pipeline(ToyAdapter(model), families, out_dir=args.out,
                          top_k=args.top_k, explainer_kind=args.explainer,
                          run_sae=args.sae, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


def _run_hf(args) -> int:
    from .model import HFAdapter
    from .pipeline import run_pipeline
    from .tasks import build_task_suite

    print("NOTE: the HF path is wired but untested in this build (per the "
          "design constraint of not touching the real model).")
    adapter = HFAdapter.from_pretrained(args.model)
    families = build_task_suite(adapter.cfg.vocab_size, seed=args.seed)
    result = run_pipeline(adapter, families, out_dir=args.out,
                          top_k=args.top_k, explainer_kind=args.explainer,
                          run_sae=args.sae, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rosetta", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="toy GQA model, full pipeline, CPU")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--steps", type=int, default=400, help="toy training steps")
    d.add_argument("--top-k", type=int, default=6)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--explainer", default="auto",
                   choices=["auto", "template", "llm"])
    d.add_argument("--sae", action="store_true", help="include the SAE stage")
    d.set_defaults(fn=_demo)

    r = sub.add_parser("run", help="HuggingFace checkpoint (untested path)")
    r.add_argument("--model", required=True,
                   help="e.g. HuggingFaceTB/SmolLM2-360M (32L, 15Q/5KV heads)")
    r.add_argument("--out", default="runs/model")
    r.add_argument("--top-k", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--explainer", default="auto",
                   choices=["auto", "template", "llm"])
    r.add_argument("--sae", action="store_true")
    r.set_defaults(fn=_run_hf)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
