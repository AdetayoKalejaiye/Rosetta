"""Rosetta command line.

  python -m rosetta.cli demo --out runs/demo
      Train the built-in toy GQA transformer on the induction task and run the
      full pipeline against it. Fully self-contained; runs on CPU in seconds.

  python -m rosetta.cli run --model HuggingFaceTB/SmolLM2-360M --out runs/smol
      Point the same pipeline at a HuggingFace Llama-family checkpoint through
      HFAdapter.

  python -m rosetta.cli serve
      Browse finished runs in the web UI and ask questions about them.
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


def _export(args) -> int:
    from .pipeline import run_export
    result = run_export(args.out)
    print(json.dumps(result, indent=2))
    return 0


def _serve(args) -> int:
    import uvicorn
    from .web.runs import discover_runs
    from .web.server import create_app

    found = discover_runs(args.runs)
    if found:
        print(f"runs in {args.runs}/: " +
              ", ".join(f"{r['id']} ({r['n_layers']}L x {r['n_heads']}H, "
                        f"{r['n_evidence']} records)" for r in found))
    else:
        print(f"no runs found in {args.runs}/ — generate one with: "
              "python -m rosetta.cli demo")
    app = create_app(runs_root=args.runs)
    print(f"Rosetta: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _run_hf(args) -> int:
    from .model import HFAdapter
    from .pipeline import run_pipeline
    from .tasks import build_task_suite

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

    e = sub.add_parser("export", help="re-run export step on an existing run (no model needed)")
    e.add_argument("--out", required=True, help="run directory, e.g. runs/smol")
    e.set_defaults(fn=_export)

    r = sub.add_parser("run", help="run the pipeline on a HuggingFace checkpoint")
    r.add_argument("--model", required=True,
                   help="e.g. HuggingFaceTB/SmolLM2-360M (32L, 15Q/5KV heads)")
    r.add_argument("--out", default="runs/model")
    r.add_argument("--top-k", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--explainer", default="auto",
                   choices=["auto", "template", "llm"])
    r.add_argument("--sae", action="store_true")
    r.set_defaults(fn=_run_hf)

    s = sub.add_parser("serve", help="browse runs in the web UI")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8742)
    s.add_argument("--runs", default="runs", help="directory holding runs")
    s.set_defaults(fn=_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
