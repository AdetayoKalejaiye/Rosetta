# Rosetta

A mechanistic interpretability toolkit. Rosetta runs several discovery methods
over a transformer, collects their evidence in one format, lets you (and a
classifier) inspect it, has an LLM propose explanations — the "Rosetta Stone" —
and then closes the loop by testing those explanations against held-out
prompts and new interventions.

The central artifact is an evolving map (`rosetta_map.json` / `.md`) whose
entries carry confidence earned from **passed predictions**, not vibes.

## Status

- Fully built and exercised end-to-end against a built-in **toy GQA
  transformer** trained on an induction task (CPU, seconds).
- The HuggingFace path (`HFAdapter`, `rosetta run --model ...`) is written to
  the same interface but **intentionally untested** — this build was done
  under the constraint of not touching the real model. Expect to iterate the
  first time it meets real weights.

## Quick start

```bash
pip install torch numpy            # tensorboard + transformers optional
python examples/run_demo.py        # or: python -m rosetta.cli demo --out runs/demo
python tests/test_smoke.py
```

The demo trains a 3-layer toy model (6 query heads sharing 2 KV heads, so
grouped-query attention is real, group size 3), runs the full pipeline, and
writes to `runs/demo/`:

| artifact | what it is |
|---|---|
| `store/evidence.jsonl` | append-only evidence records, one per intervention/measurement |
| `store/profiles.json` | per-component profiles (weights, attention, activations, examples, labels, disagreements) |
| `inspection.html` | dependency-free inspection view: click a component → stats, attention scores, highlighted text examples, intervention results, related components, explanation + prediction outcomes |
| `tb/` | TensorBoard logs (if `tensorboard` installed) + always `summaries.json` |
| `rosetta_map.json` / `.md` | the Rosetta Stone: explanations, predictions with PASS/FAIL, confidence, circuit faithfulness |

Later, to point it at a real checkpoint (untested path):

```bash
pip install transformers
python -m rosetta.cli run --model HuggingFaceTB/SmolLM2-360M --out runs/smol
```

SmolLM2-360M: 32 layers, 15 query heads, 5 KV heads per layer → group size 3.

## Discovery methods (`rosetta/methods/`)

| method | file | what it gives you |
|---|---|---|
| Weight analysis | `weights.py` | norms + singular-value summaries of W_QK / W_OV per head (KV-side stats marked as shared), MLP structure, Q-composition scores between layer pairs |
| Activation & attention profiling | `profiling.py` | per-head scores for prev-token, induction, same-token, sink, self-attention, entropy; activation stats; top text examples with the activating position |
| Zero / mean / resample ablation | `ablation.py` | all three baselines per component; the reference family and mean scope are recorded in the spec; **disagreements between baselines are flagged on the component's entry** |
| Activation patching | `patching.py` | both directions — clean→corrupted (noising) and corrupted→clean (denoising) — reported as recovery of the clean–corrupt gap |
| Direct logit attribution | `dla.py` | which components write toward the answer at the readout — recorded explicitly as a **readout contribution, not a removal effect** |
| Attribution patching / EAP-IG | `attribution.py` | fast gradient screens over every head + MLP; used only to **rank candidates**, which are then verified with actual interventions |
| Path patching + joint ablation | `patching.py` / `ablation.py` | sender→receiver path effects (with MLPs frozen or free, recorded), and joint ablation of component sets — the step from "these heads matter" to a circuit |
| SAE (later addition) | `sae.py` | trains on residual activations; **feature interventions are refused unless the reconstruction gate passes** (fraction of variance unexplained and downstream metric shift both under threshold) |

Every `EvidenceRecord` carries an `InterventionSpec` — method, baseline,
reference family, mean scope, direction, corruption, metric, seed — because
patching results change with those choices, so they live next to every number.

### The GQA distinction

Ablating one head's **output** (`HEAD_OUT`) and ablating a **shared KV head**
(`KV_K`/`KV_V`) are different interventions and Rosetta keeps them distinct:
a KV-head record names the whole query-head group it touches. In the demo this
is visible directly — single induction heads cost ~0.1–0.2 logit-diff under
mean ablation, while ablating the one shared V-representation they all read
costs ~1.6.

## Classifier & inspection (`rosetta/analyze/`)

The classifier receives the **underlying numerical measurements** (flattened
profile features), not screenshots:

- explicit pattern detectors (prev-token, induction, sink, same-token,
  direct-logit-writer, ...) surface candidates first — no labels needed;
- k-means clustering groups similar components ("related components" in the
  inspection view);
- `ProfileClassifier` is the bootstrap path: once entries have been reviewed
  and experimentally supported, train one-vs-rest heads with multiple labels
  and an explicit **"unknown"** outcome;
- `LinearProbe` is kept separate evidence, annotated *decodable ≠ used*.

`analyze/tb.py` logs everything to TensorBoard (when installed) and always to
`summaries.json`, including layer×head grids for induction/prev-token/DLA. It
also writes `inspection.html`, a static no-dependency view: component list →
weight/attention/activation stats, text examples with the activating token
highlighted, every intervention with its full spec, disagreement warnings,
related components, and the explanation with failed predictions kept visible.

## The Rosetta Stone (`rosetta/stone/`)

- `explain.py` builds a structured evidence packet (including negative
  results) and produces an `Explanation`: candidate function, relevant
  contexts, supporting evidence, contradictions, predicted intervention
  effects, next experiment. Two writers: `TemplateExplainer` (rule-based,
  every claim traceable to a record) and `APIExplainer` (Anthropic messages
  endpoint, used automatically when `ANTHROPIC_API_KEY` is set; falls back to
  the template on any failure). This stage is where Delphi-style
  explanation scoring plugs in.
- `verify.py` executes each prediction on the **held-out** family and marks it
  PASSED/FAILED; `circuit_faithfulness` measures behavior retention when every
  contribution *outside* the circuit is replaced under a specified baseline.
- `mapfile.py` keeps the map: prediction history survives explanation
  revisions, and confidence is the Laplace-smoothed pass rate
  `(passes+1)/(passes+fails+2)` — entries earn confidence by predicting.

## Pipeline

`rosetta/pipeline.py:run_pipeline` chains all twelve stages (docstring has the
order). `rosetta/cli.py` exposes `demo` and `run`. `tasks.py` builds prompt
families (clean/corrupt pairs with answer + distractor tokens) split into
train / held-out, plus a **separate reference family** for mean/resample
baselines, and defines metrics (`logit_diff`, `answer_logprob`, KL).

## Demo results worth knowing about

- The toy model is trained *only* on induction, so the induction detector
  fires broadly across heads (the pattern really is everywhere). That is the
  designed division of labor: detectors over-generate candidates, and the
  ablation/patching/prediction stages arbitrate — the map's confidence comes
  from interventions, not detector hits.
- Baseline disagreement flagging works: the demo's L2.MLP entry carries
  `zero=-1.83, mean=-2.14, resample=-5.02` as an explicit caveat.

## Layout

```
rosetta/
  schema.py        Component / InterventionSpec / EvidenceRecord /
                   ComponentProfile / Prediction / Explanation
  store.py         append-only evidence store + profiles + disagreement notes
  model.py         hook points (HEAD_OUT, KV_K, KV_V, MLP_OUT, RESID),
                   ToyGQATransformer, ToyAdapter, HFAdapter (untested)
  tasks.py         prompt families, corruption, metrics, train/heldout/reference
  methods/         weights, profiling, ablation, patching, dla, attribution, sae
  analyze/         detectors, clustering, probes, classifier, TB export,
                   inspection HTML
  stone/           explanations, prediction verification, faithfulness, the map
  pipeline.py      end-to-end orchestration
  cli.py           `rosetta demo`, `rosetta run --model ...`
examples/run_demo.py
tests/test_smoke.py
```
