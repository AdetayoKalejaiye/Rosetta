# Rosetta

A mechanistic interpretability toolkit. Rosetta runs several discovery methods
over a transformer, collects their evidence in one format, lets you (and a
classifier) inspect it, has an LLM propose explanations — the "Rosetta Stone" —
and then closes the loop by testing those explanations against held-out
prompts and new interventions.

The central artifact is an evolving map (`rosetta_map.json` / `.md`) whose
entries carry confidence earned from **passed predictions**, not vibes.

There are two ways in: a **CLI** that runs the full pipeline and writes a run
directory, and a **web UI** (`rosetta serve`) that browses finished runs and
lets you interrogate them in natural language.

## Status

- The pipeline runs end-to-end against the built-in **toy GQA transformer**
  (CPU, seconds) and against real **HuggingFace Llama-family checkpoints**.
  SmolLM2-360M is the exercised real-model path: 32 layers, 15 query heads, 5
  KV heads, ~12.8k evidence records.
- The web UI reads finished runs, visualises them, and can run new
  interventions against the run's own checkpoint.
- The SAE stage is the least exercised part; its feature interventions are
  gated behind a reconstruction check and refuse to run if it fails.

## Quick start

```bash
pip install torch numpy                  # core
pip install fastapi "uvicorn[standard]"  # web UI
pip install transformers                 # real checkpoints
pip install tensorboard                  # optional; summaries.json always written

python -m rosetta.cli demo --out runs/demo   # toy model, full pipeline
python -m rosetta.cli serve                  # browse runs at localhost:8742
python tests/test_smoke.py
```

The demo trains a 3-layer toy model (6 query heads sharing 2 KV heads, so
grouped-query attention is real, group size 3), runs the full pipeline, and
writes to `runs/demo/`:

| artifact | what it is |
|---|---|
| `store/evidence.jsonl` | append-only evidence records, one per intervention/measurement |
| `store/profiles.json` | per-component profiles (weights, attention, activations, examples, labels, disagreements) |
| `run.json` | which model produced the run, so it can be reloaded later |
| `inspection.html` | dependency-free static inspection view |
| `tb/` | TensorBoard logs (if `tensorboard` installed) + always `summaries.json` |
| `rosetta_map.json` / `.md` | the Rosetta Stone: explanations, predictions with PASS/FAIL, confidence, circuit faithfulness |

Against a real checkpoint:

```bash
python -m rosetta.cli run --model HuggingFaceTB/SmolLM2-360M --out runs/smol
```

SmolLM2-360M: 32 layers, 15 query heads, 5 KV heads per layer → group size 3.

## Web UI (`rosetta/web/`)

```bash
python -m rosetta.cli serve                    # http://localhost:8742
python -m rosetta.cli serve --host 0.0.0.0     # reachable from other devices
```

Reading a finished run needs **no checkpoint and no torch** — the evidence
store is plain JSON, so the UI opens fully populated. A model is loaded only
when you ask for a new intervention.

- **Run picker** — every directory under `runs/` holding a `store/profiles.json`.
- **Layer × head grid** — colour any of the recorded metrics across the whole
  model; sequential scale for magnitudes, diverging for signed ones. Hover for
  per-head values, click to inspect.
- **Component panel** — detector labels, the proposed function and its
  PASSED/FAILED predictions, every metric with its **percentile among peers**,
  top activating examples with the peak token highlighted, and all evidence
  records with their full method specs.
- **Attention pattern** — token × token heatmap for any head, so you can see
  *where* it looks rather than only a summary score.
- **Ask** — a tool-using agent that queries the run's evidence, screens all
  components, builds new prompt families, and runs interventions. Components
  it names are highlighted in the grid and clickable.
- **Table view** — the same data as sortable rows.

### Asking semantic questions

The built-in prompt families are **random token sequences**. That is the right
probe for positional mechanisms (induction is copying, not meaning), but it
means those prompts can say nothing about semantics.

To ask a semantic question the agent writes its own prompts
(`create_prompt_family`): minimal pairs differing in one word, where clean and
corrupt tokenize to the same length and the answer/distractor are single
tokens. Pairs failing those constraints come back with reasons rather than
being silently dropped, and the family is rejected unless the model actually
separates clean from corrupt.

Worked example — *"find the head responsible for colour knowledge"* on
SmolLM2-360M: the agent built 15 pairs ("Fresh grass in the field looks" →
` green` vs ` white`), confirmed separation, screened all 480 heads in one
gradient pass, then verified the leaders. Result: **L27.H5** is the strongest
contributor (mean ablation −0.261, ~16% of baseline; DLA z = 3.07, top ~1%),
and **L26.H2** has a *positive* ablation delta — it suppresses the correct
colour rather than supporting it.

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
| SAE | `sae.py` | trains on residual activations; **feature interventions are refused unless the reconstruction gate passes** |

Every `EvidenceRecord` carries an `InterventionSpec` — method, baseline,
reference family, mean scope, direction, corruption, metric, seed — because
patching results change with those choices, so they live next to every number.

Screen before you brute-force: a gradient screen ranks all ~480 heads of
SmolLM2-360M in one pass, where patching them one at a time is ~10s each.

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
- k-means clustering groups similar components ("related components");
- `ProfileClassifier` is the bootstrap path: once entries have been reviewed
  and experimentally supported, train one-vs-rest heads with multiple labels
  and an explicit **"unknown"** outcome;
- `LinearProbe` is kept separate evidence, annotated *decodable ≠ used*.

**Detector thresholds are absolute and do not transfer between models.** On
SmolLM2-360M `attention_sink` fires on 72% of heads — sinks are near-universal
in trained LLMs, so the label carries almost no information there, while
`induction_head` (2%) and `prev_token_head` (3%) stay selective. The web UI
therefore shows each label's coverage and marks the uninformative ones, and
reports every metric as a **percentile and z-score among peers** alongside its
raw value. Worth internalising: `L24.H11` is at the 98.5th percentile for
induction (z = 5.06) but the 47.6th for sink score — it is tagged
`attention_sink` while sitting at the median for it.

`analyze/tb.py` logs to TensorBoard (when installed) and always to
`summaries.json`, including layer×head grids, and writes the static
`inspection.html`.

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
order) and writes a `run.json` manifest recording which model produced the run.
Runs made before manifests existed are matched against known architectures and
shown as *inferred* until a checkpoint loads and its shape is verified.

`tasks.py` builds prompt families (clean/corrupt pairs with answer +
distractor tokens) split into train / held-out, plus a **separate reference
family** for mean/resample baselines, and defines metrics (`logit_diff`,
`answer_logprob`, KL). `build_text_family` builds one from real sentences.

## Results worth knowing about

- The toy model is trained *only* on induction, so the induction detector
  fires broadly across heads (the pattern really is everywhere). That is the
  designed division of labor: detectors over-generate candidates, and the
  ablation/patching/prediction stages arbitrate — the map's confidence comes
  from interventions, not detector hits.
- Baseline disagreement flagging works: the demo's L2.MLP entry carries
  `zero=-1.83, mean=-2.14, resample=-5.02` as an explicit caveat, and on
  SmolLM2 L31.MLP carries `zero=-1.06, mean=-3.01, resample=-3.82`.
- On SmolLM2-360M induction concentrates in layers 17–24, with the strongest
  heads at L24.H10, L18.H11 and L17.H8 (induction 0.37–0.44).
- Ablating the induction head L24.H11 moves the colour task by +0.0007 —
  essentially nothing. Negative results like this are the point: the head is
  real, and it is not doing that job.

## Layout

```
rosetta/
  schema.py        Component / InterventionSpec / EvidenceRecord /
                   ComponentProfile / Prediction / Explanation
  store.py         append-only evidence store + profiles + disagreement notes
  model.py         hook points (HEAD_OUT, KV_K, KV_V, MLP_OUT, RESID),
                   ToyGQATransformer, ToyAdapter, HFAdapter
  tasks.py         prompt families (synthetic + real-text), metrics,
                   train/heldout/reference
  methods/         weights, profiling, ablation, patching, dla, attribution, sae
  analyze/         detectors, clustering, probes, classifier, TB export,
                   inspection HTML
  stone/           explanations, prediction verification, faithfulness, the map
  web/             server.py (FastAPI + WebSocket), agent.py (tool-using
                   analyst), runs.py (run discovery, manifests, population
                   stats), static/ (grid, component panel, chat)
  pipeline.py      end-to-end orchestration
  cli.py           `demo`, `run --model ...`, `export`, `serve`
examples/run_demo.py
tests/test_smoke.py
```
