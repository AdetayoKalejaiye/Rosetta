"""End-to-end Rosetta pipeline.

Order of operations (mirrors the design brief):

1.  Weight analysis                      -> structural candidates
2.  Activation & attention profiling     -> per-component profiles + examples
3.  Direct logit attribution             -> candidate output writers (readout,
                                            NOT removal effect)
4.  Attribution patching + EAP-IG        -> cheap screens, rank candidates
5.  Ablation battery (zero/mean/resample)-> verify screened candidates with
    + activation patching (both dirs)       real interventions; flag method
                                            disagreements on each entry
6.  KV-head ablations                    -> the GQA distinction: a shared KV
                                            head hits a whole query-head group
7.  Joint ablation + path patching       -> from "these heads matter" to a
                                            candidate circuit
8.  Detectors + clustering               -> unsupervised labels (no gold
                                            labels yet, per the brief)
9.  LLM / template explanations          -> the Rosetta Stone entries
10. Prediction testing on held-out fam   -> confidence from passed predictions
11. Circuit faithfulness                 -> behavior retention with everything
                                            outside the circuit replaced
12. Export: TensorBoard/json, inspection -> human-inspectable evidence
    HTML, map JSON + markdown
Optional SAE stage: train on residual activations, reconstruction gate,
one feature intervention only if the gate passes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from .model import ModelAdapter, RESID
from .schema import Component, Explanation
from .store import EvidenceStore
from .tasks import PromptFamily
from .methods.weights import run_weight_analysis
from .methods.profiling import run_profiling
from .methods.dla import run_dla
from .methods.attribution import attribution_patching, eap_ig, rank_candidates
from .methods.ablation import ablate, joint_ablation, run_ablation_battery
from .methods.patching import activation_patch, path_patch
from .methods.sae import feature_intervention, reconstruction_check, train_sae
from .analyze.classify import cluster_profiles, run_detectors
from .analyze.tb import Exporter, export_store, write_inspection_html
from .stone.explain import get_explainer
from .stone.mapfile import RosettaMap
from .stone.verify import circuit_faithfulness, run_prediction


def run_export(out_dir: str, cfg=None) -> Dict:
    """Re-run just the export step (step 12) on an already-computed run.

    Reads the existing store and rosetta map from *out_dir* and regenerates:
    - TensorBoard event file (stale events in tb/ are replaced)
    - summaries.json
    - inspection.html
    - rosetta_map.md

    `cfg` is the model config object used for layer×head heatmaps; omit it and
    the grid section is skipped (everything else still exports fine).
    """
    store = EvidenceStore(os.path.join(out_dir, "store"))

    map_path = os.path.join(out_dir, "rosetta_map.json")
    rosetta_map = RosettaMap(map_path)
    explanations: Dict[str, "Explanation"] = {
        name: entry.explanation for name, entry in rosetta_map.entries.items()
    }

    exporter = Exporter(os.path.join(out_dir, "tb"))
    export_store(store, exporter, cfg, explanations=explanations)
    exporter.flush()
    exporter.close()

    html_path = os.path.join(out_dir, "inspection.html")
    write_inspection_html(store, explanations, html_path,
                          related=_related_from_clusters(store))

    md = rosetta_map.render_markdown()
    md_path = os.path.join(out_dir, "rosetta_map.md")
    with open(md_path, "w") as f:
        f.write(md)

    return {
        "store_root": store.root,
        "inspection_html": html_path,
        "map_json": map_path,
        "map_markdown": md_path,
        "tensorboard_logdir": exporter.logdir,
    }


def _add(store: EvidenceStore, rec) -> None:
    store.add(rec)
    for c in rec.components:
        prof = store.profile(c.name)
        if rec.record_id not in prof.evidence_ids:
            prof.evidence_ids.append(rec.record_id)


def _related_from_clusters(store: EvidenceStore) -> Dict[str, List[str]]:
    """Components sharing a cluster are 'related' in the inspection view."""
    by_cluster: Dict[int, List[str]] = {}
    for name, prof in store.profiles.items():
        if prof.cluster is not None:
            by_cluster.setdefault(prof.cluster, []).append(name)
    related = {}
    for members in by_cluster.values():
        for name in members:
            related[name] = [m for m in members if m != name][:6]
    return related


def run_pipeline(adapter: ModelAdapter,
                 families: Dict[str, PromptFamily],
                 out_dir: str,
                 top_k: int = 6,
                 explainer_kind: str = "auto",
                 run_sae: bool = False,
                 seed: int = 0,
                 verbose: bool = True) -> Dict:
    """Run the whole workflow. `families` must contain 'train', 'heldout',
    and 'reference' (the reference family defines the mean/resample baseline
    distribution, recorded in every spec)."""

    def log(msg):
        if verbose:
            print(msg)

    os.makedirs(out_dir, exist_ok=True)
    store = EvidenceStore(os.path.join(out_dir, "store"))
    train, heldout, ref = families["train"], families["heldout"], families["reference"]
    cfg = adapter.cfg

    # 1. weights ------------------------------------------------------------
    log("[1/12] weight analysis")
    run_weight_analysis(adapter, store)

    # 2. profiling ----------------------------------------------------------
    log("[2/12] activation & attention profiling")
    run_profiling(adapter, train, store)

    # 3. DLA ----------------------------------------------------------------
    log("[3/12] direct logit attribution (readout contribution, not removal)")
    run_dla(adapter, train, store)

    # 4. attribution screens -------------------------------------------------
    log("[4/12] attribution patching + EAP-IG screens")
    attribution_patching(adapter, train, store)
    eap_ig(adapter, train, store, steps=5)
    candidates = rank_candidates(store, method="eap_ig", top_k=top_k)
    if not candidates:  # fall back to the plain linear screen
        candidates = rank_candidates(store, method="attribution_patching",
                                     top_k=top_k)
    log(f"        candidates: {[c.name for c in candidates]}")

    # 5. verify with real interventions --------------------------------------
    log("[5/12] ablation battery (zero/mean/resample) + activation patching")
    run_ablation_battery(adapter, train, candidates, ref, store, seed=seed)
    for comp in candidates:
        for direction in ("denoising", "noising"):
            _add(store, activation_patch(adapter, train, comp, direction))

    # 6. GQA contrast: shared KV heads ---------------------------------------
    log("[6/12] KV-head ablations (GQA: one shared KV head hits a group)")
    seen_kv = set()
    for comp in candidates:
        if comp.kind != "head":
            continue
        kv_idx = comp.index // cfg.group_size
        kv = Component("kv_head", comp.layer, kv_idx)
        if kv.name in seen_kv:
            continue
        seen_kv.add(kv.name)
        for side in ("v", "k"):
            _add(store, ablate(adapter, train, kv, "mean", ref_fam=ref,
                               kv_side=side, seed=seed))

    # 7. circuit-level: joint ablation + path patching ------------------------
    log("[7/12] joint ablation + path patching")
    heads = [c for c in candidates if c.kind == "head"]
    if len(heads) >= 2:
        _add(store, joint_ablation(adapter, train, heads, "mean",
                                   ref_fam=ref, seed=seed))
        ordered = sorted(heads, key=lambda c: (c.layer, c.index))
        n_pairs = 0
        for i, s in enumerate(ordered):
            for r in ordered[i + 1:]:
                if s.layer > r.layer or (s.layer, s.index) == (r.layer, r.index):
                    continue
                _add(store, path_patch(adapter, train, s, r))
                n_pairs += 1
                if n_pairs >= 6:
                    break
            if n_pairs >= 6:
                break

    # 8. detectors + clustering -----------------------------------------------
    log("[8/12] pattern detectors + clustering (no gold labels yet)")
    for prof in store.profiles.values():
        run_detectors(prof)
    head_profiles = [p for p in store.profiles.values()
                     if p.component.kind == "head"]
    if len(head_profiles) >= 4:
        cluster_profiles(head_profiles, k=min(4, len(head_profiles)), seed=seed)

    # 9. explanations ----------------------------------------------------------
    log("[9/12] Rosetta Stone explanations")
    explainer = get_explainer(explainer_kind)
    explanations: Dict[str, Explanation] = {}
    for comp in candidates:
        explanations[comp.name] = explainer.explain(store, comp.name,
                                                    heldout_family=heldout.name)

    rosetta_map = RosettaMap(os.path.join(out_dir, "rosetta_map.json"))
    for exp in explanations.values():
        rosetta_map.upsert(exp)

    # 10. test predictions on held-out prompts ---------------------------------
    log("[10/12] testing predictions on held-out prompts")
    fam_lookup = {f.name: f for f in families.values()}
    n_pass = n_fail = 0
    for name, exp in explanations.items():
        for pred in exp.predictions:
            if pred.family not in fam_lookup:
                continue
            done = run_prediction(adapter, pred, fam_lookup, ref, store)
            rosetta_map.record_outcome(name, done)
            if done.passed:
                n_pass += 1
            else:
                n_fail += 1
    log(f"        predictions: {n_pass} passed, {n_fail} failed")

    # 11. circuit faithfulness --------------------------------------------------
    log("[11/12] circuit faithfulness (mean baseline outside the circuit)")
    faithfulness = None
    circuit = [c for c in candidates if c.kind in ("head", "mlp")]
    if circuit:
        rec = circuit_faithfulness(adapter, circuit, heldout, ref, store,
                                   baseline="mean")
        faithfulness = {"components": [c.name for c in circuit],
                        "baseline": "mean",
                        "retention": rec.values["retention"]}
        log(f"        retention: {faithfulness['retention']:.2f}")

    # optional SAE stage ---------------------------------------------------------
    sae_summary: Optional[dict] = None
    if run_sae:
        log("[SAE ] training SAE on residual stream + reconstruction gate")
        layer = cfg.n_layers // 2
        key = (RESID, layer)
        _, cache = adapter.run(train.clean, capture=[key])
        acts = cache[key].reshape(-1, cfg.d_model)
        sae = train_sae(acts, d_features=cfg.d_model * 4, steps=300, seed=seed)
        ok, rec = reconstruction_check(adapter, sae, train, key, store)
        sae_summary = {"layer": layer, "gate_passed": ok,
                       "frac_var_unexplained": rec.values["frac_var_unexplained"]}
        if ok:
            with torch.no_grad():
                _, f = sae(acts)
            feat = int(f.mean(0).argmax().item())
            feature_intervention(adapter, sae, train, key, feat, store,
                                 check_passed=ok)
            sae_summary["feature_ablated"] = feat
        else:
            log("        gate FAILED - feature interventions not trusted")

    # 12. export ------------------------------------------------------------------
    log("[12/12] export: TensorBoard/json, inspection HTML, map")
    store.save_profiles()
    exporter = Exporter(os.path.join(out_dir, "tb"))
    export_store(store, exporter, cfg, explanations=explanations)
    exporter.flush()
    exporter.close()
    html_path = os.path.join(out_dir, "inspection.html")
    write_inspection_html(store, explanations, html_path,
                          related=_related_from_clusters(store))

    rosetta_map.save()
    md = rosetta_map.render_markdown(faithfulness=faithfulness)
    md_path = os.path.join(out_dir, "rosetta_map.md")
    with open(md_path, "w") as f:
        f.write(md)

    return {
        "store_root": store.root,
        "candidates": [c.name for c in candidates],
        "predictions": {"passed": n_pass, "failed": n_fail},
        "faithfulness": faithfulness,
        "sae": sae_summary,
        "inspection_html": html_path,
        "map_json": rosetta_map.path,
        "map_markdown": md_path,
        "tensorboard_logdir": exporter.logdir,
    }
