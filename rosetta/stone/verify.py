"""The feedback loop: Rosetta tests explanations against new prompts and
interventions. Predictions run on HELD-OUT families; outcomes flow back into
the map's confidence scores.

Also implements circuit faithfulness: for a proposed circuit, replace every
contribution OUTSIDE the circuit under a specified baseline and measure how
much of the behavior survives.
"""
from __future__ import annotations

from typing import Dict, List

from ..methods.ablation import ablate, joint_ablation
from ..methods.patching import activation_patch
from ..model import ModelAdapter
from ..schema import Component, EvidenceRecord, InterventionSpec, Prediction
from ..tasks import PromptFamily, get_metric


def run_prediction(adapter: ModelAdapter, pred: Prediction,
                   families: Dict[str, PromptFamily],
                   ref_fam: PromptFamily, store) -> Prediction:
    fam = families[pred.family]
    comps = [Component.parse(n) for n in pred.components]

    if pred.method in ("zero_ablation", "mean_ablation", "resample_ablation"):
        baseline = pred.method.split("_")[0]
        if len(comps) == 1:
            rec = ablate(adapter, fam, comps[0], baseline, ref_fam=ref_fam,
                         metric_name=pred.metric)
        else:
            rec = joint_ablation(adapter, fam, comps, baseline, ref_fam=ref_fam,
                                 metric_name=pred.metric)
        measured = rec.values.get("delta_logit_diff",
                                  rec.values.get(f"delta_{pred.metric}", 0.0))
    elif pred.method == "activation_patching":
        rec = activation_patch(adapter, fam, comps[0], direction="denoising",
                               metric_name=pred.metric)
        measured = rec.values["metric_patched"] - rec.values["metric_corrupt"]
    else:
        raise ValueError(f"unsupported prediction method {pred.method}")

    rec.notes = (rec.notes + " | prediction test").strip(" |")
    store.add(rec)

    pred.measured = measured
    if pred.expected_direction == "decrease":
        pred.passed = measured <= -pred.min_abs_effect
    elif pred.expected_direction == "increase":
        pred.passed = measured >= pred.min_abs_effect
    else:  # no_change
        pred.passed = abs(measured) <= pred.tolerance
    return pred


def circuit_faithfulness(adapter: ModelAdapter, circuit: List[Component],
                         fam: PromptFamily, ref_fam: PromptFamily, store,
                         baseline: str = "mean",
                         metric_name: str = "logit_diff") -> EvidenceRecord:
    """Retention = (m_circuit_only - m_everything_ablated) /
                   (m_full_model    - m_everything_ablated)

    where m_circuit_only ablates every head+MLP NOT in the circuit under
    `baseline`. 1.0 = the circuit alone reproduces the behavior on this
    metric; 0.0 = it does no better than ablating everything.
    """
    metric = get_metric(metric_name)
    cfg = adapter.cfg
    everything: List[Component] = []
    for l in range(cfg.n_layers):
        everything.extend(Component("head", l, h) for h in range(cfg.n_heads))
        everything.append(Component("mlp", l))
    in_circuit = {c.name for c in circuit}
    complement = [c for c in everything if c.name not in in_circuit]

    logits, _ = adapter.run(fam.clean)
    m_full = metric(logits, fam).item()
    rec_none = joint_ablation(adapter, fam, everything, baseline, ref_fam,
                              metric_name)
    m_none = rec_none.values["metric_ablated"]
    rec_circ = joint_ablation(adapter, fam, complement, baseline, ref_fam,
                              metric_name)
    m_circ = rec_circ.values["metric_ablated"]

    denom = m_full - m_none
    retention = (m_circ - m_none) / denom if abs(denom) > 1e-9 else 0.0
    rec = EvidenceRecord(
        components=circuit, family=fam.name,
        spec=InterventionSpec(method="faithfulness", baseline=baseline,
                              reference_family=ref_fam.name, metric=metric_name,
                              extra={"circuit_size": len(circuit),
                                     "ablated_outside": len(complement)}),
        values={"metric_full": m_full, "metric_circuit_only": m_circ,
                "metric_all_ablated": m_none, "retention": retention},
        notes="behavior surviving when contributions outside the circuit are "
              f"replaced under the {baseline} baseline",
    )
    return store.add(rec)
