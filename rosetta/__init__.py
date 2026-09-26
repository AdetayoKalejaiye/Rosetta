"""Rosetta: a mechanistic interpretability toolkit.

Pipeline: discovery methods -> shared evidence store -> detectors/clustering ->
TensorBoard + inspection view -> LLM "Rosetta Stone" explanations -> tested
predictions -> an evolving, confidence-weighted map.
"""

from .schema import (
    Component,
    ComponentProfile,
    EvidenceRecord,
    Explanation,
    InterventionSpec,
    Prediction,
)
from .store import EvidenceStore
from .model import (
    HEAD_OUT,
    KV_K,
    KV_V,
    MLP_OUT,
    RESID,
    ModelAdapter,
    ModelConfig,
    ToyAdapter,
    ToyGQATransformer,
    train_toy_on_induction,
)
from .tasks import PromptFamily, build_induction_family, build_task_suite, get_metric
from .stone.mapfile import RosettaMap

__version__ = "0.1.0"

__all__ = [
    "Component",
    "ComponentProfile",
    "EvidenceRecord",
    "Explanation",
    "InterventionSpec",
    "Prediction",
    "EvidenceStore",
    "HEAD_OUT",
    "KV_K",
    "KV_V",
    "MLP_OUT",
    "RESID",
    "ModelAdapter",
    "ModelConfig",
    "ToyAdapter",
    "ToyGQATransformer",
    "train_toy_on_induction",
    "PromptFamily",
    "build_induction_family",
    "build_task_suite",
    "get_metric",
    "RosettaMap",
    "__version__",
]
