# -*- coding: utf-8 -*-
"""
Hyperparameters, label conventions and the experiment registry

A single ExperimentConfig dataclass parametrises every run, and EXPERIMENTS holds
the seven configurations in their neutral experimental order. The execution order
is not fixed here on purpose: which configuration performs best is an empirical
outcome, so callers choose the order at run time and the best model is determined
from the results rather than asserted in code
"""

from dataclasses import dataclass, asdict
from typing import Optional

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "roberta-base"
MAX_TOKENS = 512
BATCH_SIZE = 64
GRAD_ACCUM = 1                  # Effective batch size is BATCH_SIZE * GRAD_ACCUM = 64
EPOCHS = 10
EPOCHS_ABLATION = 5            # Ablation conditions train for fewer epochs to save time
LR = 4e-5                       # RoBERTa fine-tuning LR at effective batch 64
WARMUP_FRAC = 0.10
WEIGHT_DECAY = 0.01

NUM_LABELS = 4
LABEL_MAP = {0: "true", 1: "false", 2: "unproven", 3: "mixture"}
LABEL_MAP_STR = {"true": 0, "false": 1, "unproven": 2, "mixture": 3}
CLASS_NAMES = [LABEL_MAP[i] for i in range(NUM_LABELS)]

# Per-class focusing exponents for the focal-loss experiments, in class-index order
# [true, false, unproven, mixture]; the rarer classes are given the higher exponents
GAMMA_PER_CLASS = [1.0, 1.0, 3.0, 1.5]

# Oversampling multipliers keyed by class index, applied to the training split only
OVERSAMPLE_FACTORS = {2: 1.8, 3: 1.4}     # unproven x1.8, mixture x1.4


@dataclass
class ExperimentConfig:
    """
    Complete specification of one experiment

    Bundles the loss choice, class-weighting scheme, evidence-selection mode and
    optional oversampling so the trainer and ablation share a single code path
    The combination of fields reproduces each of the seven original scripts
    """

    name: str                                   # Registry key and results subdirectory
    description: str
    loss: str                                   # Either "ce" or "focal"
    use_class_weights: bool
    weight_temperature: Optional[float]         # None when the loss is unweighted
    evidence_mode: str                          # Either "dynamic" or "topk"
    top_k: Optional[int] = None                 # Sentence count for the top-k mode
    gamma_per_class: Optional[list] = None      # Per-class focusing exponents for focal loss
    oversample_factors: Optional[dict] = None   # Training-split oversampling, None to disable
    epochs: int = EPOCHS
    epochs_ablation: int = EPOCHS_ABLATION

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable view of the configuration
        """
        return asdict(self)


EXPERIMENTS = {
    "baseline": ExperimentConfig(
        name="baseline",
        description="Unweighted cross-entropy baseline with dynamic evidence",
        loss="ce",
        use_class_weights=False,
        weight_temperature=None,
        evidence_mode="dynamic",
    ),
    "weighted_k5": ExperimentConfig(
        name="weighted_k5",
        description="Inverse-frequency weighted cross-entropy with top-5 evidence",
        loss="ce",
        use_class_weights=True,
        weight_temperature=1.0,
        evidence_mode="topk",
        top_k=5,
    ),
    "weighted_k3": ExperimentConfig(
        name="weighted_k3",
        description="Inverse-frequency weighted cross-entropy with top-3 evidence",
        loss="ce",
        use_class_weights=True,
        weight_temperature=1.0,
        evidence_mode="topk",
        top_k=3,
    ),
    "weighted_k1": ExperimentConfig(
        name="weighted_k1",
        description="Inverse-frequency weighted cross-entropy with top-1 evidence",
        loss="ce",
        use_class_weights=True,
        weight_temperature=1.0,
        evidence_mode="topk",
        top_k=1,
    ),
    "weighted_dynamic": ExperimentConfig(
        name="weighted_dynamic",
        description="Inverse-frequency weighted cross-entropy with dynamic evidence",
        loss="ce",
        use_class_weights=True,
        weight_temperature=1.0,
        evidence_mode="dynamic",
    ),
    "focal_dynamic": ExperimentConfig(
        name="focal_dynamic",
        description="Per-class focal loss with softened weights and dynamic evidence",
        loss="focal",
        use_class_weights=True,
        weight_temperature=0.5,
        evidence_mode="dynamic",
        gamma_per_class=GAMMA_PER_CLASS,
    ),
    "focal_oversample": ExperimentConfig(
        name="focal_oversample",
        description="Per-class focal loss plus minority oversampling with dynamic evidence",
        loss="focal",
        use_class_weights=True,
        weight_temperature=0.5,
        evidence_mode="dynamic",
        gamma_per_class=GAMMA_PER_CLASS,
        oversample_factors=OVERSAMPLE_FACTORS,
    ),
}

# Neutral default order (the registry order); callers may reorder at run time
EXPERIMENT_NAMES = list(EXPERIMENTS)
