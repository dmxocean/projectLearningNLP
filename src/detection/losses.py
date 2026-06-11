# -*- coding: utf-8 -*-
"""
Loss functions and criterion selection

Provides a per-class focal loss and a factory that returns the criterion matching
an experiment configuration: plain cross-entropy, inverse-frequency weighted
cross-entropy, or focal loss with per-class focusing exponents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Per-class focal loss for multi-class classification

    Each class carries its own focusing exponent, so the down-weighting of easy
    (high-probability) examples can be set per class: a higher gamma concentrates
    the loss more strongly on a class's low-probability cases, and a gamma of zero
    recovers weighted cross-entropy for that class. The loss is
    FL_i(p_t) = -w_i * (1 - p_t)^gamma_i * log(p_t)

    Args:
        weight (torch.Tensor): Per-class weights of shape (num_classes,), or None
        gamma_per_class (list): Per-class focusing exponents in class-index order
    """

    def __init__(self, weight, gamma_per_class):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("gamma", torch.tensor(gamma_per_class, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)                          # Recover the probability of the true class
        gammas = self.gamma.to(targets.device)[targets]
        loss = ((1.0 - pt) ** gammas) * ce
        return loss.mean()


def make_criterion(cfg, class_weights):
    """
    Build the training criterion for an experiment configuration

    Args:
        cfg (ExperimentConfig): Selects focal versus cross-entropy and gammas
        class_weights (torch.Tensor): Per-class weights, or None for unweighted
    Returns:
        nn.Module: The loss module ready to apply to logits and integer targets
    """
    if cfg.loss == "focal":
        return FocalLoss(weight=class_weights, gamma_per_class=cfg.gamma_per_class)
    if cfg.loss == "ce":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss {cfg.loss}")
