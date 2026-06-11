# -*- coding: utf-8 -*-
"""
Class weighting for imbalanced training

Computes temperature-scaled inverse-frequency weights from a training split. A
temperature of 1.0 reproduces raw inverse frequency; a lower value applies a
gentler scaling (0.5 is a square root) that raises the minority-class weights less
aggressively relative to the majority classes
"""

import numpy as np
import torch

from detection.config import NUM_LABELS, LABEL_MAP, DEVICE


def compute_class_weights(records: list, temperature: float = 1.0,
                          device: torch.device = DEVICE) -> torch.Tensor:
    """
    Compute normalised inverse-frequency class weights for a training split

    IMPORTANT: always pass the original, non-oversampled split so the weights
    reflect the true corpus frequencies rather than the resampled distribution

    Args:
        records (list): Training records, each with an integer label field
        temperature (float): Scaling exponent applied to the class counts
        device (torch.device): Device the resulting tensor is moved to
    Returns:
        torch.Tensor: Weights of shape (num_classes,) summing to num_classes
    """
    counts = np.zeros(NUM_LABELS, dtype=np.float32)
    for r in records:
        counts[r["label"]] += 1

    weights = 1.0 / (counts ** temperature + 1e-6)
    weights = weights / weights.sum() * NUM_LABELS
    print(f"Class weights (temp={temperature}): "
          f"{ {LABEL_MAP[i]: round(float(weights[i]), 3) for i in range(NUM_LABELS)} }")
    return torch.tensor(weights, dtype=torch.float32).to(device)
