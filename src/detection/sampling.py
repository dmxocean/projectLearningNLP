# -*- coding: utf-8 -*-
"""
Minority-class oversampling

Duplicates minority training records by a per-class multiplier to give rare
classes more presence in each epoch

The fractional part of a multiplier is realised probabilistically so the 
expected count is exact on average
"""

import random

from detection.config import LABEL_MAP, NUM_LABELS


def oversample_minority(records: list, oversample_factors: dict) -> list:
    """
    Return a shuffled training split with minority records duplicated

    A factor of 1.8 yields one guaranteed extra copy plus a second extra copy with
    probability 0.8 per record, so the class grows by about 1.8x on average

    IMPORTANT: apply this to the training split only, never to validation or test,
    and compute class weights from the original split rather than this output

    Args:
        records (list): Original training records with integer label fields
        oversample_factors (dict): Mapping of class index to multiplier
    Returns:
        list: The original records plus probabilistic duplicates, shuffled
    """
    extras = []
    for r in records:
        factor = oversample_factors.get(r["label"], 1.0)
        whole = int(factor) - 1                # Guaranteed extra copies
        frac = factor - int(factor)            # Probabilistic extra copy
        extras.extend([r] * whole)
        if random.random() < frac:
            extras.append(r)

    combined = records + extras
    random.shuffle(combined)

    counts = {LABEL_MAP[i]: 0 for i in range(NUM_LABELS)}
    for r in combined:
        counts[LABEL_MAP[r["label"]]] += 1
    print(f"Oversampling: {len(records)} -> {len(combined)} samples  |  counts {counts}")
    return combined
