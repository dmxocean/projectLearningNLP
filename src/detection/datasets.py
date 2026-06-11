# -*- coding: utf-8 -*-
"""
Torch datasets for the full model and the ablation variants

Each example carries its original split index so that predictions produced during
evaluation can be joined back to the raw claim, explanation and evidence text,
which the qualitative analysis depends on
"""

import torch
from torch.utils.data import Dataset

from detection.inputs import full_input, ablation_input, encode_pair


class PubHealthDataset(Dataset):
    """
    Tokenised full-evidence dataset used for the main training and evaluation run

    Evidence retrieval happens here during construction, once per record, according
    to the experiment configuration
    """

    def __init__(self, records: list, cfg):
        self.samples = []
        for i, r in enumerate(records):
            part_a, part_b, _ = full_input(r, cfg)
            enc = encode_pair(part_a, part_b, truncation="only_second")
            self.samples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": torch.tensor(r["label"], dtype=torch.long),
                "orig_idx": i,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AblationDataset(Dataset):
    """
    Tokenised dataset for a single ablation input variant

    The claim is never truncated for the full-evidence variant via the only_second
    rule inside encode_pair; the other variants are short enough that plain
    truncation suffices
    """

    def __init__(self, records: list, variant: str, cfg):
        self.variant = variant
        self.samples = []
        truncation = "only_second" if variant == "full_evidence" else True
        for i, r in enumerate(records):
            part_a, part_b = ablation_input(r, variant, cfg)
            enc = encode_pair(part_a, part_b, truncation=truncation)
            self.samples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": torch.tensor(r["label"], dtype=torch.long),
                "orig_idx": i,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
