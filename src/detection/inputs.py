# -*- coding: utf-8 -*-
"""
Model input construction

Turns a raw PubHealth record into the (part_a, part_b) text pair fed to RoBERTa,
where part_a is the claim and part_b combines the explanation with the selected
evidence. The same machinery serves both the full model and the three ablation
input variants used to measure the contribution of each information source
"""

from detection.config import MAX_TOKENS
from detection.evidence import EvidenceRanker
from detection.tokenizer import get_tokenizer

_ranker = EvidenceRanker()

ABLATION_VARIANTS = ["claim_only", "claim+explanation", "full_evidence"]


def full_input(record: dict, cfg) -> tuple:
    """
    Build the full-evidence input pair and expose the retrieved evidence string

    The evidence string is returned alongside the pair so that evaluation and the
    qualitative analysis can record exactly which sentences the model saw

    Args:
        record (dict): A PubHealth record with claim, explanation and main_text
        cfg (ExperimentConfig): Supplies the evidence mode and optional top_k
    Returns:
        tuple: (part_a, part_b, evidence) where evidence may be an empty string
    """
    claim = (record.get("claim") or "").strip()
    explanation = (record.get("explanation") or "").strip()
    main_text = record.get("main_text") or ""

    evidence = _ranker.rank(
        claim, explanation, main_text, mode=cfg.evidence_mode, top_k=cfg.top_k
    )

    tok = get_tokenizer()
    sep = f"{tok.sep_token}{tok.sep_token}"     # RoBERTa marks the segment boundary with </s></s>
    part_b = f"{explanation} {sep} {evidence}" if evidence else explanation
    return claim, part_b, evidence


def ablation_input(record: dict, variant: str, cfg) -> tuple:
    """
    Build the (part_a, part_b) pair for one ablation variant

    Args:
        record (dict): A PubHealth record
        variant (str): One of claim_only, claim+explanation, full_evidence
        cfg (ExperimentConfig): Used when the variant needs evidence retrieval
    Returns:
        tuple: (part_a, part_b) text pair
    """
    claim = (record.get("claim") or "").strip()
    explanation = (record.get("explanation") or "").strip()

    if variant == "claim_only":
        return claim, ""
    if variant == "claim+explanation":
        return claim, explanation
    if variant == "full_evidence":
        part_a, part_b, _ = full_input(record, cfg)
        return part_a, part_b
    raise ValueError(f"Unknown ablation variant {variant}")


def encode_pair(part_a: str, part_b: str, truncation: str = "only_second",
                max_tokens: int = MAX_TOKENS):
    """
    Tokenise a text pair into padded fixed-length tensors for the model

    Args:
        part_a (str): First segment, never truncated when truncation is only_second
        part_b (str): Second segment, may be empty for the claim-only variant
        truncation (str): Either "only_second" to protect the claim or True
        max_tokens (int): Padding and truncation length
    Returns:
        dict: Tokenizer output with batch dimension squeezed away
    """
    tok = get_tokenizer()
    enc = tok(
        part_a,
        part_b if part_b else None,
        max_length=max_tokens,
        truncation=truncation,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
    }
