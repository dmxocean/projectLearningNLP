# -*- coding: utf-8 -*-
"""
Per-example prediction records

Joins raw predictions to the original claim, explanation and retrieved evidence to
produce one record per test example. These rows are the raw material saved as
predictions.jsonl and later read by the cross-experiment comparison; no analysis or
observations are computed here
"""

from detection.config import LABEL_MAP
from detection.inputs import full_input

EXPLANATION_LIMIT = 800
EVIDENCE_LIMIT = 1200


def _truncate(text: str, limit: int) -> str:
    """
    Shorten text to a character limit, marking where it was cut
    """
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def build_prediction_rows(cfg, records: list, eval_out: dict) -> list:
    """
    Assemble per-example prediction records joined to their source text

    Evidence is recomputed per record so the stored row reflects exactly what the
    model received under this experiment configuration

    Args:
        cfg (ExperimentConfig): Supplies the evidence mode for recomputation
        records (list): The evaluated split, indexed by eval_out indices
        eval_out (dict): Output of the trainer prediction pass
    Returns:
        list: One row per example with text, labels, probabilities and confidence
    """
    rows = []
    for position, orig_idx in enumerate(eval_out["indices"]):
        record = records[orig_idx]
        _, _, evidence = full_input(record, cfg)
        probs = [round(float(p), 4) for p in eval_out["probs"][position]]
        y_true = int(eval_out["y_true"][position])
        y_pred = int(eval_out["y_pred"][position])
        rows.append({
            "idx": int(orig_idx),
            "claim": record.get("claim", ""),
            "explanation": _truncate(record.get("explanation", ""), EXPLANATION_LIMIT),
            "evidence": _truncate(evidence, EVIDENCE_LIMIT),
            "y_true": y_true,
            "y_true_label": LABEL_MAP[y_true],
            "y_pred": y_pred,
            "y_pred_label": LABEL_MAP[y_pred],
            "probs": probs,
            "confidence": round(float(max(probs)), 4),
            "correct": y_true == y_pred,
        })
    return rows
