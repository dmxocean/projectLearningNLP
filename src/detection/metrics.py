# -*- coding: utf-8 -*-
"""
Evaluation metrics

Wraps scikit-learn to produce the macro-F1 headline figure plus a fully
JSON-serialisable report containing per-class precision, recall, F1 and support,
overall accuracy, and the confusion matrix
"""

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from detection.config import NUM_LABELS, LABEL_MAP


def macro_f1(y_true, y_pred) -> float:
    """
    Return the macro-averaged F1 over the four classes
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def report_dict(y_true, y_pred) -> dict:
    """
    Build a JSON-serialisable metrics report for one set of predictions

    Args:
        y_true (list): Ground-truth integer labels
        y_pred (list): Predicted integer labels
    Returns:
        dict: macro_f1, accuracy, per-class precision/recall/f1/support and the
            confusion matrix as nested lists in fixed class-index order
    """
    labels = list(range(NUM_LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        LABEL_MAP[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
