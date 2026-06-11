# -*- coding: utf-8 -*-
"""
Generate qualitative.json for the best experiment

Reads predictions.jsonl, computes summary statistics, error buckets and
per-class examples, then writes qualitative.json into the experiment's results
directory so the analysis notebook can load it without re-running training
"""

import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from detection.results_io import load_all_metrics, load_predictions, best_experiment, save_qualitative
from detection.config import CLASS_NAMES

MINORITY_CLASSES = ["unproven", "mixture"]
N_CONFIDENT_ERRORS = 20   # High-confidence wrong predictions to retain
N_MINORITY_ERRORS = 20    # Minority-class error rows to retain
N_PER_CLASS = 10          # Examples per class bucket


def _notes(rows):
    """
    Compute summary statistics over all test predictions

    Args:
        rows (list): Prediction rows loaded from predictions.jsonl
    Returns:
        dict: Accuracy, mean confidence by correctness, per-class recall and top confusion pairs
    """
    correct = [r for r in rows if r["correct"]]
    incorrect = [r for r in rows if not r["correct"]]

    class_totals = Counter(r["y_true_label"] for r in rows)
    class_correct = Counter(r["y_true_label"] for r in correct)
    per_class_recall = {
        cls: round(class_correct[cls] / class_totals[cls], 4) if class_totals[cls] else 0.0
        for cls in CLASS_NAMES
    }

    conf_counter = Counter(
        (r["y_true_label"], r["y_pred_label"]) for r in incorrect  # Tally each wrong pair once
    )
    top_pairs = [
        {"true": t, "pred": p, "count": c}
        for (t, p), c in conf_counter.most_common(10)
    ]

    return {
        "n_test": len(rows),
        "accuracy": round(len(correct) / len(rows), 4),
        "mean_confidence_correct": round(sum(r["confidence"] for r in correct) / len(correct), 4) if correct else 0.0,
        "mean_confidence_incorrect": round(sum(r["confidence"] for r in incorrect) / len(incorrect), 4) if incorrect else 0.0,
        "per_class_recall": per_class_recall,
        "top_confusion_pairs": top_pairs,
    }


def _buckets(rows):
    """
    Build named error buckets for qualitative inspection

    Args:
        rows (list): Prediction rows loaded from predictions.jsonl
    Returns:
        dict: confident_errors and minority_errors buckets sorted by confidence descending
    """
    errors = [r for r in rows if not r["correct"]]

    confident_errors = sorted(
        errors, key=lambda r: r["confidence"], reverse=True
    )[:N_CONFIDENT_ERRORS]

    minority_errors = sorted(
        [r for r in errors if r["y_true_label"] in MINORITY_CLASSES],
        key=lambda r: r["confidence"],
        reverse=True,
    )[:N_MINORITY_ERRORS]

    return {
        "confident_errors": confident_errors,
        "minority_errors": minority_errors,
    }


def _per_class_examples(rows):
    """
    Collect correct and incorrect examples for each minority class

    Args:
        rows (list): Prediction rows loaded from predictions.jsonl
    Returns:
        dict: Per-class dict with correct and incorrect example lists
    """
    out = {}
    for cls in MINORITY_CLASSES:
        cls_rows = [r for r in rows if r["y_true_label"] == cls]
        out[cls] = {
            "correct": [r for r in cls_rows if r["correct"]][:N_PER_CLASS],
            "incorrect": sorted(
                [r for r in cls_rows if not r["correct"]],
                key=lambda r: r["confidence"],
                reverse=True,
            )[:N_PER_CLASS],
        }
    return out


def make_qualitative(name):
    """
    Build and persist the qualitative analysis document for one experiment

    Args:
        name (str): Experiment registry name matching a results subdirectory
    """
    rows = load_predictions(name)

    payload = {
        "experiment": name,
        "notes": _notes(rows),
        "buckets": _buckets(rows),
        "per_class_examples": _per_class_examples(rows),
    }

    path = save_qualitative(name, payload)
    print(f"Wrote {path}")


if __name__ == "__main__":
    all_metrics = load_all_metrics()
    name = best_experiment(all_metrics)
    print(f"Best experiment: {name}")
    make_qualitative(name)
