# -*- coding: utf-8 -*-
"""
Figure generation from stored metrics

Renders the training curves, per-variant confusion matrices and per-class F1 bars,
and the cross-variant macro-F1 comparison for an experiment

All figures are derived from the metrics document so they can be regenerated without retraining
"""

import os

import numpy as np

from detection.config import NUM_LABELS, CLASS_NAMES
from detection.paths import experiment_figures_dir

_CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
_VARIANT_COLORS = ["#6baed6", "#fd8d3c", "#74c476"]
_VARIANT_LABELS = {
    "claim_only": "Claim\nOnly",
    "claim+explanation": "Claim +\nExplanation",
    "full_evidence": "Claim + Explanation\n+ Evidence",
}


def _new_axes(figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.subplots(figsize=figsize)


def _training_curves(history: dict, out_dir: str, loss_label: str, title: str) -> str:
    """
    Plot train and validation loss alongside the validation macro-F1 trajectory
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], marker="o", color="#4C72B0",
             linewidth=2, markersize=5, label="Train loss")
    ax1.plot(epochs, history["val_loss"], marker="s", color="#DD8452",
             linewidth=2, markersize=5, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(loss_label)
    ax1.set_title("Train vs Validation Loss", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, history["val_f1"], marker="o", color="#55A868", linewidth=2, markersize=5)
    best_f1 = max(history["val_f1"])
    best_ep = epochs[history["val_f1"].index(best_f1)]
    ax2.axvline(best_ep, linestyle="--", color="#C44E52", alpha=0.7,
                label=f"Best epoch {best_ep} (F1={best_f1:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation Macro-F1", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _confusion(cm: list, out_dir: str, variant: str) -> str:
    """
    Render a confusion matrix heatmap for one input variant
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    plt, (fig, ax) = _new_axes((6, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix\n{variant}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _per_class_f1(per_class: dict, macro: float, out_dir: str, variant: str) -> str:
    """
    Render a per-class F1 bar chart for one input variant
    """
    import matplotlib.ticker as mticker
    plt, (fig, ax) = _new_axes((6, 4))

    values = [per_class[CLASS_NAMES[i]]["f1"] for i in range(NUM_LABELS)]
    bars = ax.bar(CLASS_NAMES, values, color=_CLASS_COLORS,
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(f"Per-Class F1  |  Macro-F1 = {macro:.3f}\n{variant}",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, f"f1_perclass_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _macro_comparison(ablation: dict, out_dir: str) -> str:
    """
    Render the macro-F1 comparison across the three input variants
    """
    plt, (fig, ax) = _new_axes((8, 5))

    variants = ["claim_only", "claim+explanation", "full_evidence"]
    labels = [_VARIANT_LABELS[v] for v in variants]
    scores = [ablation[v]["test_macro_f1"] for v in variants]

    bars = ax.bar(labels, scores, color=_VARIANT_COLORS,
                  edgecolor="white", linewidth=0.8, zorder=3, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (Test)", fontsize=12)
    ax.set_title("Ablation Study - Macro-F1 by Input Configuration",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "macro_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_experiment_figures(name: str, metrics: dict, loss_label: str = "Loss") -> list:
    """
    Generate every figure for an experiment from its metrics document

    Args:
        name (str): Experiment registry name
        metrics (dict): The metrics document produced by the trainer
        loss_label (str): Y-axis label for the loss curve, for example Focal Loss
    Returns:
        list: Paths of the figures written under results/<name>/figures
    """
    out_dir = experiment_figures_dir(name)
    paths = []

    full = metrics["full_train"]
    paths.append(_training_curves(
        full["history"], out_dir, loss_label,
        title=f"{name} - Training Curves",
    ))

    test = full["test"]
    paths.append(_confusion(test["confusion_matrix"], out_dir, "full_evidence"))
    paths.append(_per_class_f1(test["per_class"], test["macro_f1"], out_dir, "full_evidence"))

    for variant, r in metrics["ablation"].items():
        paths.append(_confusion(r["confusion_matrix"], out_dir, variant))
        paths.append(_per_class_f1(r["per_class"], r["test_macro_f1"], out_dir, variant))

    paths.append(_macro_comparison(metrics["ablation"], out_dir))
    print(f"Saved {len(paths)} figures to {out_dir}")
    return paths
