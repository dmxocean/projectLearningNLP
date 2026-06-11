# -*- coding: utf-8 -*-
"""
End-to-end orchestration for a single experiment

Trains the full model, evaluates it on the test split, runs the three-variant
ablation, then writes the configuration, metrics, per-example predictions and
figures under results/<experiment>

Shared by the run_experiment and run_all scripts so the persistence schema lives in one place
"""

from detection.trainer import train_full, evaluate_full, run_ablation
from detection.metrics import report_dict
from detection.qualitative import build_prediction_rows
from detection.results_io import save_config, save_metrics, save_predictions
from detection.plotting import save_experiment_figures
from detection.logging_utils import tee_output
from detection.paths import experiment_log_path


def run_experiment(cfg, train_records: list, val_records: list, test_records: list) -> dict:
    """
    Run one experiment to completion and persist every output

    The whole run is mirrored to its own logs/<experiment>.log so each experiment
    keeps a self-contained record independent of how it was launched

    Args:
        cfg (ExperimentConfig): Experiment specification
        train_records (list): Training split
        val_records (list): Validation split
        test_records (list): Test split
    Returns:
        dict: The metrics document that was written to disk
    """
    with tee_output(experiment_log_path(cfg.name)):
        print(f"\n--- {cfg.name} ---\n")
        summary = train_full(cfg, train_records, val_records)

        test_out = evaluate_full(cfg, test_records, model=summary["model"])
        test_report = report_dict(test_out["y_true"], test_out["y_pred"])

        ablation = run_ablation(cfg, train_records, val_records, test_records)

        metrics = {
            "experiment": cfg.name,
            "description": cfg.description,
            "full_train": {
                "history": summary["history"],
                "best_val_f1": summary["best_val_f1"],
                "best_epoch": summary["best_epoch"],
                "test": test_report,
            },
            "ablation": ablation,
        }

        save_config(cfg.name, cfg.to_dict())
        save_metrics(cfg.name, metrics)
        save_predictions(cfg.name, build_prediction_rows(cfg, test_records, test_out))

        loss_label = "Focal Loss" if cfg.loss == "focal" else "Cross-Entropy Loss"
        save_experiment_figures(cfg.name, metrics, loss_label=loss_label)

        print(f"\n[{cfg.name}] full-evidence test macro-F1 = {test_report['macro_f1']:.4f}\n")
    return metrics
