# -*- coding: utf-8 -*-
"""
Reading and writing structured experiment outputs

Defines the on-disk layout under results/<experiment>/ and the JSON and JSONL
serialisation used by the trainer and the analysis tooling, so every consumer
agrees on a single schema
"""

import os
import json
import glob

from detection.paths import RESULTS_DIR, experiment_results_dir

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.jsonl"
QUALITATIVE_FILE = "qualitative.json"


def _write_json(path: str, payload) -> None:
    """
    Write a JSON document with indentation and a trailing newline
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _read_json(path: str):
    """
    Read and parse a JSON document from disk
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(name: str, config_dict: dict) -> str:
    """
    Persist the experiment configuration snapshot

    Args:
        name (str): Experiment registry name
        config_dict (dict): Serialised ExperimentConfig
    Returns:
        str: Path written
    """
    path = os.path.join(experiment_results_dir(name), CONFIG_FILE)
    _write_json(path, config_dict)
    return path


def save_metrics(name: str, metrics: dict) -> str:
    """
    Persist the full-train and ablation metrics for an experiment

    Args:
        name (str): Experiment registry name
        metrics (dict): Metrics document following the project schema
    Returns:
        str: Path written
    """
    path = os.path.join(experiment_results_dir(name), METRICS_FILE)
    _write_json(path, metrics)
    return path


def save_predictions(name: str, rows: list) -> str:
    """
    Persist per-example test predictions as JSON Lines

    Args:
        name (str): Experiment registry name
        rows (list): One dictionary per test example following the project schema
    Returns:
        str: Path written
    """
    path = os.path.join(experiment_results_dir(name), PREDICTIONS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def metrics_exists(name: str) -> bool:
    """
    Return whether a metrics document already exists for an experiment
    """
    return os.path.exists(os.path.join(RESULTS_DIR, name, METRICS_FILE))


def load_metrics(name: str) -> dict:
    """
    Load the metrics document for one experiment
    """
    return _read_json(os.path.join(RESULTS_DIR, name, METRICS_FILE))


def load_all_metrics() -> dict:
    """
    Load every available metrics document keyed by experiment name

    Returns:
        dict: Mapping of experiment name to its parsed metrics document
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", METRICS_FILE))):
        name = os.path.basename(os.path.dirname(path))
        out[name] = _read_json(path)
    return out


def best_experiment(metrics: dict = None, by: str = "macro_f1") -> str:
    """
    Determine the best experiment empirically from the stored results

    The winner is whichever experiment has the highest full-train test score on the
    chosen metric, so no experiment is privileged in code

    Args:
        metrics (dict): Pre-loaded metrics keyed by name, or None to load from disk
        by (str): Metric key inside full_train.test to rank by
    Returns:
        str: Name of the best experiment, or None when no results exist
    """
    if metrics is None:
        metrics = load_all_metrics()
    if not metrics:
        return None
    return max(metrics, key=lambda n: metrics[n]["full_train"]["test"].get(by, 0.0))


def load_predictions(name: str) -> list:
    """
    Load per-example predictions for one experiment from JSON Lines
    """
    path = os.path.join(RESULTS_DIR, name, PREDICTIONS_FILE)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_qualitative(name: str, payload: dict) -> str:
    """
    Persist the qualitative analysis document for an experiment

    Args:
        name (str): Experiment registry name
        payload (dict): Qualitative document with notes, buckets and per-class examples
    Returns:
        str: Path written
    """
    path = os.path.join(experiment_results_dir(name), QUALITATIVE_FILE)
    _write_json(path, payload)
    return path


def load_qualitative(name: str) -> dict:
    """
    Load the qualitative analysis document for one experiment
    """
    return _read_json(os.path.join(RESULTS_DIR, name, QUALITATIVE_FILE))
