# -*- coding: utf-8 -*-
"""
Canonical filesystem paths for the detection package

All project directories are resolved absolutely from this file's location so that
scripts, notebooks and tests reach the same locations regardless of the working
directory they are launched from
"""

import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_PATH, "src")
DATA_DIR = os.path.join(BASE_PATH, "data", "pubhealth")
RESULTS_DIR = os.path.join(BASE_PATH, "results")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")
MODELS_DIR = os.path.join(BASE_PATH, "models")
LOGS_DIR = os.path.join(BASE_PATH, "logs")


def experiment_results_dir(name: str) -> str:
    """
    Return the results directory for a single experiment, creating it if absent

    Args:
        name (str): Registry name of the experiment, used as the subdirectory
    Returns:
        str: Absolute path to results/<name>
    """
    path = os.path.join(RESULTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def experiment_figures_dir(name: str) -> str:
    """
    Return the figures directory for a single experiment, creating it if absent

    Args:
        name (str): Registry name of the experiment
    Returns:
        str: Absolute path to results/<name>/figures
    """
    path = os.path.join(RESULTS_DIR, name, "figures")
    os.makedirs(path, exist_ok=True)
    return path


def experiment_model_dir(name: str) -> str:
    """
    Return the checkpoint directory for a single experiment, creating it if absent

    Args:
        name (str): Registry name of the experiment
    Returns:
        str: Absolute path to models/<name>
    """
    path = os.path.join(MODELS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def experiment_log_path(name: str) -> str:
    """
    Return the log file path for a single experiment, creating the logs directory

    Args:
        name (str): Registry name of the experiment
    Returns:
        str: Absolute path to logs/<name>.log
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    return os.path.join(LOGS_DIR, f"{name}.log")
