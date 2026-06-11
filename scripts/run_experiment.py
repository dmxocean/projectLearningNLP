# -*- coding: utf-8 -*-
"""
Run a single experiment by name and persist its outputs

Loads PubHealth, then trains, evaluates and saves one experiment from the registry
under results/<experiment>
Skips work that already exists unless --force is given
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

from detection.config import EXPERIMENTS, DEVICE
from detection.data import load_pubhealth
from detection.results_io import metrics_exists
from detection.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run one PubHealth experiment")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS),
                        help="Registry name of the experiment to run")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if results already exist")
    args = parser.parse_args()

    if metrics_exists(args.experiment) and not args.force:
        print(f"{args.experiment} already has results, pass --force to rerun")
        return

    print(f"Device: {DEVICE}")
    train_records, val_records, test_records = load_pubhealth()
    run_experiment(EXPERIMENTS[args.experiment], train_records, val_records, test_records)


if __name__ == "__main__":
    main()
