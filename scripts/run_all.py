# -*- coding: utf-8 -*-
"""
Run a set of PubHealth experiments in a caller-specified order

The order is supplied on the command line rather than fixed in code, since which
configuration performs best is an empirical result and not known in advance. With
no --experiments argument the registry order is used. The loop is resumable:
experiments that already have results are skipped unless --force is given
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

from detection.config import EXPERIMENTS, EXPERIMENT_NAMES, DEVICE
from detection.data import load_pubhealth
from detection.results_io import metrics_exists
from detection.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run PubHealth experiments in a chosen order")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_NAMES, metavar="NAME",
                        default=EXPERIMENT_NAMES,
                        help="Experiment names in the order to run them (default: registry order)")
    parser.add_argument("--force", action="store_true",
                        help="Rerun experiments even if results already exist")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Run order: {args.experiments}")

    train_records, val_records, test_records = load_pubhealth()

    for name in args.experiments:
        if metrics_exists(name) and not args.force:
            print(f"\nSkipping {name}, results already exist (use --force to rerun)")
            continue
        run_experiment(EXPERIMENTS[name], train_records, val_records, test_records)

    print("\nAll requested experiments complete")


if __name__ == "__main__":
    main()
