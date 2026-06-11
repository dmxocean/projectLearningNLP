# -*- coding: utf-8 -*-
"""
Build the cross-experiment qualitative comparison

Reads results/<experiment>/predictions.jsonl for every experiment that has them and
writes one fixed set of test examples with each experiment's prediction, so the
approaches can be read side by side

Writes results/comparison/qualitative_comparison.{json,txt}
"""

import os
import sys
import json
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

from detection.config import EXPERIMENT_NAMES
from detection.results_io import load_all_metrics, load_predictions, PREDICTIONS_FILE
from detection.paths import RESULTS_DIR, COMPARISON_DIR
from detection.comparison import select_comparison_indices, build_comparison, render_text


def _present_experiments(requested: list) -> list:
    """
    Keep only the requested experiments that already have a predictions file
    """
    return [n for n in requested
            if os.path.exists(os.path.join(RESULTS_DIR, n, PREDICTIONS_FILE))]


def main():
    parser = argparse.ArgumentParser(description="Cross-experiment qualitative comparison")
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENT_NAMES, metavar="NAME",
                        help="Experiments to compare (default: all that have predictions)")
    parser.add_argument("--divergent", type=int, default=None, help="Number of divergent examples")
    parser.add_argument("--consensus", type=int, default=None, help="Number of consensus-failure examples")
    args = parser.parse_args()

    present = _present_experiments(args.experiments)
    if len(present) < 2:
        raise SystemExit(f"Need at least 2 experiments with {PREDICTIONS_FILE}; "
                         f"found {present or 'none'}. Run the sweep first.")

    metrics = load_all_metrics()
    macro_f1 = {n: metrics[n]["full_train"]["test"]["macro_f1"]
                for n in present if n in metrics}
    names = sorted(present, key=lambda n: macro_f1.get(n, -1.0), reverse=True)   # By descending macro-F1

    preds_by_exp = {n: {int(r["idx"]): r for r in load_predictions(n)} for n in present}

    kwargs = {}
    if args.divergent is not None:
        kwargs["divergent_total"] = args.divergent
    if args.consensus is not None:
        kwargs["consensus_total"] = args.consensus
    selection = select_comparison_indices(preds_by_exp, names, **kwargs)
    comparison = build_comparison(names, preds_by_exp, selection, macro_f1=macro_f1)

    os.makedirs(COMPARISON_DIR, exist_ok=True)
    json_path = os.path.join(COMPARISON_DIR, "qualitative_comparison.json")
    text_path = os.path.join(COMPARISON_DIR, "qualitative_comparison.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(render_text(comparison))

    print(f"Compared {len(names)} experiments (by descending macro-F1): {names}")
    print(f"  {comparison['n_examples']} shared examples")
    print(f"  wrote {os.path.relpath(json_path, BASE_PATH)}")
    print(f"  wrote {os.path.relpath(text_path, BASE_PATH)}")


if __name__ == "__main__":
    main()
