# -*- coding: utf-8 -*-
"""
Cross-experiment qualitative comparison

Lays out every experiment's prediction on one fixed set of test examples, joined to
the source text, so the approaches can be read side by side. No notes or conclusions
are produced; interpretation is left to the reader. Works only from the fields in the
saved prediction rows, so it needs neither the model nor the configuration
"""

import re

MINORITY_LABELS = ("unproven", "mixture")

# Default sizes of the shared example set, tunable by the caller
DIVERGENT_TOTAL = 24                                    # Examples where experiments disagree
CONSENSUS_TOTAL = 6                                     # Examples where every experiment fails

# Per-true-class quota for the divergent set, weighted towards the minority classes;
# unfilled or overflowing quotas spill over to the highest-disagreement examples
DIVERGENT_CLASS_QUOTA = {"unproven": 8, "mixture": 8, "false": 4, "true": 4}


def _common_indices(preds_by_exp: dict, names: list) -> list:
    """
    Return the test indices present in every compared experiment, sorted
    """
    common = None
    for name in names:
        keys = set(preds_by_exp[name].keys())
        common = keys if common is None else (common & keys)
    return sorted(common or [])


def _per_example_stats(preds_by_exp: dict, names: list, idx) -> dict:
    """
    Summarise one example: true label, distinct prediction count and wrong count
    """
    rows = [preds_by_exp[name][idx] for name in names]
    y_true = rows[0]["y_true_label"]
    preds = [r["y_pred_label"] for r in rows]
    return {
        "y_true": y_true,
        "n_distinct": len(set(preds)),
        "n_wrong": sum(1 for p in preds if p != y_true),
        "n_exp": len(rows),
    }


def select_comparison_indices(preds_by_exp: dict, names: list, *,
                              divergent_total: int = DIVERGENT_TOTAL,
                              consensus_total: int = CONSENSUS_TOTAL,
                              class_quota: dict = None) -> list:
    """
    Choose the fixed example set: divergent cases (minority first) plus consensus failures

    The selection is deterministic so the set is stable across runs

    Args:
        preds_by_exp (dict): Mapping name -> {idx: prediction row}
        names (list): Experiments to compare
        divergent_total (int): Target number of divergent examples
        consensus_total (int): Target number of consensus-failure examples
        class_quota (dict): Per-true-class quota for the divergent set
    Returns:
        list: Pairs of (idx, category) where category is divergent or consensus_failure
    """
    class_quota = dict(class_quota if class_quota is not None else DIVERGENT_CLASS_QUOTA)
    common = _common_indices(preds_by_exp, names)
    info = {idx: _per_example_stats(preds_by_exp, names, idx) for idx in common}

    divergent = [i for i in common if info[i]["n_distinct"] > 1]
    consensus = [i for i in common
                 if info[i]["n_distinct"] == 1 and info[i]["n_wrong"] == info[i]["n_exp"]]

    # Most informative first: widest disagreement, then most wrong, then index for stability
    def div_key(i):
        return (-info[i]["n_distinct"], -info[i]["n_wrong"], i)

    chosen, seen = [], set()

    # 1) Minority-weighted per-class quota over the divergent examples
    by_class = {}
    for i in divergent:
        by_class.setdefault(info[i]["y_true"], []).append(i)
    for bucket in by_class.values():
        bucket.sort(key=div_key)
    for cls, quota in class_quota.items():
        for i in by_class.get(cls, [])[:quota]:
            if i not in seen:
                chosen.append((i, "divergent"))
                seen.add(i)

    # 2) Fill any remaining divergent slots with the highest-disagreement leftovers
    for i in sorted(divergent, key=div_key):
        if len(chosen) >= divergent_total:
            break
        if i not in seen:
            chosen.append((i, "divergent"))
            seen.add(i)

    # 3) Consensus failures, minority true classes first, index order within a group
    cons_minority = sorted(i for i in consensus if info[i]["y_true"] in MINORITY_LABELS)
    cons_rest = sorted(i for i in consensus if info[i]["y_true"] not in MINORITY_LABELS)
    n_consensus = 0
    for i in cons_minority + cons_rest:
        if n_consensus >= consensus_total:
            break
        if i not in seen:
            chosen.append((i, "consensus_failure"))
            seen.add(i)
            n_consensus += 1

    return chosen


def build_comparison(names: list, preds_by_exp: dict, selection: list,
                     macro_f1: dict = None) -> dict:
    """
    Assemble the shared examples with every experiment's prediction on each

    Args:
        names (list): Experiments in display order (e.g. by descending macro-F1)
        preds_by_exp (dict): Mapping name -> {idx: prediction row}
        selection (list): Pairs of (idx, category) from select_comparison_indices
        macro_f1 (dict): Optional name -> full-model test macro-F1, for reference
    Returns:
        dict: experiment order, selection counts and the per-example records
    """
    macro_f1 = macro_f1 or {}
    examples = []
    for idx, category in selection:
        ref = preds_by_exp[names[0]][idx]                 # Claim/explanation are shared across runs
        per_exp = [{
            "experiment": name,
            "pred_label": preds_by_exp[name][idx]["y_pred_label"],
            "confidence": preds_by_exp[name][idx]["confidence"],
            "probs": preds_by_exp[name][idx].get("probs"),
            "correct": preds_by_exp[name][idx]["correct"],
            "evidence": preds_by_exp[name][idx].get("evidence", ""),
        } for name in names]
        examples.append({
            "idx": idx,
            "category": category,
            "true_label": ref["y_true_label"],
            "claim": ref.get("claim", ""),
            "explanation": ref.get("explanation", ""),
            "predictions": per_exp,
        })

    return {
        "experiments": [{"name": n, "full_macro_f1": macro_f1.get(n)} for n in names],
        "n_examples": len(examples),
        "selection": {
            "divergent": sum(1 for _, c in selection if c == "divergent"),
            "consensus_failure": sum(1 for _, c in selection if c == "consensus_failure"),
        },
        "examples": examples,
    }


def _flatten(text: str) -> str:
    """
    Collapse whitespace and neutralise pipes and truncation marks for one plain line
    """
    text = (text or "").replace("|", "/").replace("[...]", "(truncated)")
    return re.sub(r"\s+", " ", text).strip()


def _evidence_groups(predictions: list) -> list:
    """
    Group experiments that received identical (flattened) evidence so it is shown once
    """
    groups = []
    for p in predictions:
        evidence = _flatten(p.get("evidence", ""))
        for group in groups:
            if group[0] == evidence:
                group[1].append(p["experiment"])
                break
        else:
            groups.append([evidence, [p["experiment"]]])
    return groups


def render_text(comparison: dict) -> str:
    """
    Render the comparison as plain ASCII text, grouped per example, no interpretation

    Args:
        comparison (dict): Output of build_comparison
    Returns:
        str: Plain-text document
    """
    L = [
        "--- Cross-experiment qualitative comparison ---",
        "",
        "Same test examples shown for every experiment. Read the text and",
        "each prediction; all interpretation is left to the reader.",
        "",
        "Experiments (display order, full-model test macro-F1 for reference):",
    ]
    width = max((len(e["name"]) for e in comparison["experiments"]), default=0)
    for e in comparison["experiments"]:
        f1 = e["full_macro_f1"]
        f1s = f"{f1:.4f}" if isinstance(f1, (int, float)) else "n/a"
        L.append(f"    {e['name']:<{width}}   macro-F1={f1s}")
    s = comparison["selection"]
    L += [
        "",
        f"{comparison['n_examples']} shared examples: "
        f"{s['divergent']} where experiments disagree, "
        f"{s['consensus_failure']} where every experiment fails.",
    ]

    for ex in comparison["examples"]:
        ew = max((len(p["experiment"]) for p in ex["predictions"]), default=0)
        L += [
            "",
            "",
            f"--- idx {ex['idx']}  (true: {ex['true_label']}, {ex['category']}) ---",
            "",
            "    Claim:",
            f"        {_flatten(ex['claim'])}",
        ]
        if ex["explanation"]:
            L += ["", "    Explanation:", f"        {_flatten(ex['explanation'])}"]
        L += ["", "    Predictions:"]
        for p in ex["predictions"]:
            mark = "yes" if p["correct"] else "no"
            L.append(f"        {p['experiment']:<{ew}}   pred={p['pred_label']:<9} "
                     f"conf={p['confidence']:.4f}   correct={mark}")
        L += ["", "    Evidence received:"]
        for evidence, who in _evidence_groups(ex["predictions"]):
            L.append(f"        {', '.join(who)}:")
            L.append(f"            {evidence if evidence else '(none)'}")

    return "\n".join(L) + "\n"
