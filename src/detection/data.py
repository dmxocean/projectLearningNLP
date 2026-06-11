# -*- coding: utf-8 -*-
"""
PubHealth dataset loading

Reads the three tab-separated splits prepared by scripts/prepare_data.py and
returns plain lists of dictionaries, which is all the downstream Dataset wrappers
and training loops require, so the heavyweight datasets dependency is not needed
at training time
"""

import os

from detection.config import LABEL_MAP_STR, NUM_LABELS, LABEL_MAP
from detection.paths import DATA_DIR

REQUIRED_COLUMNS = ["claim", "explanation", "main_text", "label"]


def read_split(path: str) -> list:
    """
    Read one PubHealth TSV split into a list of record dictionaries

    String labels are normalised and mapped to the integer convention defined in
    config (true=0, false=1, unproven=2, mixture=3); rows with an unknown or empty
    label are dropped, and the optional text columns are filled with empty strings

    Args:
        path (str): Absolute path to a train, dev or test TSV file
    Returns:
        list: Records with keys claim, explanation, main_text and integer label
    """
    import pandas as pd

    df = pd.read_csv(path, sep="\t", on_bad_lines="skip", encoding="utf-8")
    df = df.dropna(subset=["claim", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP_STR)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    for col in ["explanation", "main_text"]:
        if col not in df.columns:
            df[col] = ""                       # Tolerate a missing optional column
        df[col] = df[col].fillna("").astype(str)

    df["claim"] = df["claim"].astype(str)
    return df[REQUIRED_COLUMNS].reset_index(drop=True).to_dict(orient="records")


def load_pubhealth(data_dir: str = DATA_DIR) -> tuple:
    """
    Load the train, validation and test splits from the data directory

    Args:
        data_dir (str): Directory holding train.tsv, dev.tsv and test.tsv
    Returns:
        tuple: Three lists of records in the order train, validation, test
    """
    train = read_split(os.path.join(data_dir, "train.tsv"))
    val = read_split(os.path.join(data_dir, "dev.tsv"))
    test = read_split(os.path.join(data_dir, "test.tsv"))
    print(f"Loaded PubHealth: train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


def class_distribution(records: list) -> dict:
    """
    Count records per class label name for a split
    """
    counts = {LABEL_MAP[i]: 0 for i in range(NUM_LABELS)}
    for r in records:
        counts[LABEL_MAP[r["label"]]] += 1
    return counts
