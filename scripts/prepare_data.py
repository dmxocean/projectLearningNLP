# -*- coding: utf-8 -*-
"""
Download the PubHealth dataset and write it to data/pubhealth as TSV splits

Newer datasets releases no longer run dataset loading scripts, so this fetches the
Hugging Face auto-converted Parquet files for health_fact directly

It reads the authoritative ClassLabel names from the Parquet metadata, drops
the unverified rows whose label is -1, maps the label names onto the project convention
(true=0, false=1, unproven=2, mixture=3) and writes train.tsv, dev.tsv and test.tsv
with the columns the loader expects

Tabs and newlines inside text are flattened so the TSV round-trips cleanly
"""

import os
import sys
import json

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

from detection.config import LABEL_MAP_STR
from detection.paths import DATA_DIR

CANDIDATE_DATASET_IDS = ["health_fact", "ImperialCollegeLondon/health_fact"]
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_FILES = {
    "train": "default/train/0000.parquet",
    "validation": "default/validation/0000.parquet",
    "test": "default/test/0000.parquet",
}
HF_SPLIT_TO_FILE = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
TEXT_COLUMNS = ["claim", "explanation", "main_text"]
FALLBACK_LABEL_NAMES = ["false", "mixture", "true", "unproven"]


def _flatten(text) -> str:
    """
    Replace tab and newline characters with single spaces for safe TSV storage
    """
    return " ".join(str(text or "").split())


def _download_parquet(hf_split: str) -> str:
    """
    Fetch one Parquet split from the auto-convert revision, trying known repo ids

    Args:
        hf_split (str): One of train, validation, test
    Returns:
        str: Local cached path of the downloaded Parquet file
    """
    from huggingface_hub import hf_hub_download

    last_error = None
    for dataset_id in CANDIDATE_DATASET_IDS:
        try:
            path = hf_hub_download(
                dataset_id, PARQUET_FILES[hf_split],
                repo_type="dataset", revision=PARQUET_REVISION,
            )
            print(f"Downloaded {dataset_id}:{PARQUET_FILES[hf_split]}")
            return path
        except Exception as error:                    # Try the next candidate on any failure
            last_error = error
    raise RuntimeError(f"Could not download PubHealth Parquet: {last_error}")


def _label_names(parquet_path: str) -> list:
    """
    Read the ClassLabel names for the label column from Parquet metadata

    Args:
        parquet_path (str): Path to a downloaded Parquet split
    Returns:
        list: Ordered label names, falling back to the documented order if absent
    """
    import pyarrow.parquet as pq

    metadata = pq.read_schema(parquet_path).metadata or {}
    blob = metadata.get(b"huggingface")
    if blob is not None:
        try:
            info = json.loads(blob.decode("utf-8"))
            label_feature = info["info"]["features"]["label"]
            names = label_feature.get("names")
            if names:
                return [str(n).strip().lower() for n in names]
        except Exception:                              # Fall through to the documented order
            pass
    print("Falling back to the documented health_fact label order")
    return FALLBACK_LABEL_NAMES


def main():
    import pandas as pd

    os.makedirs(DATA_DIR, exist_ok=True)
    grand_total = {}

    for hf_split, filename in HF_SPLIT_TO_FILE.items():
        parquet_path = _download_parquet(hf_split)
        names = _label_names(parquet_path)
        frame = pd.read_parquet(parquet_path)

        rows = []
        for _, example in frame.iterrows():
            label_value = example.get("label")
            if label_value is None or int(label_value) < 0:
                continue                                # Drop unverified rows
            name = names[int(label_value)]
            if name not in LABEL_MAP_STR:
                continue                                # Skip any unexpected label name
            row = {col: _flatten(example.get(col, "")) for col in TEXT_COLUMNS}
            row["label"] = name
            rows.append(row)

        out_frame = pd.DataFrame(rows, columns=TEXT_COLUMNS + ["label"])
        out_path = os.path.join(DATA_DIR, filename)
        out_frame.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

        counts = out_frame["label"].value_counts().to_dict()
        for k, v in counts.items():
            grand_total[k] = grand_total.get(k, 0) + v
        print(f"{hf_split:>10} -> {out_path}  ({len(out_frame)} rows)  {counts}")

    print(f"\nLabel order used: {names}")
    print(f"Total per-class across splits: {grand_total}")
    print("Expected train counts (PubHealth): true 5078, false 3001, mixture 1434, unproven 291")


if __name__ == "__main__":
    main()
