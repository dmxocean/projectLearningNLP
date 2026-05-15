# -*- coding: utf-8 -*-
"""PUBHEALTH dataset acquisition stage for the NLP pipeline

This script downloads the PUBHEALTH archive from Google Drive, extracts split files, normalizes their location into a shared data directory, and reports row counts for train, dev, and test outputs
Primary input is a static Google Drive file id and the primary outputs are data/train.tsv, data/dev.tsv, and data/test.tsv under the project data route

Critical design choice: all routes are anchored to __file__ to keep execution consistent when the module is imported or executed from different working directories
"""

import os
import zipfile

import gdown
import pandas as pd


# --- Routes ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR) 
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
ZIP_PATH = os.path.join(DATA_DIR, "PUBHEALTH.zip")

TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
DEV_PATH = os.path.join(DATA_DIR, "dev.tsv")
TEST_PATH = os.path.join(DATA_DIR, "test.tsv")


def download_and_extract_pubhealth():
    """Download and normalize PUBHEALTH split files into the project data directory

    Returns:
        dict: Row counts keyed by split name plus a total count

    File logic:
        - Creates the data directory when missing
        - Downloads the archive to ZIP_PATH
        - Extracts all archive content under DATA_DIR
        - Moves discovered TSV files to DATA_DIR root when extraction creates nested folders
        - Removes the temporary archive after extraction
        - Reads each expected split TSV file and computes row counts

    IMPORTANT:
        This function must complete extraction and normalization before counting rows, otherwise row totals can be incomplete when files remain in nested directories
    """

    # --- Initialization ---
    file_id = "1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj" # Fixed source id aligned with the official dataset release
    download_url = f"https://drive.google.com/uc?id={file_id}" # Direct download URL expected by gdown
    os.makedirs(DATA_DIR, exist_ok=True) # Ensure target directory exists before download starts

    # --- Download And Extraction ---
    print("Downloading PUBHEALTH archive")
    gdown.download(download_url, ZIP_PATH, quiet=True) # quiet=True keeps output constrained to explicit script status lines

    print("Extracting archive content")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR) # Extraction may create nested folders depending on archive structure

    # --- Split Normalization ---
    extracted_tsv_files = []
    for root_dir, _, file_names in os.walk(DATA_DIR):
        for file_name in file_names:
            if file_name.endswith(".tsv"):
                file_route = os.path.join(root_dir, file_name)
                extracted_tsv_files.append(file_route) # Gather all split candidates before relocation

    for source_path in extracted_tsv_files:
        target_path = os.path.join(DATA_DIR, os.path.basename(source_path))
        if source_path != target_path:
            os.replace(source_path, target_path) # Replace ensures idempotent relocation across reruns

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH) # Archive removal prevents stale artifacts from accumulating

    # --- Output Validation ---
    split_routes = {
        "train": TRAIN_PATH,
        "dev": DEV_PATH,
        "test": TEST_PATH,
    }
    split_counts = {}

    for split_name, split_path in split_routes.items():
        if os.path.exists(split_path):
            split_frame = pd.read_csv(split_path, sep="\t")
            split_counts[split_name] = len(split_frame) # DataFrame length gives row count used for downstream sanity checks
        else:
            split_counts[split_name] = 0 # Missing files are recorded as zero to keep reporting schema stable

    split_counts["total"] = sum(split_counts.values())
    print(f"PUBHEALTH ready with {split_counts['total']} rows")

    return split_counts


if __name__ == "__main__":
    download_and_extract_pubhealth()
