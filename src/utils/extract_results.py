"""
extract_results.py
------------------
Scans model output directories, parses hyperparameters from directory names,
and writes two CSVs:
  - results/all.csv  : every valid run
  - results/best.csv : best macro-F1 run per model architecture
"""

import json
import logging
import os
import re

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIR = "output"
RESULTS_DIR = "results"
HYPERPARAMS_RE = re.compile(r"epochs=(\d+)-batch=(\d+)-lr=([0-9.e+-]+)")
MODEL_NAME_RE = re.compile(r"^(.*?)-epochs=")
RESULTS_FILE = "predict_results.json"
F1_KEY = "predict_f1_macro"


def parse_model_dir(model: str) -> dict | None:
    results_path = os.path.join(DIR, model, RESULTS_FILE)
    match = HYPERPARAMS_RE.search(model)
    name_match = MODEL_NAME_RE.match(model)

    with open(results_path, "r", encoding="utf-8") as fh:
        json_obj = json.load(fh)

    return {
        "directory": DIR,
        "model": name_match.group(1),
        "epochs": int(match.group(1)),
        "batch": int(match.group(2)),
        "lr": float(match.group(3)),
        "macro_f1": json_obj[F1_KEY],
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    entries = [
        e for e in os.listdir(DIR)
        if os.path.isdir(os.path.join(DIR, e)) and not e.startswith(".")
    ]

    rows = [parse_model_dir(model) for model in entries]
    rows = [r for r in rows if r is not None]

    all_df = (
        pd.DataFrame(rows)
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    all_path = os.path.join(RESULTS_DIR, "all.csv")
    all_df.to_csv(all_path, index=False)
    logger.info("Wrote %d rows to %s", len(all_df), all_path)

    best_df = (
        all_df
        .loc[all_df.groupby("model")["macro_f1"].idxmax()]
        .reset_index(drop=True)
    )
    best_path = os.path.join(RESULTS_DIR, "best.csv")
    best_df.to_csv(best_path, index=False)
    logger.info("Wrote %d rows to %s", len(best_df), best_path)
