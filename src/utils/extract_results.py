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

DIR = "output_no_rag"
RESULTS_DIR = "results"
HYPERPARAMS_RE = re.compile(r"bs(\d+)_lr([0-9.e+-]+)_ep(\d+)")
RESULTS_FILE = "predict_results.json"
F1_KEY = "predict_f1_macro"


def parse_model_dir(configuration: str) -> dict | None:
    results_path = os.path.join(configuration, RESULTS_FILE)
    match = HYPERPARAMS_RE.search(configuration)

    with open(results_path, "r", encoding="utf-8") as fh:
        json_obj = json.load(fh)

    return {
        # "directory": DIR,
        "model": results_path.split("/")[1],
        "batch": int(match.group(1)),
        "lr": match.group(2),
        **json_obj,
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = [
        e for e in os.listdir(DIR)
        if os.path.isdir(os.path.join(DIR, e))
    ]
    configurations = [
        os.path.join(DIR, model, e) for model in models
        for e in os.listdir(os.path.join(DIR, model))
        if os.path.isdir(os.path.join(DIR, model, e))
    ]

    rows = [parse_model_dir(configuration) for configuration in configurations]
    # rows = [r for r in rows if r is not None]

    all_df = (
        pd.DataFrame(rows)
        # .dropna()
        # .drop_duplicates()
        .reset_index(drop=True)
    )
    all_path = os.path.join(RESULTS_DIR, DIR + "_all.csv")
    all_df.to_csv(all_path, index=False)
    logger.info("Wrote %d rows to %s", len(all_df), all_path)

    best_df = (
        all_df
        .loc[all_df.groupby("model")["predict_f1_macro"].idxmax()]
        .reset_index(drop=True)
    )
    best_path = os.path.join(RESULTS_DIR, DIR + "_best.csv")
    best_df.to_csv(best_path, index=False)
    logger.info("Wrote %d rows to %s", len(best_df), best_path)
