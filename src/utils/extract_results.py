"""
extract_results.py
------------------
Scans the output/ directory. For every top-level folder (one per RAG
configuration), reads all predict_results.json files found under
  output/<config>/<model>/<run>/predict_results.json
and writes a single Excel file:
  output/<config>/<config>.xlsx

Columns: agentic, temporal, retriever, summarizer, model, configuration,
         predict_f1_macro, predict_accuracy
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

OUTPUT_DIR = "output/detection"
RESULTS_FILE = "predict_results.json"
RESULTS_FILE_ALT = "predict_results_metrics.json"

# Matches folder names like:
#   agentic=false-temporal=true-retriever=bm25-summarizer=false
# Retriever may contain dashes (e.g. baai-bge-large-en-v1.5), so we use a
# greedy group for it and rely on the literal "-summarizer=" anchor at the end.
CONFIG_RE = re.compile(
    r"^agentic=(?P<agentic>[^-]+)"
    r"-temporal=(?P<temporal>[^-]+)"
    r"-retriever=(?P<retriever>.+)"
    r"-summarizer=(?P<summarizer>.+)$"
)


def parse_config_folder(folder_name: str) -> dict:
    """Parse RAG hyper-parameters from the top-level folder name."""
    m = CONFIG_RE.match(folder_name)
    if not m:
        raise ValueError(f"Cannot parse config folder name: {folder_name!r}")

    def to_bool(v: str) -> bool:
        return v.lower() == "true"

    def to_optional(v: str) -> str:
        return "none" if v.lower() == "false" else v

    return {
        "agentic": to_bool(m.group("agentic")),
        "temporal": to_bool(m.group("temporal")),
        "retriever": to_optional(m.group("retriever")),
        "summarizer": to_optional(m.group("summarizer")),
    }


def collect_rows(config_path: str, config_meta: dict) -> list[dict]:
    """Walk model/run subdirectories and collect one row per valid run."""
    rows = []
    for model_name in sorted(os.listdir(config_path)):
        model_path = os.path.join(config_path, model_name)
        if not os.path.isdir(model_path):
            continue
        for run_name in sorted(os.listdir(model_path)):
            run_path = os.path.join(model_path, run_name)
            if not os.path.isdir(run_path):
                continue
            results_path = os.path.join(run_path, RESULTS_FILE)
            alt = False
            if not os.path.isfile(results_path):
                results_path = os.path.join(run_path, RESULTS_FILE_ALT)
                alt = True
            if not os.path.isfile(results_path):
                logger.warning("Missing results file in %s — skipping", run_path)
                continue
            try:
                with open(results_path, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
            except json.JSONDecodeError as exc:
                logger.warning("Bad JSON in %s: %s — skipping", results_path, exc)
                continue

            rows.append({
                **config_meta,
                "model": model_name,
                "configuration": run_name,
                "predict_f1_macro": metrics.get("f1_macro" if alt else "predict_f1_macro"),
                "predict_accuracy": metrics.get("accuracy" if alt else "predict_accuracy"),
            })
    return rows


if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        logger.error("Output directory %r not found. Run from the project root.", OUTPUT_DIR)
        raise SystemExit(1)

    config_folders = sorted(
        e for e in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, e))
    )

    for folder_name in config_folders:
        config_path = os.path.join(OUTPUT_DIR, folder_name)
        try:
            config_meta = parse_config_folder(folder_name)
        except ValueError as exc:
            logger.warning("%s — skipping folder", exc)
            continue

        rows = collect_rows(config_path, config_meta)
        if not rows:
            logger.warning("No results found in %s — skipping", config_path)
            continue

        df = pd.DataFrame(rows, columns=[
            "agentic", "temporal", "retriever", "summarizer",
            "model", "configuration", "predict_f1_macro", "predict_accuracy",
        ])
        df = (
            df.loc[df.groupby("model")["predict_f1_macro"].idxmax()]
            .reset_index(drop=True)
        )

        excel_path = os.path.join(OUTPUT_DIR, folder_name, "results.xlsx")
        df.to_excel(excel_path, index=False)
        logger.info("Wrote %d rows → %s", len(df), excel_path)
