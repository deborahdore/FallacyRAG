"""
extract_results.py
------------------
Scans the output/ directory. For every top-level folder (one per RAG
configuration), reads all predict_results.json files found under
  output/<config>/<model>/<run>/predict_results.json
and writes a single Excel file:
  output/<config>/<config>.xlsx

Columns: temporal, retriever, search, rerank, query, summarizer,
         model, configuration, predict_f1_macro, predict_accuracy

The config folder name mirrors the layout produced by
``dataset/detection/create_db.py``:
  detection-temporal=<bool>-retriever=<r|false>[-search=<s>-rerank=<rr|false>]
           [-query=<q>]-summarizer=<sum|false>
The ``-search``/``-rerank`` segments are present only when a retriever is set,
and ``-query`` only for the relations mode.
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

# OUTPUT_DIR = "output/detection"
OUTPUT_DIR = "output/detection"
RESULTS_FILE = "predict_results.json"
RESULTS_FILE_ALT = "predict_results_metrics.json"
AGGREGATE_FILE = "all_results.csv"

COLUMNS = [
    "temporal", "retriever", "search", "rerank", "query", "summarizer",
    "model", "configuration", "predict_f1_macro", "predict_accuracy",
]

CONFIG_RE = re.compile(
    r"-?temporal=(?P<temporal>true|false)"
    r"-retriever=(?P<retriever>.+?)"
    r"(?:-search=(?P<search>.+?)-rerank=(?P<rerank>.+?))?"
    r"(?:-query=(?P<query>.+?))?"
    r"-summarizer=(?P<summarizer>.+)$"
)


def parse_config_folder(folder_name: str) -> dict:
    """Parse RAG hyper-parameters from the top-level folder name."""
    m = CONFIG_RE.search(folder_name)
    if not m:
        raise ValueError(f"Cannot parse config folder name: {folder_name!r}")

    def to_bool(v: str) -> bool:
        return v.lower() == "true"

    def to_optional(v: str | None) -> str:
        return "none" if v is None or v.lower() == "false" else v

    return {
        "temporal": to_bool(m.group("temporal")),
        "retriever": to_optional(m.group("retriever")),
        "search": to_optional(m.group("search")),
        "rerank": to_optional(m.group("rerank")),
        "query": to_optional(m.group("query")),
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

    aggregated = []
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

        df = pd.DataFrame(rows, columns=COLUMNS)
        df = (
            df.loc[df.groupby("model")["predict_f1_macro"].idxmax()]
            .reset_index(drop=True)
        )

        excel_path = os.path.join(OUTPUT_DIR, folder_name, "results.xlsx")
        df.to_excel(excel_path, index=False)
        # assert len(df)>=3
        logger.info("Wrote %d rows → %s", len(df), excel_path)

        aggregated.append(df)

    if not aggregated:
        logger.warning("No results to aggregate — %s not written.", AGGREGATE_FILE)
    else:
        agg_df = pd.concat(aggregated, ignore_index=True)[COLUMNS]
        agg_df = agg_df.sort_values("predict_f1_macro", ascending=False).reset_index(drop=True)
        agg_path = os.path.join(OUTPUT_DIR, AGGREGATE_FILE)
        agg_df.to_csv(agg_path, index=False)
        logger.info("Wrote %d aggregated rows → %s", len(agg_df), agg_path)
