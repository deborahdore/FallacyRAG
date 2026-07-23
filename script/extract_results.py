"""
extract_results.py
------------------
Scans the output/one-run/ directory. For every top-level folder (one per
dataset-task combination, e.g. elecdeb60to20-detection), reads all
predict_results.json files found under
  output/one-run/<dataset-task>/<config>/<model>/<run>/predict_results.json
and writes a single CSV file:
  output/one-run/<dataset-task>/<config>/results.csv

Columns: temporal, retriever, search, rerank, query, summarizer,
         model, configuration, predict_f1_macro, predict_accuracy

The config folder name mirrors the layout produced by
``datasets/elecdeb60to20-dataset/detection/create_db.py``:
  detection-temporal=<bool>-retriever=<r|false>[-search=<s>-rerank=<rr|false>]
           [-query=<q>]-summarizer=<sum|false>
The ``-search``/``-rerank`` segments are present only when a retriever is set,
and ``-query`` only for the relations mode.

Tasks with no RAG dimension at all (e.g. output/one-run/LIARArg-relation-classification)
use a single literal ``baseline`` config folder instead -- treated as the no-retrieval case
(temporal=false, retriever/search/rerank/query=none, summarizer=none).
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

ROOT_OUTPUT_DIR = os.path.join(rootutils.find_root(search_from=__file__, indicator=".project-root"), "output/one-run")
RESULTS_FILE = "predict_results.json"
RESULTS_FILE_ALT = "predict_results_metrics.json"
AGGREGATE_FILE = "all_results.csv"

COLUMNS = [
    "dataset", "temporal", "retriever", "search", "rerank", "query", "summarizer",
    "model", "configuration", "predict_f1_macro", "predict_accuracy"
]

CONFIG_RE = re.compile(
    r"-?temporal=(?P<temporal>true|false)"
    r"-retriever=(?P<retriever>.+?)"
    r"(?:-search=(?P<search>.+?)-rerank=(?P<rerank>.+?))?"
    r"(?:-query=(?P<query>.+?))?"
    r"-summarizer=(?P<summarizer>.+)$"
)

BASELINE_FOLDER = "baseline"


def parse_config_folder(folder_name: str) -> dict:
    """Parse RAG hyper-parameters from the top-level folder name.

    Tasks with no RAG dimension (e.g. LIARArg-relation-classification) use a single literal
    "baseline" folder instead of the temporal=/retriever=/... naming -- treat it as the
    no-retrieval case rather than failing to parse.
    """
    if folder_name == BASELINE_FOLDER:
        return {
            "temporal": False,
            "retriever": "none",
            "search": "none",
            "rerank": "none",
            "query": "none",
            "summarizer": "none",
        }

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
                "dataset": config_path,
                "predict_f1_macro": metrics.get("f1_macro" if alt else "predict_f1_macro"),
                "predict_accuracy": metrics.get("accuracy" if alt else "predict_accuracy"),
            })
    return rows


if __name__ == "__main__":
    if not os.path.isdir(ROOT_OUTPUT_DIR):
        logger.error("Output directory %r not found. Run from the project root.", ROOT_OUTPUT_DIR)
        raise SystemExit(1)
    os.system(f"find {ROOT_OUTPUT_DIR} -name '*.csv' -type f -delete")

    for task in os.listdir(ROOT_OUTPUT_DIR):
        output_dir = os.path.join(ROOT_OUTPUT_DIR, task)
        config_folders = sorted(
            e for e in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, e))
        )

        aggregated = []
        for folder_name in config_folders:
            config_path = os.path.join(output_dir, folder_name)
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
            best_metric = "predict_accuracy" if "detection" in output_dir else "predict_f1_macro"
            df = (
                df.loc[df.groupby("model")[best_metric].idxmax()]
                .reset_index(drop=True)
            )

            csv_path = os.path.join(output_dir, folder_name, "results.csv")
            df.to_csv(csv_path, index=False)
            # assert len(df)>=3
            logger.info("Wrote %d rows → %s", len(df), csv_path)

            aggregated.append(df)

        if not aggregated:
            logger.warning("No results to aggregate — %s not written.", AGGREGATE_FILE)
        else:
            agg_df = pd.concat(aggregated, ignore_index=True)[COLUMNS]
            agg_metric = "predict_accuracy" if "detection" in output_dir else "predict_f1_macro"
            agg_df = agg_df.sort_values(agg_metric, ascending=False).reset_index(drop=True)
            agg_path = os.path.join(output_dir, AGGREGATE_FILE)
            agg_df.to_csv(agg_path, index=False)
            logger.info("Wrote %d aggregated rows → %s", len(agg_df), agg_path)
