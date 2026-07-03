"""
extract_results_5runs.py
------------------------
Scans the ``output_5runs`` directory, which stores several seeds per model:

  output_5runs/<task>/<config>/<model>/seed_<n>/<run>/predict_results.json

(encoder runs finish with ``predict_results.json``; LLM runs finish with
``predict_results_metrics.json`` — both are supported.)

For every config folder it writes a CSV summarising the metrics of each model
aggregated over its seeds:

  output_5runs/<task>/<config>/results_mean_std.csv

Each row is one model, with the mean and standard deviation of the metrics
computed across all available seeds, plus the number of seeds that were found.
An aggregate CSV combining every config folder is also written to
``output_5runs/<task>/all_results_mean_std.csv``.
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

# Root that holds one sub-folder per task (e.g. "detection", "classification").
OUTPUT_ROOT = "output_5runs"
RESULTS_FILE = "predict_results.json"
RESULTS_FILE_ALT = "predict_results_metrics.json"
AGGREGATE_FILE = "all_results_mean_std.csv"

# Metrics to summarise (encoder key, alt/LLM key).
METRICS = [
    ("predict_accuracy", "accuracy"),
    ("predict_f1_macro", "f1_macro"),
    ("predict_precision", "precision"),
    ("predict_recall", "recall"),
]

CONFIG_META_COLUMNS = [
    "task", "config", "temporal", "retriever", "search", "rerank", "query", "summarizer",
]

CONFIG_RE = re.compile(
    r"-?temporal=(?P<temporal>true|false)"
    r"-retriever=(?P<retriever>.+?)"
    r"(?:-search=(?P<search>.+?)-rerank=(?P<rerank>.+?))?"
    r"(?:-query=(?P<query>.+?))?"
    r"-summarizer=(?P<summarizer>.+)$"
)


def parse_config_folder(folder_name: str) -> dict:
    """Parse RAG hyper-parameters from the config folder name."""
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


def find_results_file(seed_path: str) -> str | None:
    """Locate the metrics json under a seed folder (searches any run subdir)."""
    for dirpath, _dirnames, filenames in os.walk(seed_path):
        for candidate in (RESULTS_FILE, RESULTS_FILE_ALT):
            if candidate in filenames:
                return os.path.join(dirpath, candidate)
    return None


def load_metrics(results_path: str) -> dict | None:
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning("Bad JSON in %s: %s — skipping", results_path, exc)
        return None

    row = {}
    for main_key, alt_key in METRICS:
        value = metrics.get(main_key, metrics.get(alt_key))
        row[main_key] = value
    return row


def collect_seed_rows(model_path: str) -> list[dict]:
    """One row of raw metrics per seed found under a model folder."""
    rows = []
    for seed_name in sorted(os.listdir(model_path)):
        seed_path = os.path.join(model_path, seed_name)
        if not os.path.isdir(seed_path):
            continue
        results_path = find_results_file(seed_path)
        if results_path is None:
            logger.warning("No results file under %s — skipping", seed_path)
            continue
        metrics = load_metrics(results_path)
        if metrics is None:
            continue
        metrics["seed"] = seed_name
        rows.append(metrics)
    return rows


def summarise_config(config_path: str) -> pd.DataFrame:
    """Build a per-model mean/std DataFrame for one config folder."""
    metric_cols = [main for main, _ in METRICS]
    summary_rows = []
    for model_name in sorted(os.listdir(config_path)):
        model_path = os.path.join(config_path, model_name)
        if not os.path.isdir(model_path):
            continue
        seed_rows = collect_seed_rows(model_path)
        if not seed_rows:
            logger.warning("No seeds with results for model %s — skipping", model_path)
            continue

        model_df = pd.DataFrame(seed_rows)
        row = {"model": model_name, "n_seeds": len(seed_rows)}
        for col in metric_cols:
            values = pd.to_numeric(model_df.get(col), errors="coerce")
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std(ddof=1)  # sample std; NaN if a single seed
        summary_rows.append(row)

    columns = ["model", "n_seeds"] + [
        f"{col}_{stat}" for col in metric_cols for stat in ("mean", "std")
    ]
    return pd.DataFrame(summary_rows, columns=columns)


def process_task(task_path: str, task_name: str) -> list[pd.DataFrame]:
    aggregated = []
    config_folders = sorted(
        e for e in os.listdir(task_path)
        if os.path.isdir(os.path.join(task_path, e))
    )
    for folder_name in config_folders:
        config_path = os.path.join(task_path, folder_name)
        try:
            config_meta = parse_config_folder(folder_name)
        except ValueError as exc:
            logger.warning("%s — skipping folder", exc)
            continue

        df = summarise_config(config_path)
        if df.empty:
            logger.warning("No results found in %s — skipping", config_path)
            continue

        csv_path = os.path.join(config_path, "results_mean_std.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %d models → %s", len(df), csv_path)

        meta_df = df.copy()
        meta_df.insert(0, "config", folder_name)
        meta_df.insert(0, "task", task_name)
        for key, value in config_meta.items():
            meta_df[key] = value
        aggregated.append(meta_df)
    return aggregated


if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_ROOT):
        logger.error("Output directory %r not found. Run from the project root.", OUTPUT_ROOT)
        raise SystemExit(1)

    task_names = sorted(
        e for e in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, e))
    )

    for task_name in task_names:
        task_path = os.path.join(OUTPUT_ROOT, task_name)
        aggregated = process_task(task_path, task_name)
        if not aggregated:
            logger.warning("No results to aggregate for task %s.", task_name)
            continue

        metric_cols = [main for main, _ in METRICS]
        ordered = CONFIG_META_COLUMNS + ["model", "n_seeds"] + [
            f"{col}_{stat}" for col in metric_cols for stat in ("mean", "std")
        ]
        agg_df = pd.concat(aggregated, ignore_index=True)
        agg_df = agg_df[[c for c in ordered if c in agg_df.columns]]
        agg_df = agg_df.sort_values(
            ["config", "predict_f1_macro_mean"], ascending=[True, False]
        ).reset_index(drop=True)
        agg_path = os.path.join(task_path, AGGREGATE_FILE)
        agg_df.to_csv(agg_path, index=False)
        logger.info("Wrote %d aggregated rows → %s", len(agg_df), agg_path)
