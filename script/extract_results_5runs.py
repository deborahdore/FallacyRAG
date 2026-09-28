"""Aggregate multi-seed results into per-config CSVs.
Reads ``output/five-runs/<dataset-task>/<config>/<model>/seed_<n>/<run>/`` for
``predict_results.json`` (encoders) or ``predict_results_metrics.json`` (LLMs) and writes
``results_mean_std.csv`` per config, one row per model and run (so zero_shot, few_shot and
fine_tune stay separate) with mean, standard deviation and seed count, plus a combined
``all_results_mean_std.csv``."""

import json
import logging
import os
import re

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = "output/five-runs"
RESULTS_FILE = "predict_results.json"
RESULTS_FILE_ALT = "predict_results_metrics.json"
AGGREGATE_FILE = "all_results_mean_std.csv"

METRICS = [
    ("predict_accuracy", "accuracy"),
    ("predict_f1_macro", "f1_macro"),
    ("predict_precision", "precision"),
    ("predict_recall", "recall"),
]

BASELINE_CONFIGS = ("baseline", "temporal=false-retriever=false-summarizer=false")

MODE_RE = re.compile(r"^mode-(?P<mode>zero_shot|few_shot|fine_tune)")
SEED_SUFFIX_RE = re.compile(r"_seed\d+$")

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
    if folder_name == "baseline":
        return {key: "none" for key in
                ("temporal", "retriever", "search", "rerank", "query", "summarizer")}

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


def run_mode(run_name: str) -> str:
    """Prompting mode of a run: zero_shot/few_shot/fine_tune; encoder sweeps are all fine_tune."""
    m = MODE_RE.match(run_name)
    return m.group("mode") if m else "fine_tune"


def find_results_files(seed_path: str) -> dict[str, str]:
    """Every metrics json under a seed folder, keyed by run name with the _seed<N> suffix stripped."""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(seed_path):
        for candidate in (RESULTS_FILE, RESULTS_FILE_ALT):
            if candidate not in filenames:
                continue
            if os.path.abspath(dirpath) == os.path.abspath(seed_path):
                run_name = "default"
            else:
                run_name = SEED_SUFFIX_RE.sub("", os.path.basename(dirpath))
            found[run_name] = os.path.join(dirpath, candidate)
    return found


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
    """One row of raw metrics per (run, seed) found under a model folder."""
    rows = []
    for seed_name in sorted(os.listdir(model_path)):
        seed_path = os.path.join(model_path, seed_name)
        if not os.path.isdir(seed_path):
            continue
        found = find_results_files(seed_path)
        if not found:
            logger.warning("No results file under %s — skipping", seed_path)
            continue
        for run_name, results_path in sorted(found.items()):
            metrics = load_metrics(results_path)
            if metrics is None:
                continue
            metrics["seed"] = seed_name
            metrics["configuration"] = run_name
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
        for run_name, run_df in model_df.groupby("configuration"):
            row = {
                "model": model_name,
                "mode": run_mode(run_name),
                "configuration": run_name,
                "n_seeds": len(run_df),
            }
            for col in metric_cols:
                values = pd.to_numeric(run_df.get(col), errors="coerce")
                row[f"{col}_mean"] = round(values.mean(), 3)
                row[f"{col}_std"] = round(values.std(ddof=1), 3)
                row[f"{col}_values"] = " ".join(
                    "nan" if pd.isna(v) else f"{round(v, 3)}" for v in values
                )
            summary_rows.append(row)

    columns = ["model", "mode", "configuration", "n_seeds"] + [
        f"{col}_{stat}" for col in metric_cols for stat in ("mean", "std", "values")
    ]
    return pd.DataFrame(summary_rows, columns=columns)


def add_baseline_deltas(agg_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Add delta columns per model and mode: config mean minus the same mode's baseline mean."""
    baseline = agg_df[agg_df["config"].isin(BASELINE_CONFIGS)]
    if baseline.empty:
        logger.warning("No baseline config found — delta columns will be NaN")

    keys = ["model", "mode"]
    for col in metric_cols:
        if baseline.empty:
            agg_df[f"{col}_delta"] = float("nan")
            continue
        # a mode with several hparam runs in the baseline is represented by its best
        ref = baseline.groupby(keys)[f"{col}_mean"].max()
        base_means = pd.Series(
            pd.MultiIndex.from_frame(agg_df[keys]).map(ref), index=agg_df.index
        )
        agg_df[f"{col}_delta"] = (agg_df[f"{col}_mean"] - base_means).round(3)
    return agg_df


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
    os.system(f"find {OUTPUT_ROOT} -name '*.csv' -type f -delete")

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
        ordered = CONFIG_META_COLUMNS + ["model", "mode", "configuration", "n_seeds"] + [
            f"{col}_{stat}" for col in metric_cols
            for stat in ("mean", "std", "values", "delta")
        ]
        agg_df = pd.concat(aggregated, ignore_index=True)
        agg_df = add_baseline_deltas(agg_df, metric_cols)
        agg_df = agg_df[[c for c in ordered if c in agg_df.columns]]
        agg_df = agg_df.sort_values(
            ["config", "predict_f1_macro_mean", "model", "mode"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
        agg_path = os.path.join(task_path, AGGREGATE_FILE)
        agg_df.to_csv(agg_path, index=False)
        logger.info("Wrote %d aggregated rows → %s", len(agg_df), agg_path)
