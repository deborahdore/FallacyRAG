#!/usr/bin/env python3
"""
Verify completeness of the FallacyRAG detection experiment grid.

For every config folder under output/one-run/elecdeb60to20-detection it checks:
  - the expected model sub-folders are present
  - each model folder has the right number of hyperparameter combinations
  - each combination folder has its completion marker
      encoders -> predict_results.json
      LLMs     -> predict_results_metrics.json

Anything missing (or unexpected extra) is reported. Exit code is non-zero
if any problem is found, so it can be dropped into CI / a cron check.

Usage:
    python check_detection.py [BASE_DIR]

BASE_DIR defaults to output/one-run/elecdeb60to20-detection next to this script.
"""

import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
root = rootutils.find_root(search_from=__file__, indicator=".project-root")

# ---------------------------------------------------------------------------
# Expected structure (edit here if the grid changes)
# ---------------------------------------------------------------------------

ENCODERS = [
    "ModernBERT-base",
    "NeoBERT",
    "RooseBERT-cont-cased",
    "RooseBERT-cont-uncased",
    "RooseBERT-scr-cased",
    "RooseBERT-scr-uncased",
    "deberta-v3-base",
    "legalbert",
    "longformer-base-4096",
    "roberta-base",
]
LLMS = [
    "Llama-3.1-8B-Instruct",
    "Qwen3.5-9B-Base",
    "gpt-oss-20b",
    "Mistral-7B-Instruct-v0.3"
]

BASIC_FOLDER = ["temporal=false-retriever=false-summarizer=false", "baseline"]

BEST_FOLDER_ELECDEB60TO20_DETECTION = "temporal=true-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false"
BEST_FOLDER_ELECDEB60TO20_CLASSIFICATION = "temporal=false-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false"

BEST_FOLDER_PROPAGANDA_DETECTION = ""

BEST_FOLDER_REDDIT_DETECTION = ""
BEST_FOLDER_REDDIT_DETECTION_CLASSIFICATION = ""

REQUIRED_ENCODERS_BEST_FOLDER_ELECDEB60TO20_DETECTION = ["RooseBERT-scr-cased",
                                                         "ModernBERT-base",
                                                         "RooseBERT-cont-cased"]
REQUIRED_ENCODERS_BEST_FOLDER_ELECDEB60TO20_CLASSIFICATION = ["longformer-base-4096",
                                                              "RooseBERT-cont-cased",
                                                              "RooseBERT-scr-cased"]

REQUIRED_ENCODERS_BEST_FOLDER_PROPAGANDA_DETECTION = ["deberta-v3-base",
                                                      "Llama-3.1-8B-Instruct",
                                                      "Mistral-7B-Instruct-v0.3"]

REQUIRED_ENCODERS_BEST_FOLDER_REDDIT_DETECTION = []
REQUIRED_ENCODERS_BEST_FOLDER_REDDIT_CLASSIFICATION = []

# Completion markers
ENCODER_MARKER = "predict_results.json"
LLM_MARKER = "predict_results_metrics.json"

# Hyperparameter combinations -------------------------------------------------
ENCODER_COMBOS = [
    f"bs{bs}_lr{lr}_ep{ep}"
    for bs in (8, 16)
    for lr in ("2e-5", "3e-5", "5e-5")
    for ep in (3, 5)
]  # 12
LLM_COMBOS = (
        ["mode-zero_shot", "mode-few_shot"]
        + [
            f"mode-fine_tune_bs8_lr{lr}_ep{ep}"
            for lr in ("1e-4", "3e-4", "5e-4")
            for ep in (1, 2, 3)
        ]
)  # 11


# ---------------------------------------------------------------------------

def expected_models(folder_name: str, base: Path):
    if "elecdeb60to20" in str(base):
        if "classification" in str(base):
            if folder_name in BASIC_FOLDER or folder_name == BEST_FOLDER_ELECDEB60TO20_CLASSIFICATION:
                return ENCODERS + LLMS
            return REQUIRED_ENCODERS_BEST_FOLDER_ELECDEB60TO20_CLASSIFICATION
        else:
            if folder_name in BASIC_FOLDER or folder_name == BEST_FOLDER_ELECDEB60TO20_DETECTION:
                return ENCODERS + LLMS
            return REQUIRED_ENCODERS_BEST_FOLDER_ELECDEB60TO20_DETECTION
    elif "reddit" in str(base):
        if "classification" in str(base):
            if folder_name in BASIC_FOLDER or folder_name == BEST_FOLDER_REDDIT_DETECTION_CLASSIFICATION:
                return ENCODERS + LLMS
            return REQUIRED_ENCODERS_BEST_FOLDER_REDDIT_CLASSIFICATION
        else:
            if folder_name in BASIC_FOLDER or folder_name == BEST_FOLDER_REDDIT_DETECTION:
                return ENCODERS + LLMS
            return REQUIRED_ENCODERS_BEST_FOLDER_REDDIT_DETECTION
    elif "propaganda" in str(base):
        # propaganda is detection-only
        if folder_name in BASIC_FOLDER or folder_name == BEST_FOLDER_PROPAGANDA_DETECTION:
            return ENCODERS + LLMS
        return REQUIRED_ENCODERS_BEST_FOLDER_PROPAGANDA_DETECTION
    # unknown dataset: nothing expected
    elif "debagreeement" in str(base) or "LIARArg" in str(base):
        return ENCODERS + LLMS

    return []


def check(base: Path):
    problems = []  # list of human-readable strings
    stats = {"configs": 0, "combos_expected": 0, "combos_ok": 0}

    if not base.is_dir():
        print(f"ERROR: base directory not found: {base}")
        return 1

    for config in sorted(p for p in base.iterdir() if p.is_dir()):
        stats["configs"] += 1
        exp_models = expected_models(config.name, base)
        present = {p.name for p in config.iterdir() if p.is_dir()}

        # missing / extra model folders
        for m in exp_models:
            if m not in present:
                problems.append(f"[MISSING MODEL] {config.name}/{m}")
        for m in sorted(present - set(exp_models)):
            problems.append(f"[EXTRA MODEL]   {config.name}/{m}")

        # per-model combination + marker checks
        for m in exp_models:
            mdir = config / m
            if not mdir.is_dir():
                continue
            is_llm = m in LLMS
            combos = LLM_COMBOS if is_llm else ENCODER_COMBOS
            marker = LLM_MARKER if is_llm else ENCODER_MARKER
            have = {p.name for p in mdir.iterdir() if p.is_dir()}

            for c in sorted(have - set(combos)):
                problems.append(f"[EXTRA COMBO]   {config.name}/{m}/{c}")

            for c in combos:
                stats["combos_expected"] += 1
                cdir = mdir / c
                if not cdir.is_dir():
                    problems.append(f"[MISSING COMBO] {config.name}/{m}/{c}")
                elif not (cdir / marker).is_file():
                    problems.append(
                        f"[NO MARKER]     {config.name}/{m}/{c}  (need {marker})"
                    )
                else:
                    stats["combos_ok"] += 1

    # report -----------------------------------------------------------------
    print(f"Scanned {stats['configs']} config folders under {base}")
    print(f"Combinations expected: {stats['combos_expected']} | "
          f"complete: {stats['combos_ok']} | "
          f"problems: {len(problems)}\n")

    if not problems:
        print("All good - nothing missing.")
        return 0

    for p in problems:
        print(p)
    print(f"\n{len(problems)} problem(s) found.")
    return 1


if __name__ == "__main__":
    dataset = "propaganda"
    task = "detection"
    default = root / "output" / "one-run" / (dataset + "-" + task)
    base = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default
    sys.exit(check(base))
