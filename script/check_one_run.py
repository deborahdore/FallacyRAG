#!/usr/bin/env python3
"""Verify completeness of the FallacyRAG single-run detection experiment grid.
Checks that every config has the expected model folders, each with the right number of
hyperparameter combinations carrying the completion marker (``predict_results.json`` for
encoders, ``predict_results_metrics.json`` for LLMs). Problems are reported and the exit code
is non-zero. Usage: ``python check_one_run.py [BASE_DIR]``."""

import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
root = rootutils.find_root(search_from=__file__, indicator=".project-root")

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

REQUIRED_ENCODERS_BEST_FOLDER_ELECDEB60TO20_DETECTION = ["RooseBERT-scr-cased",
                                                         "ModernBERT-base",
                                                         "RooseBERT-cont-cased"]
REQUIRED_ENCODERS_BEST_FOLDER_ELECDEB60TO20_CLASSIFICATION = ["longformer-base-4096",
                                                              "RooseBERT-cont-cased",
                                                              "RooseBERT-scr-cased"]

REQUIRED_ENCODERS_BEST_FOLDER_PROPAGANDA_DETECTION = ["deberta-v3-base",
                                                      "Llama-3.1-8B-Instruct",
                                                      "Mistral-7B-Instruct-v0.3"]

ENCODER_MARKER = "predict_results.json"
LLM_MARKER = "predict_results_metrics.json"

ENCODER_COMBOS = [
    f"bs{bs}_lr{lr}_ep{ep}"
    for bs in (8, 16)
    for lr in ("2e-5", "3e-5", "5e-5")
    for ep in (3, 5)
]
LLM_COMBOS = (
        ["mode-zero_shot", "mode-few_shot"]
        + [
            f"mode-fine_tune_bs8_lr{lr}_ep{ep}"
            for lr in ("1e-4", "3e-4", "5e-4")
            for ep in (1, 2, 3)
        ]
)


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
    elif "propaganda" in str(base):
        if folder_name in BASIC_FOLDER or folder_name == BEST_FOLDER_PROPAGANDA_DETECTION:
            return ENCODERS + LLMS
        return REQUIRED_ENCODERS_BEST_FOLDER_PROPAGANDA_DETECTION
    elif "debagreeement" in str(base) or "LIARArg" in str(base):
        return ENCODERS + LLMS

    return []


def check(base: Path):
    problems = []
    stats = {"configs": 0, "combos_expected": 0, "combos_ok": 0}

    if not base.is_dir():
        print(f"ERROR: base directory not found: {base}")
        return 1

    for config in sorted(p for p in base.iterdir() if p.is_dir()):
        stats["configs"] += 1
        exp_models = expected_models(config.name, base)
        present = {p.name for p in config.iterdir() if p.is_dir()}

        for m in exp_models:
            if m not in present:
                problems.append(f"[MISSING MODEL] {config.name}/{m}")
        for m in sorted(present - set(exp_models)):
            problems.append(f"[EXTRA MODEL]   {config.name}/{m}")

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
