#!/usr/bin/env python3
"""Verify completeness of the FallacyRAG 5-runs (multi-seed) detection grid.
Checks that every config has all 15 model folders, each with its 5 seed runs carrying the
completion marker (``predict_results.json`` for encoders, ``predict_results_metrics.json`` for
LLMs). LLM seeds are checked per prompting mode (``mode-<mode>_*`` run folders). Missing or unexpected items are reported one per line and the exit code is non-zero.
Usage: ``python check_five_runs.py [BASE_DIR]``."""

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
ALL_MODELS = ENCODERS + LLMS

SEEDS = [3, 14, 42, 55, 89]
SEED_DIRS = [f"seed_{s}" for s in SEEDS]

ENCODER_MARKER = "predict_results.json"
LLM_MARKER = "predict_results_metrics.json"

LLM_MODES = ["zero_shot", "few_shot", "fine_tune"]
BASELINE = "temporal=false-retriever=false-summarizer=false"
# Config folders expected in each five-runs dataset: baseline + best RAG config
EXPECTED_CONFIGS = {
    "elecdeb60to20-detection": [BASELINE,
                                "temporal=true-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false"],
    "elecdeb60to20-classification": [BASELINE,
                                     "temporal=false-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false"],
    "propaganda-detection": [BASELINE,
                             "temporal=false-retriever=sentence-transformers-all-mpnet-base-v2-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false"],
    "LIARArg-relation-classification": ["baseline"],
}

# Datasets where only some LLM modes were run
MODES_BY_DATASET = {"LIARArg-relation-classification": ["fine_tune"]}


def has_marker(seed_dir: Path, marker: str) -> bool:
    """True if the marker file exists anywhere inside the seed run folder."""
    return any(seed_dir.rglob(marker))


def check(base: Path):
    modes = MODES_BY_DATASET.get(base.name, LLM_MODES)
    problems = []
    stats = {"configs": 0, "runs_expected": 0, "runs_ok": 0}

    if not base.is_dir():
        print(f"ERROR: base directory not found: {base}")
        return 1

    if base.name not in EXPECTED_CONFIGS:
        print(f"WARNING: no expected configs for {base.name}, missing configs cannot be checked")
    for c in EXPECTED_CONFIGS.get(base.name, []):
        if not (base / c).is_dir():
            problems.append(f"[MISSING CONFIG] {c}")

    for config in sorted(p for p in base.iterdir() if p.is_dir()):
        stats["configs"] += 1
        present = {p.name for p in config.iterdir() if p.is_dir()}

        for m in ALL_MODELS:
            if m not in present:
                problems.append(f"[MISSING MODEL] {config.name}/{m}")
        for m in sorted(present - set(ALL_MODELS)):
            problems.append(f"[EXTRA MODEL]   {config.name}/{m}")

        for m in ALL_MODELS:
            mdir = config / m
            if not mdir.is_dir():
                continue
            is_llm = m in LLMS
            marker = LLM_MARKER if is_llm else ENCODER_MARKER
            have = {p.name for p in mdir.iterdir() if p.is_dir()}

            for s in sorted(have - set(SEED_DIRS)):
                problems.append(f"[EXTRA SEED]    {config.name}/{m}/{s}")

            for s in SEED_DIRS:
                sdir = mdir / s
                if not is_llm:
                    stats["runs_expected"] += 1
                    if not sdir.is_dir():
                        problems.append(f"[MISSING SEED]  {config.name}/{m}/{s}")
                    elif not has_marker(sdir, marker):
                        problems.append(f"[NO MARKER]     {config.name}/{m}/{s}  (need {marker})")
                    else:
                        stats["runs_ok"] += 1
                    continue

                stats["runs_expected"] += len(modes)
                if not sdir.is_dir():
                    problems.append(f"[MISSING SEED]  {config.name}/{m}/{s}  ({len(modes)} modes)")
                    continue
                for mode in modes:
                    runs = [p for p in sdir.iterdir() if p.is_dir() and p.name.startswith(f"mode-{mode}_")]
                    if not runs:
                        problems.append(f"[MISSING MODE]  {config.name}/{m}/{s}/{mode}")
                    elif not any(has_marker(r, marker) for r in runs):
                        problems.append(f"[NO MARKER]     {config.name}/{m}/{s}/{mode}  (need {marker})")
                    else:
                        stats["runs_ok"] += 1

    print(f"Scanned {stats['configs']} config folders under {base}")
    print(f"Seed runs expected: {stats['runs_expected']} | "
          f"complete: {stats['runs_ok']} | "
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
    default = root / "output" / "five-runs" / (dataset + "-" + task)
    base = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default
    sys.exit(check(base))
