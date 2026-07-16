#!/usr/bin/env python3
"""
Verify completeness of the FallacyRAG 5-runs (multi-seed) detection grid.

Structure expected under output/five-runs/elecdeb60to20-detection:
    <config>/<model>/seed_<N>/<run>/<completion marker>

For every config folder it checks:
  - all 15 model folders are present (11 encoders + 4 LLMs)
  - each model folder has the 5 seed runs (seeds 3, 14, 42, 55, 89)
  - each seed run contains its completion marker
      encoders -> predict_results.json
      LLMs     -> predict_results_metrics.json
    (searched recursively inside seed_<N>, since the run sub-folder is
     named after the best hyperparameter combo, e.g. bs8_lr3e-5_ep5_seed42)

Anything missing (or unexpected extra) is reported, one item per line.
Exit code is non-zero if any problem is found.

Usage:
    python check_5runs.py [BASE_DIR]

BASE_DIR defaults to output/five-runs/elecdeb60to20-detection next to this script.
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
    "ModernBERT-base", "NeoBERT",
    "RooseBERT-cont-cased", "RooseBERT-cont-uncased",
    "RooseBERT-scr-cased", "RooseBERT-scr-uncased",
    "deberta-v3-base", "legalbert",
    "longformer-base-4096", "roberta-base", "xlm-roberta-base",
]
LLMS = [
    "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3",
    "Qwen3.5-9B-Base", "gpt-oss-20b",
]
ALL_MODELS = ENCODERS + LLMS  # every config folder should contain all 15

SEEDS = [3, 14, 42, 55, 89]  # the 5 runs
SEED_DIRS = [f"seed_{s}" for s in SEEDS]

# Completion markers
ENCODER_MARKER = "predict_results.json"
LLM_MARKER = "predict_results_metrics.json"


# ---------------------------------------------------------------------------

def has_marker(seed_dir: Path, marker: str) -> bool:
    """True if the marker file exists anywhere inside the seed run folder."""
    return any(seed_dir.rglob(marker))


def check(base: Path):
    problems = []
    stats = {"configs": 0, "runs_expected": 0, "runs_ok": 0}

    if not base.is_dir():
        print(f"ERROR: base directory not found: {base}")
        return 1

    for config in sorted(p for p in base.iterdir() if p.is_dir()):
        stats["configs"] += 1
        present = {p.name for p in config.iterdir() if p.is_dir()}

        # missing / extra model folders
        for m in ALL_MODELS:
            if m not in present:
                problems.append(f"[MISSING MODEL] {config.name}/{m}")
        for m in sorted(present - set(ALL_MODELS)):
            problems.append(f"[EXTRA MODEL]   {config.name}/{m}")

        # per-model seed + marker checks
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
                stats["runs_expected"] += 1
                sdir = mdir / s
                if not sdir.is_dir():
                    problems.append(f"[MISSING SEED]  {config.name}/{m}/{s}")
                elif not has_marker(sdir, marker):
                    problems.append(
                        f"[NO MARKER]     {config.name}/{m}/{s}  (need {marker})"
                    )
                else:
                    stats["runs_ok"] += 1

    # report -----------------------------------------------------------------
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
    dataset = "elecdeb60to20"
    task = "classification"
    default = root / "output" / "five-runs" / (dataset + "-" + task)
    base = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default
    sys.exit(check(base))
