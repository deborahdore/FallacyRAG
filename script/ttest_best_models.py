"""
ttest_best_models.py
--------------------
Paired t-test between the best baseline model and the best model of every other
configuration, using the per-seed f1_macro values stored in an aggregate CSV
produced by ``extract_results_5runs.py``:

  output/five-runs/<dataset-task>/all_results_mean_std.csv

"Best" is the model with the highest ``predict_f1_macro_mean`` inside its own
config, so the two sides of a comparison may be different architectures.
Seeds are paired by position (seed 0 vs seed 0, ...), so both models must have
the same number of seeds.

Alongside the paired t-test each comparison also reports:
  * Shapiro-Wilk on the paired differences (t-test normality assumption)
  * Wilcoxon signed-rank (distribution-free alternative to the t-test)
  * exact sign-flip permutation test (no distributional assumption at all)
  * Hedges' g (Cohen's d corrected for small-sample bias)
  * a confidence interval on the mean difference at the same level as alpha

With few seeds the rank-based tests are floor-limited: Wilcoxon and the
permutation test cannot go below p = 2 / 2**n_seeds (0.0625 at 5 seeds), so
treat the effect size and the interval as the primary evidence.

A comparison is flagged significant at p < 0.01 by default.

Usage:
  python script/ttest_best_models.py <csv> [<csv> ...] [-o results.csv] [--alpha 0.01]
"""

import argparse
import itertools
import logging
import math
import os

import numpy as np
import pandas as pd
import rootutils
from scipy import stats

# Above this many seeds the 2**n sign-flip enumeration is sampled instead.
EXACT_PERM_MAX_N = 20
PERM_SAMPLES = 20000

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASELINE_CONFIGS = ("baseline", "temporal=false-retriever=false-summarizer=false")
METRIC = "predict_f1_macro"
MEAN_COL = f"{METRIC}_mean"
VALUES_COL = f"{METRIC}_values"
OUTPUT_FILE = "ttest_best_models.csv"


def parse_values(cell: str) -> list[float]:
    """Turn a space-separated '_values' cell into floats, dropping nan entries."""
    if not isinstance(cell, str):
        return []
    out = []
    for token in cell.split():
        try:
            value = float(token)
        except ValueError:
            continue
        if not math.isnan(value):
            out.append(value)
    return out


def best_row(df: pd.DataFrame) -> pd.Series:
    """Row with the highest mean f1_macro; ties broken by model name for determinism."""
    ranked = df.sort_values([MEAN_COL, "model"], ascending=[False, True])
    return ranked.iloc[0]


def shapiro_normality(diffs: list[float]) -> tuple[float, float]:
    """Shapiro-Wilk on the paired differences; a small p questions the t-test."""
    if len(diffs) < 3 or len(set(diffs)) == 1:
        return float("nan"), float("nan")
    try:
        result = stats.shapiro(diffs)
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Distribution-free paired test; undefined when every difference is zero."""
    if not any(diffs):
        return float("nan"), 1.0
    try:
        result = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def permutation_p(diffs: list[float], seed: int = 0) -> float:
    """Two-sided sign-flip test: how often a random relabelling beats |mean diff|."""
    n = len(diffs)
    observed = abs(sum(diffs) / n)
    if n <= EXACT_PERM_MAX_N:
        signs = itertools.product((1, -1), repeat=n)
        total = 2 ** n
    else:
        signs = np.random.default_rng(seed).choice((1, -1), size=(PERM_SAMPLES, n))
        total = PERM_SAMPLES
    # Tolerance guards against float noise making an identical mean look larger.
    hits = sum(1 for s in signs if abs(sum(x * d for x, d in zip(s, diffs)) / n) >= observed - 1e-12)
    return hits / total


def compare(baseline: pd.Series, other: pd.Series, alpha: float) -> dict | None:
    """Paired t-test plus non-parametric checks on the seed-level f1_macro."""
    base_values = parse_values(baseline[VALUES_COL])
    other_values = parse_values(other[VALUES_COL])

    if len(base_values) != len(other_values):
        logger.warning(
            "Seed count mismatch for config %r (%d baseline vs %d) — skipping",
            other["config"], len(base_values), len(other_values),
        )
        return None
    if len(base_values) < 2:
        logger.warning("Need at least 2 seeds for config %r — skipping", other["config"])
        return None

    result = stats.ttest_rel(other_values, base_values)
    diffs = [o - b for o, b in zip(other_values, base_values)]
    n = len(diffs)
    df = n - 1
    mean_diff = sum(diffs) / n
    # Cohen's d for paired samples: mean difference over its own std.
    diff_std = pd.Series(diffs).std(ddof=1)
    cohen_d = mean_diff / diff_std if diff_std else float("nan")
    # Hedges' correction removes the upward bias of d at small n.
    hedges_g = cohen_d * (1 - 3 / (4 * df - 1)) if df > 1 else cohen_d

    # CI on the mean difference at the same level used for the significance flag.
    sem = diff_std / math.sqrt(n) if diff_std else 0.0
    margin = stats.t.ppf(1 - alpha / 2, df) * sem
    shapiro_w, shapiro_p = shapiro_normality(diffs)
    wilcoxon_stat, wilcoxon_pval = wilcoxon_signed_rank(diffs)
    perm_pval = permutation_p(diffs)
    perm_floor = 2 / 2 ** n if n <= EXACT_PERM_MAX_N else 1 / PERM_SAMPLES

    return {
        "task": other.get("task", ""),
        "baseline_config": baseline["config"],
        "baseline_model": baseline["model"],
        "baseline_f1_mean": round(float(baseline[MEAN_COL]), 3),
        "config": other["config"],
        "model": other["model"],
        "f1_mean": round(float(other[MEAN_COL]), 3),
        "n_seeds": n,
        "mean_diff": round(mean_diff, 4),
        "ci_level": round(1 - alpha, 3),
        "ci_low": round(mean_diff - margin, 4),
        "ci_high": round(mean_diff + margin, 4),
        "t_stat": round(float(result.statistic), 4),
        "p_value": round(float(result.pvalue), 5),
        "df": df,
        "cohen_d": round(cohen_d, 3),
        "hedges_g": round(hedges_g, 3),
        "shapiro_w": round(shapiro_w, 3),
        "shapiro_p": round(shapiro_p, 5),
        "wilcoxon_stat": round(wilcoxon_stat, 3),
        "wilcoxon_p": round(wilcoxon_pval, 5),
        "perm_p": round(perm_pval, 5),
        # Floor of the rank-based tests; they cannot detect anything below it.
        "min_attainable_p": round(perm_floor, 5),
        "significant": bool(result.pvalue < alpha),
        # Significant by t-test and as extreme as the seed count lets the rank tests go.
        "robust": bool(
            result.pvalue < alpha
            and perm_pval <= max(alpha, perm_floor)
            and (math.isnan(wilcoxon_pval) or wilcoxon_pval <= max(alpha, perm_floor))
        ),
    }


def process_csv(csv_path: str, alpha: float) -> list[dict]:
    df = pd.read_csv(csv_path)

    missing = [c for c in ("config", "model", MEAN_COL, VALUES_COL) if c not in df.columns]
    if missing:
        logger.error("%s is missing columns %s — skipping", csv_path, missing)
        return []

    baseline_df = df[df["config"].isin(BASELINE_CONFIGS)]
    if baseline_df.empty:
        logger.error("No baseline config in %s — skipping", csv_path)
        return []

    baseline = best_row(baseline_df)
    logger.info(
        "%s: baseline best is %s (f1=%.3f)",
        os.path.basename(os.path.dirname(csv_path)) or csv_path,
        baseline["model"], baseline[MEAN_COL],
    )

    rows = []
    for config_name, group in df[~df["config"].isin(BASELINE_CONFIGS)].groupby("config"):
        row = compare(baseline, best_row(group), alpha)
        if row is not None:
            rows.append(row)
    if not rows:
        logger.warning("No comparable non-baseline config in %s", csv_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="aggregate all_results_mean_std.csv file(s)")
    parser.add_argument("-o", "--output", help=f"output CSV (default: {OUTPUT_FILE} next to each input)")
    parser.add_argument("--alpha", type=float, default=0.01, help="significance level (default: 0.01)")
    args = parser.parse_args()

    all_rows = []
    for csv_path in args.csv:
        if not os.path.isfile(csv_path):
            logger.error("File not found: %s", csv_path)
            continue
        rows = process_csv(csv_path, args.alpha)
        all_rows.extend(rows)
        # Without -o, each input gets its own report alongside it.
        if rows and not args.output:
            out_path = os.path.join(os.path.dirname(csv_path), OUTPUT_FILE)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            logger.info("Wrote %d comparisons → %s", len(rows), out_path)

    if not all_rows:
        raise SystemExit(1)

    if args.output:
        pd.DataFrame(all_rows).to_csv(args.output, index=False)
        logger.info("Wrote %d comparisons → %s", len(all_rows), args.output)

    print(pd.DataFrame(all_rows).to_string(index=False, max_colwidth=30))


if __name__ == "__main__":
    main()
