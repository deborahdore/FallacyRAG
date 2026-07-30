"""
ablation.py
-----------
Leave-one-out ablation of the RAG pipeline, built entirely from the existing
single-seed grid in ``output/one-run/<data>-<task>/``. No new training is needed
for any cell that the grid already covers.

Design
======
The study anchors on the best RAG configuration for a task and removes one
pipeline component at a time, keeping everything else fixed:

  full            temporal + Sentence-RooseBERT + hybrid&rerank + relations_labeled
  - temporal      flip the temporal filter
  - structure     relations_labeled -> fallacy_relations -> plain sentence query
  - hybrid&rerank hybrid + cross-encoder -> dense similarity only
  - dense retr.   Sentence-RooseBERT -> bge-large / mpnet / BM25 (at the anchor's
                  query, plus a second BM25 row that also drops the query)
  - RAG           the no-retriever baseline

Two choices keep the deltas interpretable.

**The anchor model is picked on the no-RAG column** (``--anchor-by baseline``,
the default): the strongest architecture *without* retrieval. Picking it on the
RAG column instead would select the architecture where retrieval happened to
help most, and every ``delta_vs_baseline`` in the table would inherit that bias.

**Hyperparameters are frozen per model** at that model's best cell on the anchor
config, and reused unchanged for every ablated cell. Re-tuning per cell would mix
"this component was removed" with "this cell drew a luckier learning rate", and
``extract_results.py`` picks its per-cell best on the *test* metric, which would
leak selection straight into the deltas. Two selection-free columns sit alongside:

  f1_fixed   the frozen hyperparameters, applied to every cell   <- report this
  f1_mean    mean over all 12 hyperparameter cells (no selection at all)
  f1_max     max over all 12 (matches all_results.csv; for continuity only)

Note the temporal row is a removal for detection (whose winning config has
temporal=true) but an addition for classification (temporal=false) — read it as
"temporal filtering on vs off", not as a component being taken away.

Because a single seed is noisy, treat this table as the screen that decides
*which* ablated cells deserve a 5-seed re-run. ``--emit-missing`` lists the cells
that are not in the grid yet.

Usage
=====
  python script/ablation.py                                    # both tasks
  python script/ablation.py --anchor-by rag --emit-missing     # match the main table
  python script/ablation.py --all-models                       # robustness check
  python script/ablation.py --task classification --anchor-model longformer-base-4096
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ONE_RUN_ROOT = "output/one-run"
FIVE_RUN_ROOT = "output/five-runs"
RESULTS_FILE = "predict_results.json"
RESULTS_FILE_ALT = "predict_results_metrics.json"
BASELINE_CONFIG = "temporal=false-retriever=false-summarizer=false"

CONFIG_RE = re.compile(
    r"-?temporal=(?P<temporal>true|false)"
    r"-retriever=(?P<retriever>.+?)"
    r"(?:-search=(?P<search>.+?)-rerank=(?P<rerank>.+?))?"
    r"(?:-query=(?P<query>.+?))?"
    r"-summarizer=(?P<summarizer>.+)$"
)


def parse_config(name: str) -> dict:
    """Split a config folder name into its retrieval axes."""
    m = CONFIG_RE.match(name)
    if not m:
        return {k: "none" for k in ("temporal", "retriever", "search", "rerank", "query", "summarizer")}
    d = {k: (v if v is not None else "none") for k, v in m.groupdict().items()}
    d["temporal"] = d["temporal"] == "true"
    # A plain 'sentence'-mode build has no query= segment: the query is the sentence itself.
    if d["query"] == "none" and d["retriever"] != "false":
        d["query"] = "sentence"
    return d


def load_grid(task: str, data: str = "elecdeb60to20") -> pd.DataFrame:
    """Every (config, model, hyperparameter) cell of the single-seed grid."""
    root = os.path.join(ONE_RUN_ROOT, f"{data}-{task}")
    rows = []
    for config in sorted(os.listdir(root)):
        config_dir = os.path.join(root, config)
        if not os.path.isdir(config_dir):
            continue
        axes = parse_config(config)
        for model in sorted(os.listdir(config_dir)):
            model_dir = os.path.join(config_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for run in sorted(os.listdir(model_dir)):
                for fname, alt in ((RESULTS_FILE, False), (RESULTS_FILE_ALT, True)):
                    path = os.path.join(model_dir, run, fname)
                    if not os.path.isfile(path):
                        continue
                    with open(path) as fh:
                        metrics = json.load(fh)
                    rows.append({
                        "config": config, **axes, "model": model, "hparams": run,
                        "f1_macro": metrics.get("f1_macro" if alt else "predict_f1_macro"),
                        "accuracy": metrics.get("accuracy" if alt else "predict_accuracy"),
                    })
                    break
    df = pd.DataFrame(rows)
    logger.info("%s: %d cells, %d configs, %d models",
                task, len(df), df["config"].nunique(), df["model"].nunique())
    return df


def best_rag_config(task: str, anchor_by: str = "baseline",
                    data: str = "elecdeb60to20") -> tuple[str, str] | None:
    """Anchor (config, model) taken from the 5-seed aggregate.

    The config is always the proposed pipeline — the best RAG row — because the
    ablation is of *that pipeline*, not of an architecture.

    The model depends on ``anchor_by``:

    * ``baseline`` (default) — the strongest model *without* retrieval. Picking
      the anchor on the no-RAG column keeps model choice independent of the
      quantity being measured; choosing the model whose RAG score is highest
      would select for the architecture where retrieval happened to help most
      and inflate every ``delta_vs_baseline`` in the table.
    * ``rag`` — the best RAG model, i.e. the headline row of the results table.
      Matches the main table, but carries the selection bias above.
    """
    path = os.path.join(FIVE_RUN_ROOT, f"{data}-{task}", "all_results_mean_std.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    rag = df[df["config"] != BASELINE_CONFIG].sort_values("predict_f1_macro_mean", ascending=False)
    if rag.empty:
        return None
    config = rag.iloc[0]["config"]

    if anchor_by == "rag":
        return config, rag.iloc[0]["model"]
    base = df[df["config"] == BASELINE_CONFIG].sort_values("predict_f1_macro_mean", ascending=False)
    if base.empty:
        logger.warning("No baseline rows for %s — falling back to the best RAG model", task)
        return config, rag.iloc[0]["model"]
    return config, base.iloc[0]["model"]


def cell(df: pd.DataFrame, config: str, models: list[str], fixed_hparams: dict[str, str]) -> dict:
    """Fixed-hparam, mean-over-hparams and max-over-hparams F1 for one config.

    ``fixed_hparams`` is per model: a learning rate that suits ModernBERT can be
    actively bad for Longformer, so sharing one cell across architectures would
    measure hyperparameter mismatch instead of the ablated component.
    """
    sub = df[(df["config"] == config) & (df["model"].isin(models))]
    if sub.empty:
        return {"f1_fixed": float("nan"), "f1_mean": float("nan"),
                "f1_max": float("nan"), "n_hparams": 0}
    fixed = [
        g[g["hparams"] == fixed_hparams[m]]["f1_macro"].mean()
        for m, g in sub.groupby("model") if m in fixed_hparams
    ]
    # With several models the per-model values are averaged, so the row still
    # answers "what does this component do", not "which architecture is best".
    return {
        "f1_fixed": float(pd.Series(fixed).mean()) if fixed else float("nan"),
        "f1_mean": float(sub.groupby("model")["f1_macro"].mean().mean()),
        "f1_max": float(sub.groupby("model")["f1_macro"].max().mean()),
        "n_hparams": int(sub.groupby("model").size().min()),
    }


def build_variants(anchor: dict, df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(component, level, config) rows: the anchor plus every leave-one-out cell.

    Matching is done on the parsed axes rather than by string-building the folder
    name, so a variant is found whatever order create_db.py wrote the segments in.
    """
    axes = df.drop_duplicates("config").set_index("config")

    def find(**overrides) -> str | None:
        want = {**{k: anchor[k] for k in ("temporal", "retriever", "search", "rerank", "query")},
                **overrides}
        hit = axes
        for k, v in want.items():
            hit = hit[hit[k] == v]
        return hit.index[0] if len(hit) else None

    variants = [("full pipeline", "all components", anchor["config"])]

    # --- temporal filtering -------------------------------------------------
    c = find(temporal=not anchor["temporal"])
    variants.append((
        "temporal filter",
        "off" if anchor["temporal"] else "on",
        c,
    ))

    # --- structure-informed query ------------------------------------------
    for level in ("fallacy_relations", "sentence"):
        if level == anchor["query"]:
            continue
        variants.append(("query (structure)", level, find(query=level)))

    # --- fusion + reranking -------------------------------------------------
    if anchor["search"] == "hybrid":
        variants.append(("hybrid + rerank", "dense similarity only",
                         find(search="similarity", rerank="false")))

    # --- retriever ----------------------------------------------------------
    for retr in ("baai-bge-large-en-v1.5", "sentence-transformers-all-mpnet-base-v2"):
        if retr == anchor["retriever"]:
            continue
        variants.append(("dense retriever", retr, find(retriever=retr)))
    # BM25 is sparse-only (search=bm25, never reranked), but the query modes are
    # built for it too, so the swap can be made *at the anchor's query*: that row
    # isolates the retriever. The sentence-query row underneath is the older,
    # doubly-ablated cell; the gap between the two is exactly what a
    # structure-informed query is worth to a retriever that cannot embed it.
    variants.append(("dense retriever", "bm25 (sparse only)",
                     find(retriever="bm25", search="bm25", rerank="false")))
    if anchor["query"] != "sentence":
        variants.append(("dense retriever", "bm25 + sentence query",
                         find(retriever="bm25", search="bm25", rerank="false", query="sentence")))

    # --- the pipeline as a whole -------------------------------------------
    variants.append(("RAG itself", "no retrieval (baseline)", BASELINE_CONFIG))
    return variants


def ablate(task: str, anchor_model: str | None, all_models: bool,
           fixed_hparams: str | None, anchor_by: str = "baseline",
           data: str = "elecdeb60to20") -> pd.DataFrame:
    df = load_grid(task, data)

    picked = best_rag_config(task, anchor_by, data)
    anchor_config = picked[0] if picked else None
    if anchor_model is None:
        anchor_model = picked[1] if picked else None
    if anchor_config is None:
        raise SystemExit(f"No 5-seed aggregate for {task}; pass --anchor-config explicitly.")

    # Only models with the whole grid can carry an ablation; the rest were run
    # on the baseline and the winning config alone.
    coverage = df.groupby("model")["config"].nunique()
    full = coverage[coverage == df["config"].nunique()].index.tolist()
    if anchor_model not in full:
        logger.warning(
            "%s covers only %d/%d configs for %s — ablated cells must be trained first "
            "(see --emit-missing). Full-grid models: %s",
            anchor_model, coverage.get(anchor_model, 0), df["config"].nunique(), task, full,
        )
    models = full if all_models else [anchor_model]

    anchor = {"config": anchor_config, **parse_config(anchor_config)}
    # Each model keeps its own hyperparameters, chosen once on the anchor config
    # and then frozen for every ablated cell.
    if fixed_hparams is None:
        frozen = {}
        for m in models:
            rows = df[(df["config"] == anchor_config) & (df["model"] == m)]
            if rows.empty:
                logger.warning("%s has no runs in the anchor config — dropped", m)
                continue
            frozen[m] = rows.loc[rows["f1_macro"].idxmax(), "hparams"]
        if not frozen:
            raise SystemExit(f"No model has runs in {anchor_config}")
    else:
        frozen = {m: fixed_hparams for m in models}
    models = [m for m in models if m in frozen]
    logger.info("anchor: %s | frozen hparams: %s", anchor_config,
                ", ".join(f"{m}={h}" for m, h in frozen.items()))

    base = cell(df, BASELINE_CONFIG, models, frozen)
    full_cell = cell(df, anchor_config, models, frozen)

    rows = []
    for component, level, config in build_variants(anchor, df):
        if config is None:
            rows.append({"task": task, "component": component, "level": level,
                         "config": "NOT BUILT", "f1_fixed": float("nan"),
                         "f1_mean": float("nan"), "f1_max": float("nan"), "n_hparams": 0})
            continue
        c = cell(df, config, models, frozen)
        rows.append({
            "task": task, "component": component, "level": level, "config": config, **c,
            # Kept at 6 dp: the LaTeX writer rounds to one decimal of an F1 point,
            # and rounding twice can shift that last digit (0.2195 -> 21.9 vs 22.0).
            # vs the full pipeline: how much this component was contributing.
            "delta_vs_full": round(c["f1_fixed"] - full_cell["f1_fixed"], 6),
            # vs no retrieval: how much of the RAG gain survives the removal.
            "delta_vs_baseline": round(c["f1_fixed"] - base["f1_fixed"], 6),
        })
    out = pd.DataFrame(rows)
    for col in ("f1_fixed", "f1_mean", "f1_max"):
        out[col] = out[col].round(4)
    # Carried on the frame so the LaTeX writer can caption itself without
    # re-deriving which model and config the table was anchored on.
    out.attrs["anchor_model"] = ", ".join(models)
    out.attrs["anchor_config"] = anchor_config
    return out


# Component order in the paper table: the contribution first, the whole
# pipeline last, so the reader meets the structure-informed query immediately.
COMPONENT_ORDER = [
    "full pipeline", "query (structure)", "dense retriever",
    "hybrid + rerank", "temporal filter", "RAG itself",
]

COMPONENT_HEADING = {
    "query (structure)": r"\textit{Retrieval query}",
    "dense retriever": r"\textit{Retriever}",
    "hybrid + rerank": r"\textit{Fusion and reranking}",
    "temporal filter": r"\textit{Temporal filtering}",
    "RAG itself": r"\textit{Retrieval}",
}

# Kept short so nothing wraps at \columnwidth in a two-column layout; the prose
# spells out what each retrieval query mode actually contains.
LEVEL_LABEL = {
    "fallacy_relations": r"$\rightarrow$ \texttt{fallacy\_relations}",
    "sentence": r"$\rightarrow$ target sentence only",
    "dense similarity only": r"$\rightarrow$ dense similarity only",
    "baai-bge-large-en-v1.5": r"$\rightarrow$ BGE-large-en-v1.5",
    "sentence-transformers-all-mpnet-base-v2": r"$\rightarrow$ all-mpnet-base-v2",
    "bm25 (sparse only)": r"$\rightarrow$ BM25 (sparse only)",
    "bm25 + sentence query": r"\hspace{0.8em}$\hookrightarrow$ + sentence query",
    "off": r"$\rightarrow$ disabled",
    "on": r"$\rightarrow$ enabled",
    "no retrieval (baseline)": r"$\rightarrow$ none (no-RAG)",
}

TASK_LABEL = {"detection": "fallacy detection", "classification": "fallacy classification"}
TASK_COLUMN = {"detection": "Detection", "classification": "Classification"}

# The temporal filter is the one axis whose ablation is not the same edit in both
# columns: the detection pipeline has it on and the classification pipeline has it
# off, so the row is a removal on the left and an addition on the right. Collapsing
# the two into one line is what makes the tables mergeable at all; the shared label
# has to be neutral about direction, and the caption carries the asymmetry.
TEMPORAL_MERGED = "toggled"
LEVEL_LABEL[TEMPORAL_MERGED] = r"$\rightarrow$ toggled$^{\dagger}$"


def _level_key(component: str, level: str) -> str:
    """Row identity across tasks. Only the temporal row needs reconciling."""
    return TEMPORAL_MERGED if component == "temporal filter" else level


def _score(table: pd.DataFrame, component: str, key: str) -> str:
    hit = table[(table["component"] == component)
                & (table["level"].map(lambda l: _level_key(component, l)) == key)]
    if hit.empty or pd.isna(hit.iloc[0]["f1_fixed"]):
        return "--"
    return f"${hit.iloc[0]['f1_fixed'] * 100:.1f}$"


def to_latex(tables: dict[str, pd.DataFrame], anchor_models: dict[str, str],
             anchor_configs: dict[str, str]) -> str:
    """Render both tasks as one booktabs table in the repo's house style.

    One column per task, sharing a single column of row labels. Only macro-F1 is
    shown: the deltas are still in the CSV, but every row is read against the
    full-pipeline row at the top, so printing them as columns duplicated what the
    reader can already see -- and with two tasks side by side there is no room.

    Row order follows the detection column; a level that exists only for
    classification is appended after it, and a level missing from either task
    prints ``--`` rather than dropping the row.
    """
    tasks = [t for t in ("detection", "classification") if t in tables]
    ncol = len(tasks) + 1

    lines = [
        "% Ablation of the retrieval pipeline, ElecDeb60to20, both tasks.",
        "% Generated by script/ablation.py --latex.",
        "% Requires: \\usepackage{booktabs} and \\usepackage{tabularx}",
        r"\begin{table}[h]",
        r"\centering",
        # \footnotesize, not \small: two numeric columns leave the X column narrow
        # enough at \columnwidth that several labels wrapped onto a second line.
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        # tabularx rather than \resizebox: the X column absorbs the slack so the
        # table is exactly \columnwidth without rescaling the font away from the
        # body text. @{} trims the outer padding so the rules align with the margin.
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}X"
        + "r" * len(tasks) + r"@{}}",
        r"\toprule",
        r"\textbf{Configuration} & "
        + " & ".join(f"\\textbf{{{TASK_COLUMN[t]}}}" for t in tasks) + r" \\",
        r"\midrule",
        "Full pipeline & " + " & ".join(
            f"$\\mathbf{{{tables[t][tables[t]['component'] == 'full pipeline'].iloc[0]['f1_fixed'] * 100:.1f}}}$"
            for t in tasks) + r" \\",
    ]

    for component in COMPONENT_ORDER:
        if component == "full pipeline":
            continue
        keys: list[str] = []
        for t in tasks:
            sub = tables[t][tables[t]["component"] == component]
            for lvl in sub["level"]:
                k = _level_key(component, lvl)
                if k not in keys:
                    keys.append(k)
        if not keys:
            continue
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{{COMPONENT_HEADING[component]}}} \\\\")
        for key in keys:
            label = LEVEL_LABEL.get(key, key)
            scores = " & ".join(_score(tables[t], component, key) for t in tasks)
            lines.append(f"\\hspace{{0.8em}}{label} & {scores} \\\\")

    anchors = ", ".join(f"\\textsc{{{anchor_models[t]}}} ({TASK_COLUMN[t].lower()})"
                        for t in tasks)
    temporal_note = ", ".join(
        f"{'on' if parse_config(anchor_configs[t])['temporal'] else 'off'} for {TASK_COLUMN[t].lower()}"
        for t in tasks)
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        # Deliberately terse. The anchor rationale, the frozen-hyperparameter
        # protocol and the two-row BM25 decomposition are all argued in the
        # Protocol paragraph; only the dagger has to be resolved here, because
        # the symbol has no meaning outside the caption.
        f"\\caption{{Ablation on ElecDeb60to20 (macro-F1), one component changed at "
        f"a time with hyperparameters frozen. Anchors: {anchors}, each the best "
        f"model without retrieval. $^{{\\dagger}}$Toggled away from each pipeline's "
        f"own setting ({temporal_note}). Single seed.}}",
        r"\label{tab:ablation-elecdeb}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


SECTION_TEMPLATE = "paper/ablation.tex"
INPUT_LINE = r"\input{sections/ablation_tables}"

MACRO_PREAMBLE = r"""% ---------------------------------------------------------------------
% Self-contained: no \input, so it cannot fail on a missing file.
% Needs \usepackage{booktabs} and \usepackage{tabularx} in the preamble.
% Regenerate with: python script/ablation.py --section
% ---------------------------------------------------------------------
"""


def write_section(path: str, latex: list[str]) -> None:
    """Prose + tables in one file, so there is no \\input path to get wrong."""
    if not os.path.isfile(SECTION_TEMPLATE):
        logger.error("%s not found — cannot build the standalone section", SECTION_TEMPLATE)
        return
    prose = open(SECTION_TEMPLATE).read()
    if INPUT_LINE not in prose:
        logger.warning("%s no longer contains %s; appending the tables instead",
                       SECTION_TEMPLATE, INPUT_LINE)
        body = prose.rstrip() + "\n\n" + "\n".join(latex)
    else:
        body = prose.replace(INPUT_LINE, "\n".join(latex))
    # Strip the split-file header, which describes a layout this file does not use.
    body = body[body.index(r"\section"):] if r"\section" in body else body
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(MACRO_PREAMBLE + "\n" + body)
    logger.info("Wrote %s", path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", nargs="+", default=["detection", "classification"])
    p.add_argument("--data", default="elecdeb60to20")
    p.add_argument("--anchor-model", default=None,
                   help="Model to ablate; overrides --anchor-by.")
    p.add_argument("--anchor-by", choices=("baseline", "rag"), default="baseline",
                   help="Pick the anchor model on its no-RAG score (default, unbiased) "
                        "or on its RAG score (matches the main results table).")
    p.add_argument("--all-models", action="store_true",
                   help="Average over every full-grid model instead, as a robustness check.")
    p.add_argument("--fixed-hparams", default=None,
                   help="Hyperparameter cell held fixed, e.g. bs8_lr5e-5_ep5.")
    p.add_argument("--emit-missing", action="store_true",
                   help="List ablated cells that still need training.")
    p.add_argument("-o", "--output", default=None, help="Write the table to CSV.")
    p.add_argument("--latex", nargs="?", const="output/ablation_tables.tex", default=None,
                   metavar="PATH", help="Also emit booktabs tables, one per task.")
    p.add_argument("--section", nargs="?", const="paper/ablation_standalone.tex", default=None,
                   metavar="PATH",
                   help="Emit the whole section with the tables inlined: one self-contained "
                        "file with no \\input, for pasting into Overleaf.")
    args = p.parse_args()

    tables, by_task, anchor_models, anchor_configs = [], {}, {}, {}
    for task in args.task:
        t = ablate(task, args.anchor_model, args.all_models, args.fixed_hparams,
                   args.anchor_by, args.data)
        tables.append(t)
        by_task[task] = t
        anchor_models[task] = t.attrs["anchor_model"]
        anchor_configs[task] = t.attrs["anchor_config"]
        print(f"\n===== {task} =====")
        print(t.drop(columns=["config"]).to_string(index=False))
        if args.emit_missing:
            gaps = t[(t["n_hparams"] == 0) | (t["config"] == "NOT BUILT")]
            if len(gaps):
                print("\n  cells still to train:")
                for _, r in gaps.iterrows():
                    print(f"    {r['component']:<18} {r['level']:<38} {r['config']}")

    # One table for both tasks, so the reader compares the two columns row by row
    # instead of paging between two floats with identical row labels.
    latex = [to_latex(by_task, anchor_models, anchor_configs)] if args.latex else []

    if args.output:
        pd.concat(tables).to_csv(args.output, index=False)
        logger.info("Wrote %s", args.output)

    if args.latex:
        os.makedirs(os.path.dirname(args.latex) or ".", exist_ok=True)
        with open(args.latex, "w") as fh:
            fh.write("\n".join(latex))
        logger.info("Wrote %s", args.latex)

    if args.section:
        write_section(args.section, latex)


if __name__ == "__main__":
    main()
