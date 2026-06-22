"""Build detection datasets where the **retrieval query is the fallacy + its argumentative relations**.

Motivation
----------
The other pipelines query the knowledge base with a single debate *sentence*. This
variant builds a richer query from the ``*_with_relations.csv`` files: a fallacious
component is concatenated with the text of the components it is related to
(supporting / attacking premises or claims), and the **embedding of that
concatenation** is used to retrieve context.

Intuition: a bare fallacious claim ("if you pass the big drug companies' bill,
nothing will happen") is a thin query, but joined with its related component
("I'll bring new competition to bring the price down") it expresses the full
argumentative move, so the retrieved KB passages are more on-topic.

Data note
---------
In the ``*_with_relations.csv`` files the ``relations`` column holds only the
related component **ids** (e.g. ``"T6;T7"``) and ``relations_text`` holds the
corresponding **texts** — the relation *type* (support vs. attack) is **not**
recorded. We therefore concatenate the related texts without typing them; if a
typed signal becomes available it can be folded into ``build_retrieval_query``.

Retrieval engine
----------------
This script reuses the retrieval stack from ``create_db_reranked.py`` (dense, BM25,
hybrid RRF, optional cross-encoder reranking). The novelty here is purely **query
construction**; the backend is unchanged and fully local/offline.

By default it uses plain ``similarity`` retrieval — exactly the "normal" retrieval
``create_db.py`` performs — so the only difference from the base pipeline is the
relations-augmented query. ``--search-type hybrid`` and ``--rerank`` are opt-in.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

# Reuse the proven retrieval stack from the reranked pipeline (single source of truth).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_db_reranked import (  # noqa: E402
    DEFAULT_RERANKER,
    DENSE_MODELS,
    EMBEDDING_MODELS,
    KNOWLEDGE_BASE_CATEGORIES,
    SEARCH_TYPES,
    get_reranker,
    get_vector_store,
    load_bm25_retriever,
    search_knowledge_base,
    summarize_docs,
)
from src.utils.util import normalize_text  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

COMPONENT_SEP = "|||"  # used inside component_text / fallacy_text to join multiple components

QUERY_MODES = {
    "fallacy_relations": "Concatenate the fallacy text with the text of its related components (default).",
    "relations_only": "Use only the related-component text (falls back to the fallacy text if none).",
    "fallacy_only": "Use only the fallacy text — baseline for ablation.",
}


def _parse_relations_text(raw) -> list[str]:
    """Parse the ``relations_text`` cell (a stringified list) into a clean list of strings."""
    if raw is None:
        return []
    s = str(raw).strip()
    if s in ("", "nan", "[]", "None"):
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [s]
    if isinstance(parsed, (list, tuple)):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [str(parsed).strip()]


def _clean_components(text) -> list[str]:
    """Split a fallacy/component cell on the ``|||`` separator into individual spans."""
    if text is None:
        return []
    s = str(text).strip()
    if s in ("", "nan", "None"):
        return []
    return [part.strip() for part in s.split(COMPONENT_SEP) if part.strip()]


def build_retrieval_query(fallacy_text, relations_text, mode: str) -> tuple[str, bool]:
    """Construct the retrieval query for one row.

    Returns ``(query, has_relations)``. The fallacy and each related component are
    joined into a single string whose embedding is used for retrieval.
    """
    fallacy_parts = _clean_components(fallacy_text)
    relation_parts = _parse_relations_text(relations_text)
    has_relations = len(relation_parts) > 0

    if not has_relations:
        # No relation: don't concatenate anything — retrieve with the fallacy alone.
        parts = fallacy_parts
    elif mode == "fallacy_only":
        parts = fallacy_parts
    elif mode == "relations_only":
        parts = relation_parts
    else:  # fallacy_relations (default)
        parts = fallacy_parts + relation_parts

    # Join components with sentence-final punctuation so the embedding sees discrete spans.
    query = ". ".join(p.rstrip(". ") for p in parts if p)
    if query and not query.endswith("."):
        query += "."
    return normalize_text(query), has_relations


def main(persist_directory_path: str = "./chromadb",
         retriever: str = "sentence-transformers/all-mpnet-base-v2",
         search_type: str = "similarity",
         query_mode: str = "fallacy_relations",
         only_with_relations: bool = False,
         chunk_size: int = 1000,
         chunk_overlap: int = 200,
         top_k: int = 3,
         fetch_k: int = 20,
         lambda_mult: float = 0.5,
         score_threshold: float = 0.5,
         summarization_model: str | None = None,
         temporal_rag: bool = False,
         categories: list[str] | None = None,
         rerank: bool = False,
         reranker_model: str = DEFAULT_RERANKER,
         rrf_k: int = 60,
         dense_weight: float = 1.0,
         bm25_weight: float = 1.0,
         dataset_dir: str = "dataset",
         path: str = "dataset/detection"):

    # Warm the retrievers / reranker once before the per-row loop.
    if search_type == "bm25":
        logger.info("Pre-building BM25 index (this may take a while)...")
        load_bm25_retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                            top_k=top_k, categories=categories)
    elif search_type == "hybrid":
        logger.info("Pre-building BM25 index for the hybrid sparse leg...")
        load_bm25_retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                            top_k=fetch_k, categories=categories)
        get_vector_store(retriever, persist_directory_path)
    else:
        get_vector_store(retriever, persist_directory_path)
    if rerank:
        get_reranker(reranker_model)

    for split in ["train", "dev", "test"]:
        in_path = os.path.join(dataset_dir, f"{split}_with_relations.csv")
        if not os.path.exists(in_path):
            logger.warning(f"Missing {in_path}; skipping {split}.")
            continue

        df = pd.read_csv(in_path)
        logger.info(f"[{split}] loaded {len(df)} rows from {in_path}")

        enriched_rows = []
        skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Retrieving for {split}"):
            query, has_relations = build_retrieval_query(
                row.get("fallacy_text"), row.get("relations_text"), query_mode
            )

            if only_with_relations and not has_relations:
                skipped += 1
                continue
            if not query:
                skipped += 1
                continue

            copy_row = row.copy()
            copy_row["retrieval_query"] = query
            copy_row["has_relations"] = has_relations

            year = None
            try:
                year = int(row["year"])
            except (KeyError, ValueError, TypeError):
                pass

            passages = search_knowledge_base(
                query,
                search_type=search_type,
                retriever=retriever,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                chromadb_path=persist_directory_path,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                score_threshold=score_threshold,
                temporal_rag=temporal_rag,
                year=year,
                categories=categories,
                rerank=rerank,
                reranker_model=reranker_model,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
            if passages and summarization_model:
                passages = summarize_docs(passages, summarization_model)

            copy_row["rag_context"] = normalize_text(" ".join(passages))
            enriched_rows.append(copy_row)

        out_df = pd.DataFrame(enriched_rows).reset_index(drop=True)
        out_path = os.path.join(path, f"{split}.csv")
        out_df.to_csv(out_path, index=False)

        with_rel = int(out_df["has_relations"].sum()) if "has_relations" in out_df else 0
        logger.info(
            f"[{split}] wrote {len(out_df)} rows to {out_path} "
            f"({with_rel} with relations, {skipped} skipped)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build detection datasets whose retrieval query is the fallacy "
                    "concatenated with its argumentative relations."
    )
    parser.add_argument(
        "--retriever",
        default="sentence-transformers/all-mpnet-base-v2",
        choices=list(EMBEDDING_MODELS),
        help="Embedding model for the dense leg.",
    )
    parser.add_argument(
        "--search-type",
        default="similarity",
        choices=list(SEARCH_TYPES),
        help="Retrieval strategy. Default 'similarity' is plain create_db.py-style retrieval; "
             "'hybrid' (dense+BM25 RRF) and --rerank are opt-in.",
    )
    parser.add_argument(
        "--query-mode",
        default="fallacy_relations",
        choices=list(QUERY_MODES),
        help="; ".join(f"{k}: {v}" for k, v in QUERY_MODES.items()),
    )
    parser.add_argument(
        "--only-with-relations",
        action="store_true",
        default=False,
        help="Process only rows that actually have relations (skip the fallacy-only fallbacks).",
    )
    parser.add_argument("--rerank", action="store_true", default=False,
                        help="Rerank the candidate pool with a local cross-encoder.")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER,
                        help=f"Local cross-encoder for reranking (default: {DEFAULT_RERANKER}).")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF constant (default: 60).")
    parser.add_argument("--dense-weight", type=float, default=1.0,
                        help="Weight of the dense leg in RRF fusion (default: 1.0).")
    parser.add_argument("--bm25-weight", type=float, default=1.0,
                        help="Weight of the BM25 leg in RRF fusion (default: 1.0).")
    parser.add_argument(
        "--summarization-model",
        default=None,
        choices=[
            "openai/gpt-oss-20b",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen3.5-9B-Base",
        ],
        help="Optional model name to summarize retrieved passages.",
    )
    parser.add_argument("--persist-directory", default="./chromadb",
                        help="Path to the Chroma vector store directory (default: ./chromadb).")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Directory holding {train,dev,test}_with_relations.csv (default: dataset).")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="BM25 splitter chunk size in characters (default: 1000).")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="BM25 chunk overlap in characters (default: 200).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Passages kept per query after fusion/reranking (default: 3).")
    parser.add_argument("--fetch-k", type=int, default=20,
                        help="Candidate pool size per retriever for hybrid/MMR/reranking (default: 20).")
    parser.add_argument("--lambda-mult", type=float, default=0.5,
                        help="MMR diversity factor 0..1 (default: 0.5).")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Min similarity for similarity_score_threshold (default: 0.5).")
    parser.add_argument("--temporal-rag", action="store_true", default=False,
                        help="Only retrieve KB documents whose year precedes the debate year.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=KNOWLEDGE_BASE_CATEGORIES,
        choices=KNOWLEDGE_BASE_CATEGORIES,
        metavar="CATEGORY",
        help="Restrict retrieval to these knowledge-base categories (default: all).",
    )
    args = parser.parse_args()

    # ── Validation ──────────────────────────────────────────────────────────────
    if args.search_type == "hybrid" and args.retriever not in DENSE_MODELS:
        parser.error("--search-type hybrid requires a dense --retriever (not bm25).")
    if args.search_type == "bm25" and args.retriever != "bm25":
        parser.error("--search-type bm25 requires --retriever bm25.")
    if args.retriever == "bm25" and args.search_type != "bm25":
        parser.error("--retriever bm25 only supports --search-type bm25.")

    # ── Output path (mirrors the project's naming scheme, plus the query mode) ────
    path = Path("dataset") / "detection" / "agentic=false"
    path = str(path)
    path += f"-temporal={str(args.temporal_rag)}"
    path += f"-retriever={args.retriever.replace('/', '-')}"
    path += f"-search={args.search_type}"
    path += f"-rerank={args.reranker_model.replace('/', '-') if args.rerank else 'false'}"
    path += f"-query={args.query_mode}"
    if args.summarization_model:
        path += f"-summarizer={args.summarization_model.replace('/', '-')}"
    else:
        path += "-summarizer=false"
    path = str(path).lower()
    os.makedirs(path, exist_ok=True)

    main(
        persist_directory_path=args.persist_directory,
        retriever=args.retriever,
        search_type=args.search_type,
        query_mode=args.query_mode,
        only_with_relations=args.only_with_relations,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        score_threshold=args.score_threshold,
        summarization_model=args.summarization_model,
        temporal_rag=args.temporal_rag,
        categories=args.categories,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        dataset_dir=args.dataset_dir,
        path=path,
    )
