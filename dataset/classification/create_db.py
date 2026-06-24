"""Build BIO-tag fallacy-classification datasets enriched with RAG context.

It supports two **dataset-construction modes** (``--mode``) sharing one retrieval backend:

* ``sentence`` (default) — read ``{split}.csv``, spaCy sentence-split each speech
  turn, balance/downsample the NoFallacy class, re-split into train/dev/test, and
  enrich every sentence. With no ``--retriever`` this is the plain no-RAG pipeline;
  with a dense/bm25/hybrid retriever and optional reranking it is the enriched pipeline.

* ``relations`` — identical dataset construction to ``sentence`` (same sentence
  splitting, NoFallacy downsampling, and stratified re-split). The *only* difference
  is the retrieval query: for each fallacy sentence the query is its ``fallacy_text``
  concatenated with its related components' ``relations_text`` (looked up from
  ``{split}_with_relations.csv`` by speech turn); NoFallacy or unmatched sentences
  fall back to the bare sentence.

Retrieval backend (shared by both modes)
-----------------------------------------
* Dense vector search over a persisted Chroma store (``similarity`` / ``mmr`` /
  ``similarity_score_threshold``).
* Sparse BM25 via bm25s (scipy-backed, vectorized).
* ``hybrid`` — dense + BM25 fused with Reciprocal Rank Fusion (no score
  calibration needed; a passage need only rank well in one leg to surface).
* Optional local cross-encoder reranking of the candidate pool.
* Optional per-passage summarization.
* Category and temporal (year) filtering on both retrieval legs.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path

import nltk
import numpy as np
import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt_tab", quiet=True)

import pandas as pd
import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import bm25s as _bm25s
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils.fallacy_loader import FallacyLoader
from src.utils.util import normalize_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
KNOWLEDGE_BASE_PATH = "./knowledge_base/"

KNOWLEDGE_BASE_CATEGORIES = [
    "bills",
    "definitions",
    "papers",
    "exemplars",
    "geography",
    "history",
    "newspapers",
    "people",
]

CATEGORY_TO_DIRECTORY = {
    "bills": "bills",
    "definitions": "definitions",
    "papers": ("definitions/papers", "papers"),
    "exemplars": "exemplars",
    "geography": "geography",
    "history": "history",
    "newspapers": "newspapers",
    "people": "people",
}

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_vector_store_cache: dict[tuple[str, str], Chroma] = {}
_bm25_cache: dict[tuple, "FastBM25Retriever"] = {}
_pipeline_cache: dict[str, pipeline] = {}
_reranker_cache: dict[str, "CrossEncoderReranker"] = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {"device": DEVICE}
encode_kwargs = {"batch_size": 2048, "normalize_embeddings": True}

EMBEDDING_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    ),
    "ddore14/Sentence-RooseBERT": lambda: HuggingFaceEmbeddings(
        model_name="ddore14/Sentence-RooseBERT",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    ),
    "BAAI/bge-large-en-v1.5": lambda: HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    ),
    "bm25": "bm25-as-retriever",
}

DENSE_MODELS = [m for m in EMBEDDING_MODELS if m != "bm25"]

SEARCH_TYPES = {
    "similarity": "Retrieve the top-k most similar documents.",
    "mmr": "Maximal Marginal Relevance — balances relevance and diversity.",
    "similarity_score_threshold": "Similarity search filtered to results above a score threshold.",
    "bm25": "Keyword-based BM25 retrieval — no embeddings or vector store required.",
    "hybrid": "Dense + BM25 fused with Reciprocal Rank Fusion (recommended).",
}

DATASET_MODES = {
    "sentence": "Sentence-split speech turns from {split}.csv, balance, re-split, enrich "
                "(retrieval query = the sentence).",
    "relations": "Same construction as 'sentence' (split, balance, re-split); the only "
                 "difference is the retrieval query = fallacy_text + relations_text "
                 "(from {split}_with_relations.csv).",
}

COMPONENT_SEP = "|||"  # joins multiple components inside fallacy_text / relations cells

QUERY_MODES = {
    "fallacy_relations": "Concatenate the fallacy text with the text of its related components (default).",
    "relations_only": "Use only the related-component text (falls back to the fallacy text if none).",
    "fallacy_only": "Use only the fallacy text — baseline for ablation.",
}


class FastBM25Retriever:
    def __init__(self, docs: list, k: int = 5):
        self.docs = docs
        self.k = k
        corpus = [d.page_content for d in docs]
        tokenized = _bm25s.tokenize(corpus, show_progress=False)
        self._index = _bm25s.BM25()
        self._index.index(tokenized)

    def invoke(self, query: str) -> list:
        tokenized_query = _bm25s.tokenize([query], show_progress=False)
        k = min(self.k, len(self.docs))
        results, _ = self._index.retrieve(tokenized_query, k=k)
        return [self.docs[int(i)] for i in results[0]]


def load_bm25_retriever(chunk_size: int, chunk_overlap: int, top_k: int,
                        categories: list[str] | None = None) -> FastBM25Retriever:
    cats = list(categories) if categories else list(KNOWLEDGE_BASE_CATEGORIES)
    directories = [CATEGORY_TO_DIRECTORY[c] for c in cats]
    cache_key = (chunk_size, chunk_overlap, tuple(sorted(cats)))
    if cache_key not in _bm25_cache:
        loader = FallacyLoader(source_path=KNOWLEDGE_BASE_PATH, directories=directories)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        chunks = splitter.split_documents(loader.load())
        _bm25_cache[cache_key] = FastBM25Retriever(chunks, k=top_k)
    else:
        _bm25_cache[cache_key].k = top_k
    return _bm25_cache[cache_key]


def get_vector_store(retriever: str, chromadb_path: str) -> Chroma:
    store_path = os.path.join(chromadb_path, retriever.replace("/", "-"))
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Chroma store not found at {store_path!r}. Run create_persistent_db.py first."
        )
    cache_key = (retriever, store_path)
    if cache_key not in _vector_store_cache:
        logger.info(f"Loading vector store for {retriever} from {store_path}")
        _vector_store_cache[cache_key] = Chroma(
            collection_name="FallacyRAG",
            embedding_function=EMBEDDING_MODELS[retriever](),
            persist_directory=store_path,
        )
    return _vector_store_cache[cache_key]


def _build_chroma_filter(temporal_rag: bool, year: int | None,
                         categories: list[str] | None) -> dict | None:
    filters = []
    if temporal_rag and year is not None:
        filters.append({"year": {"$lt": int(year)}})
    if categories:
        filters.append({"category": {"$in": list(categories)}})
    if len(filters) == 1:
        return filters[0]
    if len(filters) > 1:
        return {"$and": filters}
    return None


def _dense_candidates(query: str, retriever: str, chromadb_path: str, k: int,
                      search_type: str, fetch_k: int, lambda_mult: float,
                      score_threshold: float, temporal_rag: bool,
                      year: int | None, categories: list[str] | None) -> list:
    vector_store = get_vector_store(retriever, chromadb_path)
    where = _build_chroma_filter(temporal_rag, year, categories)

    effective_type = "similarity" if search_type == "hybrid" else search_type
    kwargs = {"k": k}
    if effective_type == "mmr":
        kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
    elif effective_type == "similarity_score_threshold":
        kwargs.update({"score_threshold": score_threshold})
    if where:
        kwargs["filter"] = where

    ret = vector_store.as_retriever(search_type=effective_type, search_kwargs=kwargs)
    return ret.invoke(query)


def _bm25_candidates(query: str, chunk_size: int, chunk_overlap: int, k: int,
                     temporal_rag: bool, year: int | None,
                     categories: list[str] | None) -> list:
    fetch_count = k * 10 if temporal_rag else k
    bm25 = load_bm25_retriever(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=fetch_count, categories=categories
    )
    docs = bm25.invoke(query)
    if temporal_rag and year is not None:
        docs = [d for d in docs if int(d.metadata.get("year", year + 1)) < year]
    return docs[:k]


def reciprocal_rank_fusion(ranked_lists: list[list], rrf_k: int = 60,
                           weights: list[float] | None = None) -> list:
    """Merge several ranked document lists by Reciprocal Rank Fusion.

    Documents are identified by exact page_content; a passage appearing in more than
    one list accumulates score from each. Returns documents ordered by fused score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    doc_by_key: dict[str, object] = {}
    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank + 1))
            doc_by_key.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [doc_by_key[k] for k in ordered]


class CrossEncoderReranker:
    """Local cross-encoder that re-scores (query, passage) pairs jointly."""

    def __init__(self, model_name: str, device: str):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder reranker {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: list, top_k: int) -> list:
        if not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs, batch_size=64, show_progress_bar=False)
        order = np.argsort(scores)[::-1][:top_k]
        return [docs[int(i)] for i in order]


def get_reranker(model_name: str) -> CrossEncoderReranker:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoderReranker(model_name, DEVICE)
    return _reranker_cache[model_name]


def search_knowledge_base(query: str,
                          search_type: str,
                          retriever: str,
                          chunk_size: int,
                          chunk_overlap: int,
                          top_k: int,
                          chromadb_path: str,
                          fetch_k: int = 20,
                          lambda_mult: float = 0.5,
                          score_threshold: float = 0.5,
                          temporal_rag: bool = False,
                          year: int | None = None,
                          categories: list[str] | None = None,
                          rerank: bool = False,
                          reranker_model: str = DEFAULT_RERANKER,
                          rrf_k: int = 60,
                          dense_weight: float = 1.0,
                          bm25_weight: float = 1.0) -> list[str]:
    """Return a list of passage strings relevant to *query*.

    ``search_type='hybrid'`` fuses a dense and a BM25 candidate pool with RRF; any
    search type can additionally be reranked by a local cross-encoder when
    ``rerank=True``. When reranking, each leg fetches ``fetch_k`` candidates and the
    reranker trims to ``top_k``; otherwise the pools are sized to ``top_k`` directly.

    If *categories* is given, retrieval is restricted to those knowledge-base
    categories; otherwise all categories are searched.
    """
    pool = fetch_k if (rerank or search_type == "hybrid") else top_k

    if search_type == "hybrid":
        dense = _dense_candidates(
            query, retriever, chromadb_path, k=pool, search_type="hybrid",
            fetch_k=fetch_k, lambda_mult=lambda_mult, score_threshold=score_threshold,
            temporal_rag=temporal_rag, year=year, categories=categories,
        )
        sparse = _bm25_candidates(
            query, chunk_size, chunk_overlap, k=pool,
            temporal_rag=temporal_rag, year=year, categories=categories,
        )
        candidates = reciprocal_rank_fusion(
            [dense, sparse], rrf_k=rrf_k, weights=[dense_weight, bm25_weight]
        )
    elif search_type == "bm25":
        candidates = _bm25_candidates(
            query, chunk_size, chunk_overlap, k=pool,
            temporal_rag=temporal_rag, year=year, categories=categories,
        )
    else:
        candidates = _dense_candidates(
            query, retriever, chromadb_path, k=pool, search_type=search_type,
            fetch_k=fetch_k, lambda_mult=lambda_mult, score_threshold=score_threshold,
            temporal_rag=temporal_rag, year=year, categories=categories,
        )

    if rerank:
        candidates = get_reranker(reranker_model).rerank(query, candidates, top_k)
    else:
        candidates = candidates[:top_k]

    return [d.page_content if hasattr(d, "page_content") else str(d) for d in candidates]


def warm_retrievers(search_type: str, retriever: str | None, chunk_size: int,
                    chunk_overlap: int, top_k: int, fetch_k: int,
                    categories: list[str] | None, rerank: bool,
                    reranker_model: str, chromadb_path: str) -> None:
    """Pre-build BM25 index / load vector store / load reranker before the per-row loop."""
    if search_type == "bm25":
        logger.info("Pre-building BM25 index (this may take a while)...")
        load_bm25_retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                            top_k=top_k, categories=categories)
        logger.info("BM25 index ready.")
    elif search_type == "hybrid":
        logger.info("Pre-building BM25 index for the hybrid sparse leg...")
        load_bm25_retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                            top_k=fetch_k, categories=categories)
        get_vector_store(retriever, chromadb_path)
        logger.info("Hybrid retrievers ready.")
    elif retriever and retriever != "bm25":
        get_vector_store(retriever, chromadb_path)
    if rerank:
        get_reranker(reranker_model)  # warm the cross-encoder once


def summarize_docs(docs: list[str], model_name: str, max_tokens: int = 256) -> list[str]:
    """Summarize each passage with the given model. Returns one summary per doc."""
    prompt_template = "Summarize the following text in a concise paragraph:\n\n{}"
    summaries: list[str] = []

    if model_name == "gpt-4o-mini":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        for doc in docs:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_template.format(doc)}],
                max_tokens=max_tokens,
            )
            summaries.append(response.choices[0].message.content.strip())
    else:
        if model_name not in _pipeline_cache:
            logger.info(f"Loading summarization model {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            _pipeline_cache[model_name] = pipeline("text-generation", model=model, tokenizer=tokenizer)
        pipe = _pipeline_cache[model_name]

        formatted_list = [
            pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_template.format(doc)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for doc in docs
        ]
        outputs = pipe(
            formatted_list,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
            batch_size=len(docs),
        )
        summaries = [o[0]["generated_text"].strip() for o in outputs]

    return summaries


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
    """Construct the relations-augmented retrieval query for one row.

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


def _build_balanced_splits(dataset_dir: str) -> dict[str, pd.DataFrame]:
    """Read {split}.csv, spaCy sentence-split each speech turn, balance the NoFallacy
    class by downsampling, and re-split into stratified train/dev/test.
    """
    dataset = []
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(dataset_dir, f"{split}.csv"))
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} dataset."):
            text = row["speech_turn"]

            fallacy_labels = ast.literal_eval(row["fallacy_labels"])
            component_labels = ast.literal_eval(row["components_labels"])

            if len(text) != len(fallacy_labels):
                logger.warning(
                    f"Row {idx} ({split}): text length {len(text)} != "
                    f"fallacy_labels length {len(fallacy_labels)} — skipping."
                )
                continue

            doc = nlp(text)

            for sent in doc.sents:
                start = sent.start_char
                end = sent.end_char

                flabels = list(set(label for label in fallacy_labels[start:end] if label != 'O'))
                flabels = flabels[0] if flabels and flabels[0] is not None else "NoFallacy"

                clabels = list(set(label for label in component_labels[start:end] if label != 'O'))
                clabels = clabels[0] if clabels and clabels[0] is not None else "NoComponent"

                item = {
                    "text": f"{text} [SEP] {sent}",
                    "fallacy_label": flabels,
                    "component_label": clabels,
                    "speaker": row["speaker"],
                    "year": row["year"],
                    "sentence": text[start:end],
                    "speech_turn": text,
                }
                dataset.append(item)

    df = pd.DataFrame(dataset)
    df = df.replace({"nan": None})
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["fallacy_label"])
    test_df, dev_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df["fallacy_label"])

    return {"train": train_df, "dev": dev_df, "test": test_df}


def _enrich_split(split_df: pd.DataFrame, split_name: str, query_fn,
                  *, retriever, search_type, chunk_size, chunk_overlap, top_k, fetch_k,
                  lambda_mult, score_threshold, temporal_rag, categories, rerank,
                  reranker_model, rrf_k, dense_weight, bm25_weight, summarization_model,
                  persist_directory_path, record_query: bool) -> pd.DataFrame:
    """Attach a ``rag_context`` column to each row. ``query_fn(row)`` returns
    ``(query, has_relations)`` and is the only thing that differs between modes."""
    enriched_rows = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                       desc=f"Enriching {split_name} with RAG context"):
        copy_row = row.copy()
        query, has_relations = query_fn(row)
        if record_query:
            copy_row["retrieval_query"] = query
            copy_row["has_relations"] = bool(has_relations)

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
            year=int(row["year"]),
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

    return pd.DataFrame(enriched_rows).reset_index(drop=True)


def _run_pipeline(splits: dict[str, pd.DataFrame], query_fn, *, retriever, path,
                  record_query: bool, **enrich_kwargs) -> None:
    if retriever:
        warm_retrievers(
            enrich_kwargs["search_type"], retriever, enrich_kwargs["chunk_size"],
            enrich_kwargs["chunk_overlap"], enrich_kwargs["top_k"], enrich_kwargs["fetch_k"],
            enrich_kwargs["categories"], enrich_kwargs["rerank"], enrich_kwargs["reranker_model"],
            enrich_kwargs["persist_directory_path"],
        )
        for name in ("train", "dev", "test"):
            splits[name] = _enrich_split(
                splits[name], name, query_fn,
                retriever=retriever, record_query=record_query, **enrich_kwargs
            )

    splits["train"].to_csv(f"{path}/train.csv", index=False)
    splits["test"].to_csv(f"{path}/test.csv", index=False)
    splits["dev"].to_csv(f"{path}/dev.csv", index=False)

    _print_dataset_statistics(splits["train"], splits["test"], splits["dev"])


def build_sentence_mode(persist_directory_path: str,
                        retriever: str | None,
                        search_type: str,
                        chunk_size: int,
                        chunk_overlap: int,
                        top_k: int,
                        fetch_k: int,
                        lambda_mult: float,
                        score_threshold: float,
                        summarization_model: str | None,
                        temporal_rag: bool,
                        categories: list[str] | None,
                        rerank: bool,
                        reranker_model: str,
                        rrf_k: int,
                        dense_weight: float,
                        bm25_weight: float,
                        dataset_dir: str,
                        path: str) -> None:
    splits = _build_balanced_splits(dataset_dir)

    def query_fn(row):
        return normalize_text(row["sentence"]), None

    _run_pipeline(
        splits, query_fn, retriever=retriever, path=path, record_query=False,
        search_type=search_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        top_k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult, score_threshold=score_threshold,
        temporal_rag=temporal_rag, categories=categories, rerank=rerank,
        reranker_model=reranker_model, rrf_k=rrf_k, dense_weight=dense_weight,
        bm25_weight=bm25_weight, summarization_model=summarization_model,
        persist_directory_path=persist_directory_path,
    )


def _load_relations_lookup(dataset_dir: str) -> dict[str, list[tuple]]:
    """Map ``speech_turn`` -> list of ``(fallacy_text, relations_text)`` from the
    ``{split}_with_relations.csv`` files (rows without a fallacy span are ignored)."""
    lookup: dict[str, list[tuple]] = {}
    for split in ["train", "dev", "test"]:
        p = os.path.join(dataset_dir, f"{split}_with_relations.csv")
        if not os.path.exists(p):
            logger.warning(f"Missing {p}; relations for these rows will fall back to the sentence.")
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            ft = r.get("fallacy_text")
            if ft is None or str(ft).strip() in ("", "nan", "None"):
                continue
            lookup.setdefault(str(r["speech_turn"]), []).append((ft, r.get("relations_text")))
    return lookup


def _match_relations(sentence: str, speech_turn, lookup: dict[str, list[tuple]]):
    """Find the fallacy entry whose span overlaps *sentence* within the same speech
    turn. Returns ``(fallacy_text, relations_text)`` or ``(None, None)`` if unmatched."""
    s = str(sentence).strip()
    if not s:
        return None, None
    for fallacy_text, relations_text in lookup.get(str(speech_turn), []):
        for comp in _clean_components(fallacy_text):
            if s in comp or comp in s:
                return fallacy_text, relations_text
    return None, None


def build_relations_mode(persist_directory_path: str,
                         retriever: str,
                         search_type: str,
                         query_mode: str,
                         chunk_size: int,
                         chunk_overlap: int,
                         top_k: int,
                         fetch_k: int,
                         lambda_mult: float,
                         score_threshold: float,
                         summarization_model: str | None,
                         temporal_rag: bool,
                         categories: list[str] | None,
                         rerank: bool,
                         reranker_model: str,
                         rrf_k: int,
                         dense_weight: float,
                         bm25_weight: float,
                         dataset_dir: str,
                         path: str) -> None:
    splits = _build_balanced_splits(dataset_dir)
    lookup = _load_relations_lookup(dataset_dir)

    def query_fn(row):
        sentence = row["sentence"]
        if row["fallacy_label"] == "NoFallacy":
            return normalize_text(sentence), False
        fallacy_text, relations_text = _match_relations(sentence, row["speech_turn"], lookup)
        if fallacy_text is None:
            # Fallacy sentence with no matched relation row: fall back to the sentence.
            return normalize_text(sentence), False
        query, has_relations = build_retrieval_query(fallacy_text, relations_text, query_mode)
        if not query:
            return normalize_text(sentence), False
        return query, has_relations

    _run_pipeline(
        splits, query_fn, retriever=retriever, path=path, record_query=True,
        search_type=search_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        top_k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult, score_threshold=score_threshold,
        temporal_rag=temporal_rag, categories=categories, rerank=rerank,
        reranker_model=reranker_model, rrf_k=rrf_k, dense_weight=dense_weight,
        bm25_weight=bm25_weight, summarization_model=summarization_model,
        persist_directory_path=persist_directory_path,
    )


def _print_dataset_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame, dev_df: pd.DataFrame) -> None:
    """Print size and label distribution statistics for each dataset split."""
    splits = {"train": train_df, "dev": dev_df, "test": test_df}

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)

    for name, df in splits.items():
        logger.info(f"\n--- {name.upper()} ({len(df):,} samples) ---")

        logger.info("  Fallacy label distribution:")
        fallacy_counts = df["fallacy_label"].value_counts()
        fallacy_pct = df["fallacy_label"].value_counts(normalize=True) * 100
        for label in fallacy_counts.index:
            logger.info(f"    {label:<30s} {fallacy_counts[label]:>6,}  ({fallacy_pct[label]:.1f}%)")

        logger.info("  Component label distribution:")
        component_counts = df["component_label"].value_counts()
        component_pct = df["component_label"].value_counts(normalize=True) * 100
        for label in component_counts.index:
            logger.info(f"    {label:<30s} {component_counts[label]:>6,}  ({component_pct[label]:.1f}%)")

    total = sum(len(df) for df in splits.values())
    logger.info(f"\nTotal samples across all splits: {total:,}")
    logger.info("=" * 60)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build BIO-tag fallacy-classification datasets with optional (hybrid / "
                    "reranked) RAG context. Use --mode to pick the dataset-construction path."
    )
    parser.add_argument(
        "--mode",
        default="sentence",
        choices=list(DATASET_MODES),
        help="; ".join(f"{k}: {v}" for k, v in DATASET_MODES.items()),
    )
    parser.add_argument(
        "--retriever",
        default=None,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model for the dense leg. In 'sentence' mode, omit it for a "
             "no-RAG dataset. Required for hybrid/dense search and for 'relations' mode.",
    )
    parser.add_argument(
        "--search-type",
        default="similarity",
        choices=list(SEARCH_TYPES),
        help="Retrieval strategy (default: similarity).",
    )
    parser.add_argument(
        "--query-mode",
        default="fallacy_relations",
        choices=list(QUERY_MODES),
        help="(relations mode only) " + "; ".join(f"{k}: {v}" for k, v in QUERY_MODES.items()),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Rerank the candidate pool with a local cross-encoder before keeping top-k.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER,
        help=f"Local cross-encoder for reranking (default: {DEFAULT_RERANKER}). "
             "E.g. BAAI/bge-reranker-base or BAAI/bge-reranker-large for higher quality.",
    )
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF constant; larger values flatten top-rank contribution (default: 60).")
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
                        help="Directory holding the input CSVs (default: dataset).")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="BM25 text splitter chunk size in characters (default: 1000).")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Overlap between consecutive BM25 chunks in characters (default: 200).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Passages kept per query after fusion/reranking (default: 3).")
    parser.add_argument("--fetch-k", type=int, default=20,
                        help="Candidate pool size per retriever for hybrid/MMR/reranking (default: 20).")
    parser.add_argument("--lambda-mult", type=float, default=0.5,
                        help="MMR diversity factor between 0 (max diversity) and 1 (max relevance) (default: 0.5).")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Minimum similarity score for similarity_score_threshold retrieval (default: 0.5).")
    parser.add_argument("--temporal-rag", action="store_true", default=False,
                        help="Only retrieve knowledge base documents whose year is before the debate year.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=KNOWLEDGE_BASE_CATEGORIES,
        choices=KNOWLEDGE_BASE_CATEGORIES,
        metavar="CATEGORY",
        help="Restrict retrieval to these knowledge-base categories "
             f"(choices: {', '.join(KNOWLEDGE_BASE_CATEGORIES)}; default: all).",
    )
    return parser


def _validate(args, parser) -> None:
    if args.search_type == "hybrid" and args.retriever not in DENSE_MODELS:
        parser.error("--search-type hybrid requires a dense --retriever (not bm25).")
    if args.search_type == "bm25" and args.retriever != "bm25":
        parser.error("--search-type bm25 requires --retriever bm25.")
    if args.retriever == "bm25" and args.search_type != "bm25":
        parser.error("--retriever bm25 only supports --search-type bm25.")
    if args.temporal_rag:
        assert args.retriever is not None, "--temporal-rag requires a --retriever."
    if args.mode == "relations" and args.retriever is None:
        parser.error("--mode relations requires a --retriever.")


def _build_output_path(args) -> str:
    path = str(Path("dataset") / "classification" / "agentic=false")
    path += f"-temporal={str(args.temporal_rag)}"
    if args.retriever:
        path += f"-retriever={args.retriever.replace('/', '-')}"
        path += f"-search={args.search_type}"
        path += f"-rerank={args.reranker_model.replace('/', '-') if args.rerank else 'false'}"
    else:
        path += "-retriever=false"
    if args.mode == "relations":
        path += f"-query={args.query_mode}"
    if args.summarization_model:
        path += f"-summarizer={args.summarization_model.replace('/', '-')}"
    else:
        path += "-summarizer=false"
    return path.lower()


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate(args, parser)

    if args.categories and set(args.categories) == set(KNOWLEDGE_BASE_CATEGORIES):
        args.categories = None

    path = _build_output_path(args)
    os.makedirs(path, exist_ok=True)

    if args.mode == "sentence":
        build_sentence_mode(
            persist_directory_path=args.persist_directory,
            retriever=args.retriever,
            search_type=args.search_type,
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
    else:  # relations
        build_relations_mode(
            persist_directory_path=args.persist_directory,
            retriever=args.retriever,
            search_type=args.search_type,
            query_mode=args.query_mode,
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
