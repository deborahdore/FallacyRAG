"""
Utils:

  - Knowledge-base retrieval  (BM25 + dense vector store)
  - Retrieved-passage summarisation (local LLM or GPT-4o-mini)
  - RAG context assembly
  - Compute-metrics factory for token classification
"""

from __future__ import annotations

import os
import os.path
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.create_persistent_db import EMBEDDING_MODELS
from src.utils.fallacy_loader import FallacyLoader

PERSIST_DIRECTORY = "./chromadb/"
KNOWLEDGE_BASE_PATH = "./knowledge_base/"

SEARCH_TYPES = {
    "similarity": {
        "description": "Retrieve the top-k most similar documents.",
        "search_kwargs": lambda args: {"k": args.top_k},
    },
    "mmr": {
        "description": "Maximal Marginal Relevance — balances relevance and diversity.",
        "search_kwargs": lambda args: {
            "k": args.top_k,
            "fetch_k": args.fetch_k,
            "lambda_mult": args.lambda_mult,
        },
    },
    "similarity_score_threshold": {
        "description": "Similarity search filtered to results above a score threshold.",
        "search_kwargs": lambda args: {
            "k": args.top_k,
            "score_threshold": args.score_threshold,
        },
    },
    "bm25": {
        "description": "Keyword-based BM25 retrieval — no embeddings or vector store required.",
        "search_kwargs": lambda args: {"k": args.top_k},
    },
}

_PIPELINE_CACHE: dict[str, pipeline] = {}
_VECTOR_STORE_CACHE: dict[tuple[str, str], Chroma] = {}


# ── Retrieval ─────────────────────────────────────────────────────────────────
def load_bm25_retriever(chunk_size: int, chunk_overlap: int, top_k: int) -> BM25Retriever:
    """Build a BM25Retriever directly from the source documents."""
    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=["people", "bills", "definitions", "geography"],
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    return BM25Retriever.from_documents(all_splits, k=top_k)


def load_vector_store(embeddings_model: str, persist_directory_path: str) -> Chroma:
    persist_directory_path = os.path.join(persist_directory_path, embeddings_model.replace("/", "-"))
    if not Path(persist_directory_path).exists():
        raise FileNotFoundError(
            f"No vector store found at '{persist_directory_path}'. "
            f"Run the indexing script first with the same model and chunk settings."
        )

    embedding_fn = EMBEDDING_MODELS.get(embeddings_model)

    return Chroma(
        collection_name="FallacyRAG",
        embedding_function=embedding_fn(),
        persist_directory=persist_directory_path,
    )


def search_knowledge_base(
        query: str,
        *,
        search_type: str,
        embeddings_model: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        persist_directory_path: str,
) -> list[str]:
    """Return a list of raw document strings for *query*."""
    if search_type == "bm25":
        retriever = load_bm25_retriever(chunk_size, chunk_overlap, top_k)
    else:
        search_config = SEARCH_TYPES[search_type]
        cache_key = (embeddings_model, persist_directory_path)
        if cache_key not in _VECTOR_STORE_CACHE:
            _VECTOR_STORE_CACHE[cache_key] = load_vector_store(embeddings_model, persist_directory_path)
        vector_store = _VECTOR_STORE_CACHE[cache_key]
        search_kwargs = search_config["search_kwargs"](SimpleNamespace(top_k=top_k))
        retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    docs = retriever.invoke(query)
    return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]


# ── Summarisation ─────────────────────────────────────────────────────────────

def _get_pipeline(model_name: str):
    """Return a cached text-generation pipeline for *model_name*."""
    if model_name not in _PIPELINE_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _PIPELINE_CACHE[model_name] = pipeline("text-generation", model=mdl, tokenizer=tok)
    return _PIPELINE_CACHE[model_name]


def summarize_docs(docs: list[str], summarization_model_name: str) -> list[str]:
    """Summarise each retrieved passage with the chosen model."""
    prompt_template = "Summarize the following text in a concise paragraph:\n\n{}"
    summaries: list[str] = []

    if summarization_model_name == "gpt-4o-mini":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        for doc in docs:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_template.format(doc)}],
                max_tokens=256,
            )
            summaries.append(response.choices[0].message.content.strip())
    else:
        pipe = _get_pipeline(summarization_model_name)
        tok = pipe.tokenizer
        for doc in docs:
            messages = [{"role": "user", "content": prompt_template.format(doc)}]
            formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = pipe(formatted, max_new_tokens=256, do_sample=False, return_full_text=False)
            summaries.append(output[0]["generated_text"].strip())

    return summaries


# ── RAG context assembly ──────────────────────────────────────────────────────

def build_rag_context(phrase_text: str, *, data_args, summarization_model: Optional[str]) -> str:
    """
    Retrieve and optionally summarise documents for *phrase_text*.
    Returns a single string to be appended after the phrase [SEP], or ""
    if nothing was retrieved.
    """
    passages = search_knowledge_base(
        phrase_text,
        search_type=data_args.search_type,
        embeddings_model=data_args.embeddings_model,
        chunk_size=data_args.chunk_size,
        chunk_overlap=data_args.chunk_overlap,
        top_k=data_args.top_k,
        persist_directory_path=data_args.persist_directory_path,
    )
    if not passages:
        return ""
    if summarization_model:
        passages = summarize_docs(passages, summarization_model)
    return " [SEP] ".join(passages)


# ── Metrics ───────────────────────────────────────────────────────────────────

def make_compute_metrics(label_list: list[str], eval_do_concat_batches: bool):
    """Return a compute_metrics function for token classification."""

    def compute_metrics(p):
        predictions, labels = p

        if not eval_do_concat_batches:
            predictions = np.hstack(predictions)
            labels = np.hstack(labels)

        predictions = np.argmax(predictions, axis=2)

        # Filter out padding (-100) and flatten in one pass per tensor.
        flat_preds = list(chain.from_iterable(
            [label_list[pred] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ))
        flat_labels = list(chain.from_iterable(
            [label_list[lbl] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ))

        return {
            "accuracy": accuracy_score(y_true=flat_labels, y_pred=flat_preds),
            "precision": precision_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
            "recall": recall_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
        }

    return compute_metrics
