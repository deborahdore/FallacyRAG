from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from types import SimpleNamespace

import nltk
import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt_tab", quiet=True)

import pandas as pd
import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, CohereEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings
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
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
KNOWLEDGE_BASE_PATH = "./knowledge_base/"

_vector_store_cache: dict[tuple[str, str], Chroma] = {}
_bm25_cache: dict[tuple[int, int], BM25Retriever] = {}
_pipeline_cache: dict[str, pipeline] = {}

model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
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
    "llama3": lambda: OllamaEmbeddings(model="llama3"),
    "embed-english-v3.0": lambda: CohereEmbeddings(model="embed-english-v3.0"),
    "mistral-embed": lambda: MistralAIEmbeddings(model="mistral-embed"),
    "bm25": "bm25-as-retriever",
}

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
                          year: int | None = None):
    """Return a list of passage strings relevant to *query*."""

    if search_type == "bm25":
        # Fetch more candidates when filtering temporally, since some will be dropped.
        fetch_count = top_k * 10 if temporal_rag else top_k
        bm25 = load_bm25_retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=fetch_count)
        docs = bm25.invoke(query)
        if temporal_rag and year is not None:
            docs = [d for d in docs if int(d.metadata.get("year", year + 1)) < year]
            docs = docs[:top_k]
    else:
        store_path = os.path.join(chromadb_path, retriever.replace("/", "-"))
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Chroma store not found at {store_path!r}. Run create_persistent_db.py first.")

        cache_key = (retriever, store_path)
        if cache_key not in _vector_store_cache:
            logger.info(f"Loading vector store for {retriever} from {store_path}")
            _vector_store_cache[cache_key] = Chroma(
                collection_name="FallacyRAG",
                embedding_function=EMBEDDING_MODELS[retriever](),
                persist_directory=store_path,
            )
        vector_store = _vector_store_cache[cache_key]
        search_kwargs = SEARCH_TYPES[search_type]["search_kwargs"](
            SimpleNamespace(
                top_k=top_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                score_threshold=score_threshold,
            )
        )
        if temporal_rag and year is not None:
            search_kwargs["filter"] = {"year": {"$lt": int(year)}}
        ret = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        docs = ret.invoke(query)

    return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]


def load_bm25_retriever(chunk_size: int, chunk_overlap: int, top_k: int) -> BM25Retriever:
    """Build a BM25 retriever directly from the knowledge base source documents."""
    cache_key = (chunk_size, chunk_overlap)
    if cache_key not in _bm25_cache:
        loader = FallacyLoader(
            source_path=KNOWLEDGE_BASE_PATH,
            directories=["people", "bills", "definitions", "geography"],
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        chunks = splitter.split_documents(loader.load())
        _bm25_cache[cache_key] = BM25Retriever.from_documents(chunks, k=top_k)
    else:
        _bm25_cache[cache_key].k = top_k
    return _bm25_cache[cache_key]


def summarize_docs(docs: list[str], model_name: str, max_tokens: int = 1024) -> list[str]:
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
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
            )
            _pipeline_cache[model_name] = pipeline("text-generation", model=model, tokenizer=tokenizer)
        pipe = _pipeline_cache[model_name]

        for doc in docs:
            messages = [{"role": "user", "content": prompt_template.format(doc)}]
            formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = pipe(formatted, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)
            summaries.append(output[0]["generated_text"].strip())

    return summaries


def main(persist_directory_path: str = "./chromadb",
         retriever: str = None,
         search_type: str = "similarity",
         chunk_size: int = 1000,
         chunk_overlap: int = 200,
         top_k: int = 3,
         fetch_k: int = 20,
         lambda_mult: float = 0.5,
         score_threshold: float = 0.5,
         summarization_model: str | None = None,
         temporal_rag: bool = False,
         path: str = "dataset/detection"):
    dataset = []
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(f"dataset/{split}.csv")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} dataset."):
            text = row["speech_turn"]

            # Parse label lists
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
                # flabels = flabels[0] if flabels and flabels[0] is not None else "NoFallacy"
                flabels = "Fallacy" if flabels and flabels[0] is not None else "NoFallacy"

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

    no_fallacy_df = df[df["fallacy_label"] == "NoFallacy"]
    with_fallacy_df = df[df["fallacy_label"] != "NoFallacy"]

    no_fallacy_downsampled = no_fallacy_df.sample(n=len(with_fallacy_df), random_state=42)

    df_balanced = pd.concat([no_fallacy_downsampled, with_fallacy_df]).sample(frac=1, random_state=42).reset_index(
        drop=True)

    train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42,
                                         stratify=df_balanced["fallacy_label"])
    test_df, dev_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df["fallacy_label"])

    if retriever:
        if search_type == "bm25":
            logger.info("Pre-building BM25 index (this may take a while)...")
            load_bm25_retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
            logger.info("BM25 index ready.")

        enriched_rows = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Enriching train with RAG context"):
            copy_row = row.copy()
            speech_turn = row['speech_turn']

            passages = search_knowledge_base(
                speech_turn,
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
            )
            if passages and summarization_model:
                #  Summarise single paragraphs
                passages = summarize_docs(passages, summarization_model)

            #  Summarise paragraphs together instead of summarising single passages
            # passages = summarize_docs([" ".join(passages)], summarization_model)

            rag_context = " [SEP] ".join(passages)
            rag_context = normalize_text(rag_context)
            copy_row['rag_context'] = rag_context
            enriched_rows.append(copy_row)

        train_df = pd.DataFrame(enriched_rows).reset_index(drop=True)

    train_df.to_csv(f"{path}/train.csv", index=False)
    test_df.to_csv(f"{path}/test.csv", index=False)
    dev_df.to_csv(f"{path}/dev.csv", index=False)

    _print_dataset_statistics(train_df, test_df, dev_df)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BIO-tag datasets with and without RAG context.")
    parser.add_argument(
        "--retriever",
        default=None,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model for vector search.",
    )
    parser.add_argument(
        "--summarization-model",
        default=None,
        choices=[
            "openai/gpt-oss-20b",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen3.5-9B-Base"
        ],
        help="Optional model name to summarize retrieved passages.",
    )
    parser.add_argument(
        "--persist-directory",
        default="./chromadb",
        help="Path to the Chroma vector store directory (default: ./chromadb).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text splitter chunk size in characters (default: 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks in characters (default: 80).",
    )
    parser.add_argument(
        "--search-type",
        default="similarity",
        choices=list(SEARCH_TYPES),
        help="Retrieval strategy to use (default: similarity).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to retrieve per query (default: 3).",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
        help="Candidate pool size for MMR retrieval (default: 20).",
    )
    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=0.5,
        help="MMR diversity factor between 0 (max diversity) and 1 (max relevance) (default: 0.5).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score for similarity_score_threshold retrieval (default: 0.5).",
    )
    parser.add_argument(
        "--temporal-rag",
        action="store_true",
        default=False,
        help="Only retrieve knowledge base documents whose year is before the debate year.",
    )
    args = parser.parse_args()

    if args.retriever == "bm25" and args.search_type != "bm25":
        parser.error("--retriever bm25 requires --search-type bm25")
    if args.search_type == "bm25" and args.retriever != "bm25":
        parser.error("--search-type bm25 requires --retriever bm25")

    path = f"dataset/detection"
    if args.retriever:
        retriever_name = args.retriever.replace("/", "-")
        path = f"dataset/detection/{retriever_name}"
        if args.summarization_model:
            path = f"dataset/detection/{retriever_name}/{args.summarization_model}"
            logger.info(
                f"Generating dataset with RAG Context with {args.retriever} as retriever and {args.summarization_model} as summarization model.")
        else:
            logger.info(
                f"Generating dataset with RAG Context with {args.retriever} as retriever and NO summarization model.")
    else:
        logger.info("Generating dataset without RAG Context.")

    os.makedirs(path, exist_ok=True)

    main(
        search_type=args.search_type,
        retriever=args.retriever,
        persist_directory_path=args.persist_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        score_threshold=args.score_threshold,
        summarization_model=args.summarization_model,
        temporal_rag=args.temporal_rag,
        path=path,
    )
