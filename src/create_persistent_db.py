"""Index the knowledge base into a persisted Chroma vector store with a chosen embedder."""

import argparse
import logging
from pathlib import Path

import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.utils.fallacy_loader import FallacyLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PERSIST_DIRECTORY = "./chromadb_complete/"
KNOWLEDGE_BASE_PATH = "./knowledge_base/"

Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_kwargs = {"device": DEVICE}
encode_kwargs = {
    "batch_size": 2048,
    "normalize_embeddings": True
}

LOCAL_HF_MODELS = {
    "sentence-transformers/all-mpnet-base-v2",
    "ddore14/Sentence-RooseBERT",
    "BAAI/bge-large-en-v1.5",
}

RETRIEVERS = {
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
}


def _apply_gpu_optimizations(embedding_function, retriever: str):
    """Apply FP16 + torch.compile to local HuggingFace models on CUDA for faster encoding."""
    if DEVICE != "cuda" or retriever not in LOCAL_HF_MODELS:
        return

    client = getattr(embedding_function, "client", None)
    if client is None:
        log.warning("Could not access embedding model client for GPU optimizations.")
        return

    try:
        client.half()
        log.info("Applied FP16 (half precision) to embedding model")
    except Exception as e:
        log.warning(f"FP16 conversion failed: {e}")

    try:
        embedding_function.client = torch.compile(client, mode="reduce-overhead")
        log.info("Applied torch.compile(mode='reduce-overhead') to embedding model")
    except Exception as e:
        log.warning(f"torch.compile not available or failed: {e}")


DEFAULT_DIRECTORIES = [
    "bills",
    "definitions",
    ("definitions/papers", "papers"),
    "exemplars",
    "geography",
    "history",
    "newspapers",
    "people",
]


def _deterministic_ids(splits: list) -> list[str]:
    """Build a stable "category:id:index" id per chunk so re-indexing upserts, not duplicates."""
    counters: dict[tuple[str, str], int] = {}
    ids: list[str] = []
    for doc in splits:
        category = str(doc.metadata.get("category", "uncat"))
        doc_id = str(doc.metadata.get("id", "noid"))
        key = (category, doc_id)
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        ids.append(f"{category}:{doc_id}:{idx}")
    return ids


def _build_text_splitter(embedding_function, retriever: str,
                         chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Build a token-aware splitter clamped to the embedding model's input window."""
    client = getattr(embedding_function, "client", None)
    tokenizer = getattr(client, "tokenizer", None)
    max_seq_length = getattr(client, "max_seq_length", None)

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(retriever)
        except Exception as e:
            log.warning(
                f"Could not load a tokenizer for {retriever!r} ({e}); "
                f"falling back to character-based splitting."
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )

    if max_seq_length:
        token_limit = max(8, int(max_seq_length) - 2)
        if chunk_size > token_limit:
            log.info(
                f"Requested chunk_size={chunk_size} tokens exceeds {retriever}'s input window "
                f"({max_seq_length}); clamping to {token_limit}."
            )
            chunk_size = token_limit

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 10
        log.info(f"Adjusted chunk_overlap to {chunk_overlap} tokens (must be < chunk_size).")

    log.info(
        f"Token-based splitting via the {retriever} tokenizer "
        f"(chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens)."
    )
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def main(retriever: str, chunk_size: int, chunk_overlap: int, batch_size: int,
         update: bool = False, directories: list | None = None):
    """Load, split, embed and index the knowledge base into the Chroma store."""
    torch.set_grad_enabled(False)

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log.info(f"Using device: {DEVICE}")
    log.info(f"Loading embedding model: {retriever}")

    persist_directory_path = PERSIST_DIRECTORY + retriever.replace("/", "-")
    Path(persist_directory_path).mkdir(parents=True, exist_ok=True)

    store_exists = any(Path(persist_directory_path).iterdir())
    if store_exists and not update:
        log.info("Vector database already exists. Skipping indexing. "
                 "Use --update to append documents incrementally.")
        return
    if update:
        log.info("Update mode: appending to the existing store (no full rebuild).")

    embedding_function = RETRIEVERS[retriever]()

    _apply_gpu_optimizations(embedding_function, retriever)

    vector_store = Chroma(
        collection_name="FallacyRAG",
        embedding_function=embedding_function,
        persist_directory=persist_directory_path,
    )

    log.info("Loading documents")

    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=directories if directories is not None else DEFAULT_DIRECTORIES,
    )

    docs = loader.load()
    log.info(f"Loaded {len(docs)} documents")

    log.info(
        f"Splitting documents (requested chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    text_splitter = _build_text_splitter(
        embedding_function, retriever, chunk_size, chunk_overlap
    )

    splits = text_splitter.split_documents(docs)

    ids = _deterministic_ids(splits)

    for i in tqdm(range(0, len(splits), batch_size), desc="Adding to Chroma"):
        vector_store.add_documents(splits[i: i + batch_size], ids=ids[i: i + batch_size])

    log.info("Indexing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents into a Chroma vector store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        choices=RETRIEVERS.keys(),
        help="Embedding model used to encode documents",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum tokens per chunk (clamped to the embedding model's input window)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Token overlap between chunks (default: 10% of chunk-size)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks indexed per batch",
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="Append to an existing store instead of skipping. Combined with deterministic "
             "ids, re-running upserts chunks rather than duplicating them.",
    )

    parser.add_argument(
        "--directories",
        nargs="+",
        default=None,
        help="Knowledge-base sub-folders to (re)index, e.g. --directories history. "
             "Defaults to all sections. Useful with --update to add only new content.",
    )

    args = parser.parse_args()

    overlap = (
        int(args.chunk_size * 0.1)
        if args.chunk_overlap is None
        else args.chunk_overlap
    )

    main(
        retriever=args.retriever,
        chunk_size=args.chunk_size,
        chunk_overlap=overlap,
        batch_size=args.batch_size,
        update=args.update,
        directories=args.directories,
    )
