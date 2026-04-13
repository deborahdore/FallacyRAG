import argparse
import logging
from pathlib import Path

import rootutils
import torch
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.utils.fallacy_loader import FallacyLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PERSIST_DIRECTORY = "./chromadb/"
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
    "text-embedding-005": lambda: VertexAIEmbeddings(model="text-embedding-005"),
    "llama3": lambda: OllamaEmbeddings(model="llama3"),
    "text-embedding-3-large": lambda: OpenAIEmbeddings(model="text-embedding-3-large"),
    "mistral-embed": lambda: MistralAIEmbeddings(model="mistral-embed"),
    "embed-english-v3.0": lambda: CohereEmbeddings(model="embed-english-v3.0"),
}


def _apply_gpu_optimizations(embedding_function, embedding_model: str):
    """
    Apply FP16 and torch.compile to local HuggingFace models on CUDA.
    FP16 halves memory usage and roughly doubles throughput on tensor cores.
    torch.compile fuses ops for an additional ~20-30% speedup.
    """
    if DEVICE != "cuda" or embedding_model not in LOCAL_HF_MODELS:
        return

    client = getattr(embedding_function, "client", None)
    if client is None:
        log.warning("Could not access embedding model client for GPU optimizations.")
        return

    # FP16: ~2x throughput on modern NVIDIA GPUs via tensor cores
    try:
        client.half()
        log.info("Applied FP16 (half precision) to embedding model")
    except Exception as e:
        log.warning(f"FP16 conversion failed: {e}")

    # torch.compile: graph-level fusion (PyTorch >= 2.0)
    # "reduce-overhead" minimises Python dispatcher overhead — best for one long encode pass
    try:
        embedding_function.client = torch.compile(client, mode="reduce-overhead")
        log.info("Applied torch.compile(mode='reduce-overhead') to embedding model")
    except Exception as e:
        log.warning(f"torch.compile not available or failed: {e}")


def main(embedding_model: str, chunk_size: int, chunk_overlap: int, batch_size: int):
    torch.set_grad_enabled(False)

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log.info(f"Using device: {DEVICE}")
    log.info(f"Loading embedding model: {embedding_model}")

    persist_directory_path = PERSIST_DIRECTORY + embedding_model.replace("/", "-")
    Path(persist_directory_path).mkdir(parents=True, exist_ok=True)

    if any(Path(persist_directory_path).iterdir()):
        log.info("Vector database already exists. Skipping indexing.")
        return

    embedding_function = EMBEDDING_MODELS[embedding_model]()

    # Apply FP16 + torch.compile for local HuggingFace models on GPU
    _apply_gpu_optimizations(embedding_function, embedding_model)

    vector_store = Chroma(
        collection_name="FallacyRAG",
        embedding_function=embedding_function,
        persist_directory=persist_directory_path,
    )

    log.info("Loading documents")

    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=["people", "bills", "definitions", "geography"],
    )

    docs = loader.load()
    log.info(f"Loaded {len(docs)} documents")

    log.info(
        f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    splits = text_splitter.split_documents(docs)

    for i in tqdm(range(0, len(splits), batch_size), desc="Adding to Chroma"):
        vector_store.add_documents(splits[i: i + batch_size])

    log.info("Indexing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents into a Chroma vector store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--embeddings-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        choices=EMBEDDING_MODELS.keys(),
        help="Embedding model used to encode documents",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum characters per chunk",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Character overlap between chunks",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks indexed per batch",
    )

    args = parser.parse_args()

    overlap = (
        int(args.chunk_size * 0.1)
        if args.chunk_overlap is None
        else args.chunk_overlap
    )

    main(
        embedding_model=args.embeddings_model,
        chunk_size=args.chunk_size,
        chunk_overlap=overlap,
        batch_size=args.batch_size,
    )