import argparse
import logging
from pathlib import Path

import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.fallacy_loader import FallacyLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PERSIST_DIRECTORY = "./chromadb/"
KNOWLEDGE_BASE_PATH = "./knowledge_base/"


def select_embedding_model(model_provider: str, model: str):
    """Select dense embedding model."""
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

    if model_provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs)

    elif model_provider == "openai":
        return OpenAIEmbeddings(model=model)

    elif model_provider == "google":
        return VertexAIEmbeddings(model=model)

    elif model_provider == "facebook":
        return OllamaEmbeddings(model=model)

    elif model_provider == "BAAI":
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    else:
        raise ValueError(f"Unknown embedding provider: {model_provider}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-model-provider", type=str, default="huggingface")
    parser.add_argument("--embeddings-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()

    Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
    log.info(f"Loading {args.embeddings_model} as embedding model.")
    embeddings = select_embedding_model(
        model_provider=args.embeddings_model_provider,
        model=args.embeddings_model,
    )

    log.info(f"Loading Vector database.")
    vector_store = Chroma(
        collection_name=f"FallacyRAG",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY + args.embeddings_model.replace("/", "-")
    )

    log.info(f"Loading documents.")
    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=["people", "bills", "definitions", "geography"],
    )

    docs = loader.load()

    log.info(f"Splitting documents. \n Chunk overlap: {args.chunk_overlap} | Chunk size: {args.chunk_size}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)

    log.info(f"Adding documents to database.")
    vector_store.add_documents(documents=all_splits)

    log.info("Indexing completed.")
