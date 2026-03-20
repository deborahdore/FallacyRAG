import argparse
import logging
from pathlib import Path

import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": True}

EMBEDDING_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs=model_kwargs
    ),
    "BAAI/bge-large-en-v1.5": lambda: HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    ),
    "text-embedding-3-large": lambda: OpenAIEmbeddings(model="text-embedding-3-large"),
    "text-embedding-005": lambda: VertexAIEmbeddings(model="text-embedding-005"),
    "llama3": lambda: OllamaEmbeddings(model="llama3"),
    "mistral-embed": lambda: MistralAIEmbeddings(model="mistral-embed"),
}


def main(embedding_model: str, chunk_size: int, chunk_overlap: int):
    log.info(f"Loading {embedding_model} as embedding model.")

    log.info(f"Loading Vector database.")
    persist_directory_path = (PERSIST_DIRECTORY +
                              embedding_model.replace("/", "-") +
                              f"/overlap={chunk_overlap}-size={chunk_size}")
    Path(persist_directory_path).mkdir(parents=True, exist_ok=True)

    embedding_model = EMBEDDING_MODELS.get(embedding_model)
    vector_store = Chroma(
        collection_name=f"FallacyRAG",
        embedding_function=embedding_model(),
        persist_directory=persist_directory_path
    )

    log.info(f"Loading documents.")
    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=["people", "bills", "definitions", "geography"],
    )

    docs = loader.load()

    log.info(f"Splitting documents. \n Chunk overlap: {chunk_overlap} | Chunk size: {chunk_size}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)

    log.info(f"Adding documents to database.")
    vector_store.add_documents(documents=all_splits)

    log.info("Indexing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Index documents into a Chroma vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        choices=EMBEDDING_MODELS.keys(),
        help="Embedding model used to encode documents. Must match the model used at retrieval time.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Number of characters overlapping between consecutive chunks. Higher values preserve more context across chunk boundaries.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Maximum number of characters per chunk. Larger chunks retain more context but may exceed model token limits.",
    )
    args = parser.parse_args()

    main(args.embeddings_model, args.chunk_size, args.chunk_overlap)
