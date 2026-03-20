import argparse
import logging
from pathlib import Path

import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
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


def load_vector_store(embeddings_model: str, persist_directory_path: str) -> Chroma:
    if not Path(persist_directory_path).exists():
        raise FileNotFoundError(
            f"No vector store found at '{persist_directory_path}'. "
            f"Run the indexing script first with the same model and chunk settings."
        )

    embedding_fn = EMBEDDING_MODELS.get(embeddings_model)

    log.info(f"Loading vector store from: {persist_directory_path}")
    return Chroma(
        collection_name="FallacyRAG",
        embedding_function=embedding_fn(),
        persist_directory=persist_directory_path,
    )


def load_bm25_retriever(args) -> BM25Retriever:
    """Build a BM25Retriever directly from the source documents."""
    log.info("Loading documents for BM25 (no vector store used).")
    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=["people", "bills", "definitions", "geography"],
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    log.info(f"Building BM25 index over {len(all_splits)} chunks.")
    return BM25Retriever.from_documents(all_splits, k=args.top_k)


def main(args):
    search_config = SEARCH_TYPES[args.search_type]
    log.info(f"Search type: {args.search_type} — {search_config['description']}")

    if args.search_type == "bm25":
        retriever = load_bm25_retriever(args)
    else:
        vector_store = load_vector_store(args.embeddings_model, args.persist_directory_path)
        search_kwargs = search_config["search_kwargs"](args)
        log.info(f"Search kwargs: {search_kwargs}")
        retriever = vector_store.as_retriever(search_type=args.search_type, search_kwargs=search_kwargs)

    log.info(f"Query: '{args.query}'")
    return retriever.invoke(args.query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve documents from a Chroma vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--query", type=str, required=True, help="Natural-language query to search for.")
    parser.add_argument("--embeddings-model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        choices=EMBEDDING_MODELS.keys(),
                        help="Embedding model used to encode documents. Ignored when --search-type is bm25.")
    parser.add_argument("--persist_directory_path", type=str,
                        default="./chromadb/sentence-transformers-all-mpnet-base-v2",
                        help="Path to the persistent directory of ChromaDB")
    parser.add_argument("--search-type", type=str, default="similarity", choices=SEARCH_TYPES.keys(),
                        help=(
                            "similarity: top-k by cosine distance. "
                            "mmr: diverse results via Maximal Marginal Relevance. "
                            "similarity_score_threshold: top-k above a minimum score. "
                            "bm25: keyword-based retrieval, no embeddings or vector store required."))
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to return.")
    parser.add_argument("--fetch-k", type=int, default=20, help="[mmr] Candidate pool size before MMR re-ranking.")
    parser.add_argument("--lambda-mult", type=float, default=0.5,
                        help="[mmr] Diversity factor: 1.0=pure relevance, 0.0=pure diversity.")
    parser.add_argument("--score-threshold", type=float, default=0.6,
                        help="[similarity_score_threshold] Minimum relevance score (0–1).")

    args = parser.parse_args()
    docs = main(args)
    print(f"\n── {args.search_type} | {len(docs)} results ──")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.metadata}")
        print(f"     {doc.page_content[:200]}...")
        print()
