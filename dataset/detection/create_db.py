from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
import unicodedata
from types import SimpleNamespace

import pandas as pd
import rootutils
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, CohereEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils.fallacy_loader import FallacyLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
KNOWLEDGE_BASE_PATH = "./knowledge_base/"

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
    "text-embedding-3-small": lambda: OpenAIEmbeddings(model="text-embedding-3-large"),
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


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[‘’´`]", "'", text)  # apostrophes
    text = re.sub(r"[“”]", '"', text)  # double quotes
    text = re.sub(r"[–—]", "-", text)  # dashes
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def convert(tag: str) -> str:
    """Normalize a BIO tag to a binary fallacy label (B-Fallacy, I-Fallacy, or O)."""
    if tag == "O":
        return "O"
    if tag.startswith("B-"):
        return "B-Fallacy"
    if tag.startswith("I-"):
        return "I-Fallacy"
    raise ValueError(f"Unknown tag: {tag}")


def main(
        persist_directory_path: str = "./chromadb",
        retriever: str = None,
        search_type: str = "similarity",
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        top_k: int = 3,
        summarization_model: str | None = None,
):
    path = f"dataset/detection"

    if retriever:
        retriever_name = retriever.replace("/", "-")
        path = f"dataset/detection/{retriever_name}"
        if summarization_model:
            path = f"dataset/detection/bio_tags/rag/{retriever_name}/{summarization_model}"
            logger.info(
                f"Generating dataset with RAG Context with {retriever} as retriever and {summarization_model} as summarization model.")
        else:
            logger.info(
                f"Generating dataset with RAG Context with {retriever} as retriever and NO summarization model.")
    else:
        logger.info("Generating dataset without RAG Context.")

    os.makedirs(path, exist_ok=True)

    for split in ["train", "dev", "test"]:
        dataset = []
        df = pd.read_csv(f"dataset/{split}.csv")

        for idx, (text, raw_tags) in enumerate(zip(df["Context"], df["bio_tags"])):
            tokens = normalize_text(text).split()
            ner_tags = [convert(t) for t in ast.literal_eval(raw_tags)]

            if retriever:
                passages = search_knowledge_base(
                    text,
                    search_type=search_type,
                    retriever=retriever,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    chromadb_path=persist_directory_path,
                )
                if passages and summarization_model:
                    passages = summarize_docs(passages, summarization_model)
                rag_context = " [SEP] ".join(passages)

                dataset.append({"id": idx,
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                                "context": rag_context})
            else:
                dataset.append({"id": idx,
                                "tokens": tokens,
                                "ner_tags": ner_tags})

        with open(f"{path}/{split}.json", "w") as f:
            json.dump(dataset, f, indent=4)


def search_knowledge_base(query: str,
                          search_type: str,
                          retriever: str,
                          chunk_size: int,
                          chunk_overlap: int,
                          top_k: int,
                          chromadb_path: str):
    """Return a list of passage strings relevant to *query*."""

    if search_type == "bm25":
        loader = FallacyLoader(source_path=KNOWLEDGE_BASE_PATH,
                               directories=["people", "bills", "definitions", "geography"])
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap,
                                                  add_start_index=True)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        retriever = BM25Retriever.from_documents(chunks, k=top_k)
    else:
        # TODO cache vector store
        chromadb_path = os.path.join(chromadb_path, retriever.replace("/", "-"))
        assert os.path.exists(chromadb_path)

        vector_store = Chroma(
            collection_name="FallacyRAG",
            embedding_function=EMBEDDING_MODELS[retriever](),
            persist_directory=chromadb_path,
        )
        search_kwargs = SEARCH_TYPES[search_type]["search_kwargs"](SimpleNamespace(top_k=top_k))
        retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    docs = retriever.invoke(query)
    return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]


def load_bm25_retriever(chunk_size: int, chunk_overlap: int, top_k: int) -> BM25Retriever:
    """Build a BM25 retriever directly from the knowledge base source documents."""
    loader = FallacyLoader(
        source_path=KNOWLEDGE_BASE_PATH,
        directories=["people", "bills", "definitions", "geography"],
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(loader.load())
    return BM25Retriever.from_documents(chunks, k=top_k)


def summarize_docs(docs: list[str], model_name: str) -> list[str]:
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
                max_tokens=256,
            )
            summaries.append(response.choices[0].message.content.strip())
    else:
        #  TODO cache summarization model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        for doc in docs:
            messages = [{"role": "user", "content": prompt_template.format(doc)}]
            formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = pipe(formatted, max_new_tokens=256, do_sample=False, return_full_text=False)
            summaries.append(output[0]["generated_text"].strip())

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BIO-tag datasets with and without RAG context.")
    parser.add_argument(
        "--retriever",
        default=None,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model for vector search (default: sentence-transformers/all-mpnet-base-v2).",
    )
    parser.add_argument(
        "--summarization-model",
        default=None,
        choices=[
            "gpt-4o-mini",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-4-31B-it",
            "meta-llama/Llama-3.1-8B-Instruct"
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
        default=800,
        help="Text splitter chunk size in characters (default: 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
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
    args = parser.parse_args()

    main(
        search_type=args.search_type,
        retriever=args.retriever,
        persist_directory_path=args.persist_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        summarization_model=args.summarization_model,
    )
