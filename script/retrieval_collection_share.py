#!/usr/bin/env python3
"""Share of retrieved RAG passages coming from each knowledge-base collection, for the best
elecdeb60to20 configuration of the detection and classification tasks.
``rag_context`` only stores the joined passage text, so each context is split back into the chunks
it was built from: token-based chunks from the Chroma store (dense leg) and character-based
chunks rebuilt as in create_db.py (BM25 leg)."""

from collections import Counter, defaultdict

import chromadb
import pandas as pd
import rootutils
from langchain_text_splitters import RecursiveCharacterTextSplitter

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils.fallacy_loader import FallacyLoader
from src.utils.util import normalize_text

KNOWLEDGE_BASE_PATH = "/Users/ddore/Documents/knowledge_base/"
KNOWLEDGE_BASE_DIRECTORIES = [
    "bills",
    "definitions",
    ("definitions/papers", "papers"),
    "exemplars",
    "geography",
    "history",
    "newspapers",
    "people",
]
# Same store create_db.py reads (--persist-directory default + retriever name)
CHROMA_PATH = "/Users/ddore/Documents/chromadb/ddore14-Sentence-RooseBERT"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KEY_LEN = 50

BEST_FOLDERS = {
    "detection": "temporal=true-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false",
    "classification": "temporal=false-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false",
}
SPLITS = ["train", "dev", "test"]
OUTPUT_FILE = "output/retrieval_collection_share_elecdeb60to20.csv"

# BM25 leg: rebuild the character-based chunks exactly as create_db.py does
docs = FallacyLoader(source_path=KNOWLEDGE_BASE_PATH, directories=KNOWLEDGE_BASE_DIRECTORIES).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
pool = [(c.page_content, c.metadata["category"]) for c in splitter.split_documents(docs)]
n_bm25 = len(pool)

# Dense leg: read the indexed token-based chunks straight from Chroma (no embedding model needed)
store = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("FallacyRAG")
total_dense = store.count()
for start in range(0, total_dense, 10000):
    got = store.get(include=["documents", "metadatas"], limit=10000, offset=start)
    pool += [(d, m["category"]) for d, m in zip(got["documents"], got["metadatas"])]

# Normalized chunk text -> set of collections it appears in (identical text can occur in several)
text2cats = defaultdict(set)
for text, cat in pool:
    t = normalize_text(text)
    if t:
        text2cats[t].add(cat)
categories = sorted({cat for _, cat in pool})

# Index long chunks by prefix; short ones are checked one by one
by_prefix = defaultdict(list)
short_texts = []
for t in text2cats:
    if len(t) >= KEY_LEN:
        by_prefix[t[:KEY_LEN]].append(t)
    else:
        short_texts.append(t)
print(f"Chunks: {n_bm25} BM25 + {total_dense} dense, {len(text2cats)} unique texts, collections: {categories}")

rows = []
for task, folder in BEST_FOLDERS.items():
    for split in SPLITS:
        df = pd.read_csv(f"datasets/elecdeb60to20-dataset/{task}/{folder}/{split}.csv", keep_default_na=False)
        passage_counts = Counter()
        row_hits = Counter()
        n_passages, n_unmatched_rows, n_empty = 0, 0, 0

        for ctx in df["rag_context"]:
            ctx = str(ctx)
            if not ctx:
                n_empty += 1
                continue
            pos, row_cats = 0, set()
            while pos < len(ctx):
                cands = by_prefix.get(ctx[pos:pos + KEY_LEN], []) + short_texts
                # Keep candidates that match here and end at a passage boundary
                ok = [t for t in cands if ctx.startswith(t, pos) and (pos + len(t) == len(ctx) or ctx[pos + len(t)] == " ")]
                if not ok:
                    n_unmatched_rows += 1
                    break
                t = max(ok, key=len)
                # Ambiguous text gives fractional credit to each collection
                for cat in text2cats[t]:
                    passage_counts[cat] += 1 / len(text2cats[t])
                    row_cats.add(cat)
                n_passages += 1
                pos += len(t) + 1
            row_hits.update(row_cats)

        n_rows = len(df) - n_empty
        for cat in categories:
            rows.append({
                "task": task,
                "split": split,
                "collection": cat,
                "passages": round(passage_counts[cat], 2),
                "passage_pct": round(100 * passage_counts[cat] / n_passages, 2) if n_passages else 0.0,
                "rows_with_collection_pct": round(100 * row_hits[cat] / n_rows, 2) if n_rows else 0.0,
            })
        print(f"[{task}/{split}] rows={len(df)} empty={n_empty} passages={n_passages} "
              f"rows_not_fully_matched={n_unmatched_rows}")

res = pd.DataFrame(rows)

# Aggregate over all splits per task
total = res.groupby(["task", "collection"], as_index=False)["passages"].sum()
total["passage_pct"] = (100 * total["passages"] / total.groupby("task")["passages"].transform("sum")).round(2)
total["split"] = "all"
res = pd.concat([res, total], ignore_index=True)

res.to_csv(OUTPUT_FILE, index=False)
pd.set_option("display.width", 200)
for task in BEST_FOLDERS:
    print(f"\n=== {task} ===")
    print(res[res["task"] == task].pivot(index="collection", columns="split", values="passage_pct")[SPLITS + ["all"]])
print(f"\nSaved to {OUTPUT_FILE}")
