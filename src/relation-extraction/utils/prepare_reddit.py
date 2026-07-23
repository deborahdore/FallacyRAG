"""
Prepare the reddit dataset for relation classification (support / attack / no_relation)
*inference*, in the same CSV format as prepare_propaganda.py (columns: label, label_value,
fallacy, candidate_relation), plus a few extra metadata columns kept for traceability after
prediction.

Source dataset
---------------
datasets/reddit-dataset/relation-classification/original/{train,dev,test}.csv. Each row is one
whole comment/speech-turn ("id", "speech_turn", "fallacy_labels"), where fallacy_labels is a
per-character label list the same length as speech_turn (one of a single fallacy type, e.g.
'SlipperySlope', or 'O' for non-fallacy characters -- each row carries at most one fallacy
type, see reddit-dataset/relation-classification/original/transform.py).

Unlike propaganda-dataset, there is no separate article/sentence table: each row is its own
document (no cross-row grouping) and fallacies are marked at the character level rather than
per-sentence. There is also no relation annotation -- as with prepare_propaganda.py, the goal
here is not to build a labeled training set but a *prediction* set: every "fallacy" sentence
paired with every other sentence of its own document, so a relation-classification model
trained elsewhere (e.g. on change-my-view) can be run over it with --do_predict to see which
sentences it links a support/attack relation to.

Method
------
1. Split each row's speech_turn into sentences with character offsets (simple regex sentence
   splitter -- this repo has no nltk/spacy dependency).
2. For each sentence, look up the character-level labels covering its span. A sentence is a
   "fallacy" anchor if any of its characters carry a non-'O' label; its technique is the most
   common non-'O' label within its span (ties broken by first occurrence).
3. Skip rows with no fallacy sentences, or with fewer than two sentences (nothing to pair).
4. Pair each fallacy sentence with every other sentence in the same row (all sentences,
   fallacy or not, excluding the fallacy sentence's own index -- a sentence is never paired
   with itself), matching prepare_propaganda.py's per-article pairing but at the per-row level.
5. Since there is no gold relation label, label is written as -1 / label_value 'unknown'.
   run_relation_classification.py already special-cases label == -1 as "no label" during
   tokenisation, so these rows can be fed straight in as a --test_file for --do_predict.

Run: python3 prepare_reddit.py
"""

from __future__ import annotations

import csv
import re
from collections import Counter

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

csv.field_size_limit(10 ** 9)

INPUT_DIR = ROOT / "datasets" / "reddit-dataset" / "original"
OUTPUT_DIR = ROOT / "datasets" / "reddit-dataset" / "relation-classification"

SPLITS = ("train", "dev", "test")

NO_LABEL = -1
NO_LABEL_VALUE = "unknown"

# Split on '.', '!', '?' followed by whitespace (or end of string), keeping the punctuation
# with the preceding sentence. Good enough for pairing purposes; no nltk/spacy dependency.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into (start, end, sentence_text) spans covering the original string."""
    spans = []
    pos = 0
    for part in SENTENCE_SPLIT_RE.split(text):
        if not part:
            continue
        start = text.index(part, pos)
        end = start + len(part)
        stripped = part.strip()
        if stripped:
            spans.append((start, end, stripped))
        pos = end
    return spans


def sentence_technique(labels: list[str], start: int, end: int) -> str | None:
    """Most common non-'O' label covering [start, end); None if the sentence has no fallacy."""
    counts = Counter(lbl for lbl in labels[start:end] if lbl != "O")
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def load_rows(split: str):
    path = INPUT_DIR / f"{split}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_pairs(rows: list[dict], split: str) -> list[dict]:
    """For every row (document), pair each fallacy-labeled sentence with every other sentence."""
    import ast

    pairs = []
    n_docs_no_fallacy = 0
    n_docs_too_short = 0

    for row in rows:
        text = row["speech_turn"]
        try:
            labels = ast.literal_eval(row["fallacy_labels"])
        except (ValueError, SyntaxError):
            continue
        if len(labels) != len(text):
            continue

        sentences = split_sentences(text)
        if len(sentences) < 2:
            n_docs_too_short += 1
            continue

        annotated = [
            (idx, start, end, sent, sentence_technique(labels, start, end))
            for idx, (start, end, sent) in enumerate(sentences)
        ]
        fallacies = [s for s in annotated if s[4] is not None]
        if not fallacies:
            n_docs_no_fallacy += 1
            continue

        for f_idx, _f_start, _f_end, f_sent, f_technique in fallacies:
            for p_idx, _p_start, _p_end, p_sent, _p_technique in annotated:
                if p_idx == f_idx:
                    continue  # never pair a sentence with itself

                pairs.append(
                    {
                        "label": NO_LABEL,
                        "label_value": NO_LABEL_VALUE,
                        "fallacy": f_sent,
                        "candidate_relation": p_sent,
                        "doc_id": row["id"],
                        "fallacy_sentence_id": f_idx,
                        "phrase_sentence_id": p_idx,
                        "fallacy_techniques": f_technique,
                        "split": split,
                    }
                )

    print(f"[{split}] documents with usable pairs: "
          f"{len({p['doc_id'] for p in pairs})} / {len(rows)}")
    print(f"[{split}] documents skipped (no fallacy sentence): {n_docs_no_fallacy}")
    print(f"[{split}] documents skipped (fewer than 2 sentences): {n_docs_too_short}")

    return pairs


def write_split(pairs: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "dataset.csv"
    fieldnames = [
        "label",
        "label_value",
        "fallacy",
        "candidate_relation",
        "doc_id",
        "fallacy_sentence_id",
        "phrase_sentence_id",
        "fallacy_techniques",
        "split",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    n_docs = len({p["doc_id"] for p in pairs})
    n_fallacies = len({(p["doc_id"], p["fallacy_sentence_id"]) for p in pairs})
    print(
        f"Dataset: {len(pairs)} pairs from {n_fallacies} fallacy sentences "
        f"across {n_docs} documents -> {path.relative_to(ROOT)}"
    )


def main():
    pairs = []
    for split in SPLITS:
        rows = load_rows(split)
        pairs.extend(build_pairs(rows, split))
    write_split(pairs)


if __name__ == "__main__":
    main()
