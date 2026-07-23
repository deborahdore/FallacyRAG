"""
Prepare the propaganda dataset for relation classification (support / attack / no_relation)
*inference*, in the same CSV format as src/relation-extraction/dataset/persuasive-essays and
src/relation-extraction/dataset/LIARArg (columns: label, label_value, sentence1, sentence2 --
here named fallacy, candidate_relation), plus a few extra metadata columns kept for
traceability after prediction.

Source dataset
---------------
datasets/propaganda-dataset/relation-classification/original/{train,test,dev}.csv. Each row is
one sentence ("phrase") of a newspaper article, identified by article_id + sentence_id
(article_id + split are 1:1 with the raw article text under
datasets/propaganda-dataset/relation-classification/original/{train,test,dev}/article<id>.txt), with
a binary label (propaganda / non-propaganda) and, when propaganda, a ';'-separated list of
propaganda technique names in the 'techniques' column.

There is no relation annotation in this dataset -- the goal here is not to build a labeled
training set but a *prediction* set: every "fallacy" (a propaganda-labeled sentence) paired
with every other phrase (sentence) of its own article, so a relation-classification model
trained elsewhere (e.g. on LIARArg) can be run over it with --do_predict to see which phrases
it links a support/attack relation to.

Method
------
1. Group rows by (split, article_id), preserving each article's sentence order (by
   sentence_begin, the character offset into the raw article text).
2. Within each article, take every row with label == 'propaganda' as a fallacy anchor.
3. Pair each fallacy with every other sentence in the same article (all rows, propaganda or
   not, excluding the fallacy's own sentence -- a sentence is never paired with itself).
4. Unlike prepare_liarArg.py's undirected no_relation candidates, the role of each sentence
   here is already known, so no position-based reordering is needed: the fallacy sentence
   always goes in the 'fallacy' column and the other article phrase in 'candidate_relation'.
5. Since there is no gold relation label, label is written as -1 / label_value 'unknown'.
   run_relation_classification.py already special-cases label == -1 as "no label" during
   tokenisation, so these rows can be fed straight in as a --test_file for --do_predict.

Run: python3 prepare_propaganda.py
"""

from __future__ import annotations

import csv

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

csv.field_size_limit(10 ** 9)

INPUT_DIR = ROOT / "datasets" / "propaganda-dataset" / "original"
OUTPUT_DIR = ROOT / "datasets" / "propaganda-dataset" / "relation-classification"

SPLITS = ("train", "dev", "test")

NO_LABEL = -1
NO_LABEL_VALUE = "unknown"


def load_articles(split: str):
    """Read one split CSV and group its rows by article_id, ordered by sentence_begin."""
    path = INPUT_DIR / f"{split}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    articles = {}
    for row in rows:
        articles.setdefault(row["article_id"], []).append(row)

    for article_id, article_rows in articles.items():
        article_rows.sort(key=lambda r: int(r["sentence_begin"]))

    return articles


def build_pairs(articles: dict, split: str) -> list[dict]:
    """For every article, pair each propaganda-labeled sentence with every other sentence."""
    pairs = []
    for article_id, article_rows in articles.items():
        fallacies = [r for r in article_rows if r["label"] == "propaganda"]
        if not fallacies:
            continue

        for fallacy in fallacies:
            for phrase in article_rows:
                if phrase["sentence_id"] == fallacy["sentence_id"]:
                    continue  # never pair a sentence with itself

                pairs.append(
                    {
                        "label": NO_LABEL,
                        "label_value": NO_LABEL_VALUE,
                        "fallacy": fallacy["sentence"],
                        "candidate_relation": phrase["sentence"],
                        "article_id": article_id,
                        "fallacy_sentence_id": fallacy["sentence_id"],
                        "phrase_sentence_id": phrase["sentence_id"],
                        "fallacy_techniques": fallacy["techniques"],
                        "split": split
                    }
                )
    return pairs


def write_split(pairs: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "dataset.csv"
    fieldnames = [
        "label",
        "label_value",
        "fallacy",
        "candidate_relation",
        "article_id",
        "fallacy_sentence_id",
        "phrase_sentence_id",
        "fallacy_techniques",
        "split"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    n_articles = len({p["article_id"] for p in pairs})
    n_fallacies = len({(p["article_id"], p["fallacy_sentence_id"]) for p in pairs})
    print(
        f"Dataset: {len(pairs)} pairs from {n_fallacies} fallacies "
        f"across {n_articles} articles -> {path.relative_to(ROOT)}"
    )


def main():
    pairs = []
    for split in SPLITS:
        articles = load_articles(split)
        pairs.extend(build_pairs(articles, split))
    write_split(pairs)


if __name__ == "__main__":
    main()
