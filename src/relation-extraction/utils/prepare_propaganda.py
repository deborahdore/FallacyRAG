"""Prepare the propaganda dataset for relation-classification inference.
Groups sentences by article and pairs every propaganda-labeled sentence with every other phrase
of the same article, so a model trained elsewhere (e.g. on LIARArg) can be run over it with
``--do_predict``. There is no gold relation, so ``label`` is written as -1 / 'unknown'.
Run: ``python3 prepare_propaganda.py``."""

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
                    continue

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
