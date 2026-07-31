"""Prepare the LIARArg dataset for relation classification (support / attack / no_relation).
Keeps annotated rows holding at least one claim, one premise and one relation, builds positive
pairs from the four relation columns (psupport/pattack merged into support/attack), and samples a
matching number of no_relation pairs from that row's unlinked component combinations.
Splits at statement level into 80/10/10. Run: ``python3 prepare_liarArg.py``."""

from __future__ import annotations

import ast
import csv
import random
from itertools import combinations

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

csv.field_size_limit(10 ** 9)

INPUT_CSV = ROOT / "datasets" / "LIARArg-dataset" / "original" / "LIARArg.csv"
OUTPUT_DIR = ROOT / "datasets" / "LIARArg-dataset" / "relation-classification"

SEED = 42
TRAIN_FRAC = 0.8
DEV_FRAC = 0.1

LABEL_VALUE = {0: "attack", 1: "support", 2: "no_relation"}


def parse_list(s: str):
    """Parse a stringified python list column (e.g. '[123, 456]') into a real list."""
    if not s or s == "[]":
        return []
    return ast.literal_eval(s)


def load_documents():
    """Read LIARArg.csv and build, per usable statement, a list of (label, sent1, sent2)."""
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rng = random.Random(SEED)

    documents = {}
    n_skipped_relation_ids = 0
    n_docs_no_components = 0
    n_docs_no_positive = 0

    for row in rows:
        if row["quality"] != "":
            continue

        claim_ids = parse_list(row["claim_id"])
        claim_texts = parse_list(row["claim"])
        claim_positions = parse_list(row["claim_position"])
        premise_ids = parse_list(row["premise_id"])
        premise_texts = parse_list(row["premise"])
        premise_positions = parse_list(row["premise_position"])

        if not claim_ids or not premise_ids:
            n_docs_no_components += 1
            continue

        components = {}
        for i, (cid, text) in enumerate(zip(claim_ids, claim_texts)):
            pos = claim_positions[2 * i] if 2 * i < len(claim_positions) else None
            components[cid] = {"text": text, "pos": pos}
        for i, (pid, text) in enumerate(zip(premise_ids, premise_texts)):
            pos = premise_positions[2 * i] if 2 * i < len(premise_positions) else None
            components[pid] = {"text": text, "pos": pos}

        pos_pairs = []
        for column, label in (
                ("support_relation", 1),
                ("psupport_relation", 1),
                ("attack_relation", 0),
                ("pattack_relation", 0),
        ):
            rel = parse_list(row[column])
            for i in range(0, len(rel), 2):
                dep, gov = rel[i], rel[i + 1]
                if dep in components and gov in components:
                    pos_pairs.append((dep, gov, label))
                else:
                    n_skipped_relation_ids += 1

        if not pos_pairs:
            n_docs_no_positive += 1
            continue

        linked_unordered = {frozenset((d, g)) for d, g, _ in pos_pairs}

        all_ids = list(components.keys())
        candidates = []
        for a, b in combinations(all_ids, 2):
            if frozenset((a, b)) in linked_unordered:
                continue
            pos_a, pos_b = components[a]["pos"], components[b]["pos"]
            if pos_a is not None and pos_b is not None and pos_a != pos_b:
                first, second = (a, b) if pos_a < pos_b else (b, a)
            else:
                first, second = a, b
            candidates.append((first, second))

        rng.shuffle(candidates)
        no_relation_sample = candidates[: len(pos_pairs)]

        examples = []
        for dep, gov, label in pos_pairs:
            examples.append((label, components[dep]["text"], components[gov]["text"]))
        for first, second in no_relation_sample:
            examples.append((2, components[first]["text"], components[second]["text"]))

        documents[row["id"]] = examples

    print(f"Usable statements with pairs: {len(documents)}")
    print(f"Statements skipped (no claim+premise components): {n_docs_no_components}")
    print(f"Statements skipped (components but no valid relation): {n_docs_no_positive}")
    print(f"Relation entries skipped (id not found in own statement): {n_skipped_relation_ids}")

    return documents, rng


def split_documents(documents, rng):
    """Split at the statement level so a statement's pairs never cross splits."""
    doc_ids = list(documents.keys())
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(round(n * TRAIN_FRAC))
    n_dev = int(round(n * DEV_FRAC))

    train_ids = doc_ids[:n_train]
    dev_ids = doc_ids[n_train: n_train + n_dev]
    test_ids = doc_ids[n_train + n_dev:]

    splits = {}
    for name, ids in (("train", train_ids), ("dev", dev_ids), ("test", test_ids)):
        examples = []
        for doc_id in ids:
            examples.extend(documents[doc_id])
        rng.shuffle(examples)
        splits[name] = examples
    return splits


def write_split(name: str, examples: list[tuple[int, str, str]]):
    path = OUTPUT_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "label_value", "sentence1", "sentence2"])
        for label, sent1, sent2 in examples:
            writer.writerow([label, LABEL_VALUE[label], sent1, sent2])

    from collections import Counter

    counts = Counter(LABEL_VALUE[label] for label, _, _ in examples)
    print(f"{name}: {len(examples)} pairs -> {path.name}  ({dict(counts)})")


def main():
    documents, rng = load_documents()
    splits = split_documents(documents, rng)
    for name in ("train", "dev", "test"):
        write_split(name, splits[name])


if __name__ == "__main__":
    main()
