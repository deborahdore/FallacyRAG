import ast
import csv

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def style_label(label):
    """Convert a reddit-dataset label (e.g. 'worse_problems') to PascalCase
    (e.g. 'WorseProblems'), matching the style of the main dataset's
    fallacy_labels column. Non-fallacy labels become 'O'."""
    if label in ("none", "non_fallacy", "None"):
        return "O"
    parts = label.split("_")
    return "".join(p.capitalize() for p in parts)


def align_tokens(text, tokens):
    """Align a list of word tokens to character spans in the raw text.

    The reddit dataset stores text twice: once as a whitespace/punctuation
    tokenized list (`tokens`) and once as the original raw string (`text`).
    To turn per-token fallacy labels into per-character labels we need to
    find where each token lands in the raw text.

    Returns a list of (start, end) tuples the same length as `tokens`, with
    None for any token that couldn't be located.
    """
    spans = []
    pos = 0
    n = len(text)
    for tok in tokens:
        tok_len = len(tok)
        while pos < n and text[pos].isspace():
            pos += 1
        if text[pos:pos + tok_len] == tok:
            spans.append((pos, pos + tok_len))
            pos += tok_len
            continue
        # Tokenizers sometimes normalize quote characters; try that too.
        norm_tok = tok.replace("``", '"').replace("''", '"')
        found = text.find(tok, pos, pos + 200)
        if found == -1:
            found = text.find(norm_tok, pos, pos + 200)
        if found != -1:
            spans.append((found, found + len(tok)))
            pos = found + len(tok)
            continue
        spans.append(None)
    return spans


def parse_row(row):
    """Parse one tab-separated row into its 8 fields, returning None if the
    row is malformed or its token/label lists don't line up."""
    parts = row.rstrip("\n").split("\t")
    if len(parts) != 8:
        return None
    id_, coi_tokens, coi, detection_label, classification_label, parent_coi, title, comment = parts
    try:
        tokens = ast.literal_eval(coi_tokens)
        classify = ast.literal_eval(classification_label)
    except (ValueError, SyntaxError):
        return None
    if len(tokens) != len(classify):
        return None
    return id_, coi, tokens, classify


def to_char_labels(text, tokens, classify):
    """Build a character-level fallacy label array (one label per character
    in `text`) from token-level classification labels."""
    spans = align_tokens(text, tokens)
    labels = ["O"] * len(text)
    for span, cls in zip(spans, classify):
        if span is None:
            continue
        lab = style_label(cls)
        if lab == "O":
            continue
        start, end = span
        for i in range(start, end):
            labels[i] = lab
    return labels


def load(path):
    """Yield (id, speech_turn, fallacy_labels) tuples for every parseable
    row in the given reddit-dataset .txt file."""
    with open(path, "r", encoding="utf-8") as f:
        for row in f.readlines():
            parsed = parse_row(row)
            if parsed is None:
                continue
            _, text, tokens, classify = parsed
            labels = to_char_labels(text, tokens, classify)
            yield text, labels


def transform(in_path, out_path):
    """Transform a reddit-dataset .txt file into a CSV with the same
    id/speech_turn/fallacy_labels shape as dataset/train.csv."""
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "speech_turn", "fallacy_labels"])
        idx = 0
        for text, labels in load(in_path):
            writer.writerow([idx, text, repr(labels)])
            idx += 1
    print(f"wrote {out_path}: {idx} rows")


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        transform(f"./reddit-dataset/{split}.txt", f"./reddit-dataset/{split}.csv")
