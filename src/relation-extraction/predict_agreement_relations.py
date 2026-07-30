"""
Predict agreement relations with the DEBAGREEMENT-trained relation-classification checkpoint.

This runs a relation-classification model fine-tuned on DEBAGREEMENT (by default
``output/trained-models/reddit-model-to-add``) over the reddit *prediction* pairs built by
``prepare_reddit.py`` (``datasets/reddit-dataset/relation-classification/dataset.csv``).

Each row pairs a fallacy-labeled sentence (``fallacy``) with another phrase from the same
comment/speech-turn (``candidate_relation``). The model predicts one of three relations:

    0 -> attack   (disagree)
    1 -> neutral
    2 -> support  (agree)

The id->relation map comes from how ``run_relation_classification.py`` builds ``label_list``:
it sorts the string labels of the training set (``"0","1","2"``), and in the DEBAGREEMENT data
those ids mean ``0=attack, 1=neutral, 2=support`` (see the ``label``/``label_value`` columns of
``datasets/debagreeement-dataset/relation-classification/train.csv``). This differs from the
LIARArg/propaganda mapping, where ``1=support`` and ``2=no_relation``.

There are no gold relations for the reddit pairs (``label = -1``), so this is pure inference --
output is written back to CSV with a predicted relation and per-class softmax probabilities,
alongside the original metadata columns.

Run (from the project root):

    python src/relation-extraction/predict_agreement_relations.py

    # custom checkpoint / batch size
    python src/relation-extraction/predict_agreement_relations.py \
        --model-dir output/trained-models/reddit-model-to-add \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import rootutils
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Pin cwd to the project root (same as run_relation_classification.py) so relative paths work.
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

# Semantic label map. The checkpoint's config.id2label is just {0:"0",1:"1",2:"2"};
# the meaning comes from the DEBAGREEMENT label_value column.
ID2RELATION = {0: "attack", 1: "neutral", 2: "support"}

# Column names in dataset.csv (mirrors --text_column_name / --speech_turn_column_name in run.sh)
TEXT_COLUMN = "fallacy"  # sentence1 / segment A
PAIR_COLUMN = "candidate_relation"  # sentence2 / segment B

DEFAULT_MODEL_DIR = (
        ROOT / "output" / "trained-models" / "DEBAGREEMENT-ModernBERT-base-bs16_lr3e-5_ep3"
)
DEFAULT_DATA_FILE = ROOT / "datasets" / "reddit-dataset" / "relation-classification" / "dataset.csv"
DEFAULT_OUTPUT_FILE = ROOT / "datasets" / "reddit-dataset" / "relation-classification" / "predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the fine-tuned relation-classification checkpoint.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help="Reddit prediction pairs CSV (from prepare_reddit.py).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Where to write the predictions CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token sequence length (DeBERTa-v3-base max_position_embeddings).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a device (cuda / mps / cpu). Auto-detected if omitted.",
    )
    return parser.parse_args()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def predict(frame: pd.DataFrame, model, tokenizer, device: str, batch_size: int, max_length: int) -> np.ndarray:
    """Return an (N, num_labels) array of softmax probabilities."""
    s1 = frame[TEXT_COLUMN].tolist()
    s2 = frame[PAIR_COLUMN].tolist()
    all_probs = []

    n = len(frame)
    start = time.time()
    for i in tqdm(range(0, n, batch_size)):
        enc = tokenizer(
            s1[i: i + batch_size],
            s2[i: i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

        done = min(i + batch_size, n)
        if done % (batch_size * 50) == 0 or done == n:
            rate = done / (time.time() - start + 1e-9)
            print(f"  {done}/{n}  ({rate:.0f} rows/s)")

    return np.concatenate(all_probs, axis=0)


def main() -> None:
    from pathlib import Path

    args = parse_args()
    device = pick_device(args.device)
    print(f"Project root: {ROOT}")
    print(f"Device: {device}")

    # Resolve the checkpoint path and fail early if missing; otherwise transformers
    # treats it as a Hub repo id and raises a confusing error.
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {model_dir}\n"
            f"Pass the correct path with --model-dir (it must be a local folder containing "
            f"config.json + model.safetensors + tokenizer files)."
        )
    print(f"Loading checkpoint from: {model_dir}")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    # local_files_only=True keeps loading offline so a local path is never mistaken for a
    # Hub repo id (and never triggers a download).
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    print(f"num_labels: {num_labels} | config.id2label: {model.config.id2label}")
    assert num_labels == len(ID2RELATION), "Label count mismatch with ID2RELATION map."
    print(f"Applied relation map: {ID2RELATION}")

    # ── Load the reddit prediction pairs ──────────────────────────────────────
    df = pd.read_csv(args.data_file)
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    # Guard against missing/NaN text so tokenisation never crashes.
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[PAIR_COLUMN] = df[PAIR_COLUMN].fillna("").astype(str)

    work_df = df
    print(f"Predicting {len(work_df)} rows.")

    # ── Inference ─────────────────────────────────────────────────────────────
    probs = predict(work_df, model, tokenizer, device, args.batch_size, args.max_length)
    pred_ids = probs.argmax(axis=1)

    # ── Attach predictions ────────────────────────────────────────────────────
    result = work_df.copy()
    result["predicted_label"] = pred_ids
    result["predicted_relation"] = [ID2RELATION[i] for i in pred_ids]
    result["confidence"] = probs.max(axis=1)
    for i, name in ID2RELATION.items():
        result[f"prob_{name}"] = probs[:, i]

    print("\nPredicted relation distribution:")
    print(result["predicted_relation"].value_counts())
    print(f"Mean confidence: {float(result['confidence'].mean()):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
