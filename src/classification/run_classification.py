"""
Fine-tuning BERT-style models for fallacy classification, augmented with retrieved
knowledge-base context (RAG).

The label format is auto-detected (see ``detect_label_columns``):
  - single-label  : one categorical column (``--label_column_name``, default ``fallacy_label``),
                    softmax head + class-weighted cross-entropy, argmax decoding
                    (elecdeb / reddit).
  - multi-label   : several 0/1 indicator columns (one per fallacy + "None of the
                    above"), sigmoid head + pos_weight BCEWithLogitsLoss (per-class
                    inverse-frequency weighting), threshold decoding, and
                    macro/micro/weighted/samples F1 metrics. Force it with
                    ``--label_column_names`` and tune ``--multi_label_threshold``.

Architecture
------------
For each example the tokenizer receives:

    [CLS] sentence [SEP] context_1 … context_n [SEP]   (with RAG)
    [CLS] sentence [SEP]                                 (without RAG)

The [CLS] token representation is used for binary classification
(fallacy / not-fallacy).  BERT's bidirectional self-attention propagates
knowledge-base signal into the sentence representation.

Retrieval (optional, controlled by --use_rag)
---------------------------------------------
  - BM25  : lexical retrieval via BM25Retriever
  - dense : vector-store similarity / MMR search (ChromaDB)

Summarisation (optional, controlled by --summarization_model)
--------------------------------------------------------------
  - Local  : meta-llama/Llama-3.1-8B-Instruct
             mistralai/Mistral-7B-Instruct-v0.3
  - API    : gpt-4o-mini  (requires OPENAI_API_KEY)
  - None   : raw retrieved passages are appended verbatim (truncated)
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Counter

import datasets
import numpy as np
import rootutils
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

check_min_version("4.57.0.dev0")
require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ModelArguments:
    """Arguments for the encoder model / tokenizer / config."""

    model_name_or_path: str = field(
        default="google-bert/bert-base-uncased",
        metadata={"help": "Pretrained model identifier or local path."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name (defaults to model_name_or_path)."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name (defaults to model_name_or_path)."},
    )
    cache_dir: Optional[str] = field(
        default="./cache/",
        metadata={"help": "Directory for cached pretrained models."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Branch / tag / commit id to use."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace auth token for private models."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow execution of remote code from Hub repos."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Load a model whose head dimensions differ from checkpoint."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer (backed by the tokenizers library)."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for data loading, preprocessing, and retrieval-augmentation."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (via datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name for the dataset."},
    )
    train_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/classification/train.csv",
        metadata={"help": "Training data file (CSV or JSON). Use the rag/ subdirectory when --use_rag is set."},
    )
    validation_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/classification/dev.csv",
        metadata={"help": "Validation data file (CSV or JSON). Use the rag/ subdirectory when --use_rag is set."},
    )
    test_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/classification/test.csv",
        metadata={
            "help": "Test / prediction data file (CSV or JSON). Use the rag/ subdirectory when --use_rag is set."},
    )
    text_column_name: Optional[str] = field(
        default="sentence",
        metadata={"help": "Column containing the input sentence text."},
    )
    speech_turn_column_name: Optional[str] = field(
        default="speech_turn",
        metadata={"help": "Column containing the input speech turn text."},
    )
    context_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "Column containing the RAG context text (used when --use_rag is set)."},
    )
    label_column_name: Optional[str] = field(
        default="fallacy_label",
        metadata={"help": "Column containing the single-label classification label. Used when the "
                          "dataset is single-label (elecdeb / reddit)."},
    )
    label_column_names: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated 0/1 label columns for MULTI-LABEL classification. "
                          "If set, forces multi-label mode. If omitted, the label format is "
                          "auto-detected: a present --label_column_name -> single-label; otherwise "
                          "two or more binary (0/1) non-metadata columns -> multi-label."},
    )
    multi_label_threshold: float = field(
        default=0.5,
        metadata={"help": "Sigmoid probability threshold above which a class is predicted present "
                          "(multi-label mode only)."},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite cached tokenised datasets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for dataset.map()."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Pad all samples to max_seq_length (needed for TPU)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training set for debugging."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate evaluation set for debugging."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate prediction set for debugging."},
    )
    shuffle_train_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Random seed for shuffling the training dataset."},
    )
    oversample_minority: bool = field(
        default=False,
        metadata={"help": "Duplicate rare-class training rows up to ~1/3 of the majority class count "
                          "(single-label only; skipped for multi-label, where labels aren't hashable)."},
    )
    oversample_max_repeats: int = field(
        default=10,
        metadata={"help": "Cap on how many times a single minority-class row can be duplicated, so one "
                          "example isn't cloned hundreds of times when it's extremely rare."},
    )
    # ── RAG-specific ──────────────────────────────────────────────────────
    use_rag: bool = field(
        default=False,
        metadata={"help": "Read the pre-built 'context' field from the dataset and append it to each BERT input."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Provide either --dataset_name or --train_file / --validation_file.")
        for attr in ("train_file", "validation_file"):
            path = getattr(self, attr)
            if path is not None and path.rsplit(".", 1)[-1] not in ("csv", "json"):
                raise ValueError(f"{attr} must be a .csv or .json file.")
        # Convenience: if --use_rag is set and the default no_rag paths are still in place,
        # automatically redirect to the rag/ subdirectory.
        if self.use_rag:
            for attr in ("train_file", "validation_file", "test_file"):
                path = getattr(self, attr)
                if path is not None and "/no_rag/" in path:
                    setattr(self, attr, path.replace("/no_rag/", "/rag/"))


# Columns that are never fallacy labels; used to auto-detect the multi-label 0/1 columns.
METADATA_COLUMNS = {
    "id", "component_id", "year", "sentence", "speech_turn", "speaker",
    "relations", "relations_type", "relations_text", "relations_label",
    "fallacy_text", "fallacy_labels", "components_labels", "component_labels",
    "component_label", "rag_context", "retrieval_query", "has_relations", "text",
}


def get_label_list(raw_datasets, split: str = "train", label_column: str = "label") -> list[str]:
    """Get the sorted list of unique labels from a dataset split."""
    label_list = raw_datasets[split].unique(label_column)
    # Treat labels as strings for consistency with model.config.label2id
    return [str(label) for label in label_list]


def _looks_binary(values) -> bool:
    """True if every non-null value in *values* is 0 or 1 (i.e. an indicator column)."""
    seen = set()
    for v in values:
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return False
        if iv not in (0, 1):
            return False
        seen.add(iv)
    return len(seen) > 0


def detect_label_columns(raw_datasets, data_args, ref_split: str) -> tuple[bool, list[str]]:
    """Decide single- vs multi-label and return ``(is_multi_label, label_columns)``.

    Priority:
      1. Explicit ``--label_column_names`` (comma-separated) -> multi-label with those columns.
      2. A present single ``--label_column_name`` -> single-label.
      3. Otherwise auto-detect: >= 2 binary (0/1) non-metadata columns -> multi-label.
    """
    columns = raw_datasets[ref_split].column_names

    if data_args.label_column_names:
        cols = [c.strip() for c in data_args.label_column_names.split(",") if c.strip()]
        missing = [c for c in cols if c not in columns]
        if missing:
            raise ValueError(f"--label_column_names lists columns not present in the data: {missing}")
        return True, cols

    if data_args.label_column_name in columns:
        return False, [data_args.label_column_name]

    candidates = [
        c for c in columns
        if c not in METADATA_COLUMNS and _looks_binary(raw_datasets[ref_split].unique(c))
    ]
    if len(candidates) >= 2:
        return True, candidates

    raise ValueError(
        f"Label column '{data_args.label_column_name}' is absent and could not auto-detect multi-label "
        f"0/1 columns (candidates found: {candidates}). Pass --label_column_names explicitly."
    )


# Normalized names that denote the negative "no fallacy" class.
_NONE_LABEL_KEYS = {"noneoftheabove", "none", "nofallacy", "no"}


def _find_none_index(label_list: list[str]) -> Optional[int]:
    """Return the index of the negative 'no fallacy' class in *label_list*, or None."""
    for i, label in enumerate(label_list):
        key = "".join(ch for ch in label.lower() if ch.isalnum())
        if key in _NONE_LABEL_KEYS:
            return i
    return None


def _apply_none_exclusivity(preds: np.ndarray, none_idx: Optional[int]) -> np.ndarray:
    """Enforce the 'none' class rules on multi-label predictions.

    - Exclusivity: if the 'none' class fires, it is the sole label (drop all others).
    - Empty default: a row with no label above threshold falls back to 'none'.
    """
    if none_idx is None:
        return preds
    preds = preds.copy()
    none_on = preds[:, none_idx] == 1
    if none_on.any():
        preds[none_on, :] = 0
        preds[none_on, none_idx] = 1
    empty = preds.sum(axis=1) == 0
    if empty.any():
        preds[empty, none_idx] = 1
    return preds


def make_multilabel_compute_metrics(label_list: list[str], threshold: float = 0.5):
    """Return a compute_metrics function for multi-label classification (sigmoid + threshold)."""
    none_idx = _find_none_index(label_list)

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > threshold).astype(int)
        # 'None of the above' is the sole answer when it fires; empty rows default to it.
        preds = _apply_none_exclusivity(preds, none_idx)
        labels = np.array(p.label_ids).astype(int)

        metrics = {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "f1_samples": f1_score(labels, preds, average="samples", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            # Exact-match ratio: the fraction of samples whose entire label set is correct.
            "subset_accuracy": accuracy_score(labels, preds),
        }
        # Per-class F1 for error analysis (mirrors the paper's per-class report).
        per_class = f1_score(labels, preds, average=None, zero_division=0)
        for label, score in zip(label_list, per_class):
            metrics[f"f1_{label}"] = float(score)
        return metrics

    return compute_metrics


def make_compute_metrics(label_list: list[str]):
    """Return a compute_metrics function for binary sentence classification."""

    def compute_metrics(p: EvalPrediction):
        predictions = np.argmax(p.predictions, axis=1)
        labels = p.label_ids

        # Map integer indices back to string labels for sklearn
        pred_labels = [label_list[pred] for pred in predictions]
        true_labels = [label_list[lbl] for lbl in labels]

        metrics = {
            "accuracy": accuracy_score(y_true=true_labels, y_pred=pred_labels),
            "precision": precision_score(y_true=true_labels, y_pred=pred_labels, average="macro", zero_division=0),
            "recall": recall_score(y_true=true_labels, y_pred=pred_labels, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true=true_labels, y_pred=pred_labels, average="macro", zero_division=0),
        }
        # Per-class F1 for error analysis
        per_class = f1_score(
            y_true=true_labels, y_pred=pred_labels, labels=label_list, average=None, zero_division=0
        )
        for label, score in zip(label_list, per_class):
            metrics[f"f1_{label}"] = float(score)
        return metrics

    return compute_metrics


def _prepare_split(raw_datasets, split_name, max_samples, tokenize_fn, map_kwargs,
                   cast_label_int: bool = True):
    """Optionally slice a split and tokenize it.

    ``cast_label_int`` forces the single-label integer ``label`` to int64 (string/NER-style
    sources can otherwise be inferred as list[int]). It is disabled for multi-label, whose
    ``label`` is a per-example vector of floats.
    """
    from datasets import Value
    ds = raw_datasets[split_name]
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    # Drop every column except 'label' (raw text columns would crash the collator); keep the
    # 'label' that preprocess_function writes.
    non_label_cols = [c for c in ds.column_names if c != "label"]
    tokenized = ds.map(
        tokenize_fn,
        desc=f"Tokenising {split_name}",
        remove_columns=non_label_cols,
        **map_kwargs,
    )
    if cast_label_int and "label" in tokenized.column_names:
        tokenized = tokenized.cast_column("label", Value("int64"))
    return tokenized


def main():
    """Load data, tokenize with optional RAG context, and train/evaluate/predict the classifier."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_process_index}, "
        f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed: {training_args.parallel_mode.value == 'distributed'}, "
        f"fp16: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    # ── Dataset loading ───────────────────────────────────────────────────
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file,
    }
    extension = (data_args.train_file or data_args.validation_file).rsplit(".", 1)[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # ── Label setup (auto-detect single- vs multi-label) ──────────────────
    ref_split = "train" if training_args.do_train else "validation"
    is_multi_label, label_columns = detect_label_columns(raw_datasets, data_args, ref_split)

    if is_multi_label:
        # Each fallacy class (incl. "None of the above") is its own 0/1 column.
        # Column order defines the output-neuron order and the multi-hot target order.
        label_list = list(label_columns)
        num_labels = len(label_list)
        label_to_id = {v: i for i, v in enumerate(label_list)}
        logger.info(f"Detected MULTI-LABEL setup with {num_labels} classes: {label_list}")
    else:
        label_column_name = label_columns[0]
        # Rename to "label" so the Trainer can find it automatically
        if label_column_name != "label":
            for key in raw_datasets:
                raw_datasets[key] = raw_datasets[key].rename_column(label_column_name, "label")
            label_column_name = "label"

        label_list = get_label_list(raw_datasets, split=ref_split, label_column=label_column_name)

        # Merge labels across splits to handle labels that appear only in val/test
        for split in ["validation", "test"]:
            if split in raw_datasets:
                extra = get_label_list(raw_datasets, split=split, label_column=label_column_name)
                diff = set(extra) - set(label_list)
                if diff:
                    logger.warning(f"Labels {diff} in {split} set but not in training set — adding them.")
                    label_list += list(diff)

        label_list.sort()
        num_labels = len(label_list)
        label_to_id = {v: i for i, v in enumerate(label_list)}
        logger.info(f"Detected SINGLE-LABEL setup with {num_labels} classes: {label_list}")

    # ── Model / tokenizer / config ────────────────────────────────────────
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.problem_type = "multi_label_classification" if is_multi_label else "single_label_classification"
    logger.info(f"Setting problem type to {config.problem_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if config.model_type in {"bloom", "gpt2", "roberta", "deberta"}:
        tokenizer.add_prefix_space = True

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
    )

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}

    # ── Tokenisation ──────────────────────────────────────────────────────
    # Use the model's own capacity; some tokenizers report a huge sentinel, so fall back safely.
    max_seq_length = tokenizer.model_max_length
    if max_seq_length is None or max_seq_length > 100_000:
        max_seq_length = getattr(config, "max_position_embeddings", 512)
    logger.info(f"Using model maximum sequence length: {max_seq_length}")
    padding = "max_length" if data_args.pad_to_max_length else False

    text_column_name = data_args.text_column_name
    speech_turn_column_name = data_args.speech_turn_column_name
    context_column_name = data_args.context_column_name

    def preprocess_function(examples):
        """Tokenize sentence (segment A) and speech_turn + optional RAG context (segment B)."""
        if data_args.use_rag and context_column_name:
            # speech_turn comes first so truncation trims rag_context, not the local context.
            if speech_turn_column_name == text_column_name:
                # reddit-dataset
                result = tokenizer(
                    examples[text_column_name],
                    examples[context_column_name],
                    padding=padding,
                    max_length=max_seq_length,
                    truncation=True,
                )
            else:
                combined_b = [
                    f"{st} {ctx}" if (ctx is not None and isinstance(ctx, str) and ctx.strip()) else st
                    for st, ctx in zip(examples[speech_turn_column_name], examples[context_column_name])
                ]
                result = tokenizer(
                    examples[text_column_name],
                    combined_b,
                    padding=padding,
                    max_length=max_seq_length,
                    truncation=True,
                )
        else:
            result = tokenizer(
                examples[text_column_name],
                examples[speech_turn_column_name],
                padding=padding,
                max_length=max_seq_length,
                truncation=True,
            )

        if is_multi_label:
            # Multi-hot float target per example, built from the 0/1 label columns
            # (the original columns are still present here; map removes them afterwards).
            n_rows = len(examples[text_column_name])
            result["label"] = [
                [float(examples[col][i]) for col in label_columns] for i in range(n_rows)
            ]
        elif "label" in examples:
            result["label"] = [
                (label_to_id[str(lbl)] if lbl != -1 else -1) for lbl in examples["label"]
            ]
        return result

    map_kwargs = dict(
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    split_configs = [
        (training_args.do_train, "train", data_args.max_train_samples, "--do_train", data_args.train_file),
        (training_args.do_eval, "validation", data_args.max_eval_samples, "--do_eval", data_args.validation_file),
        (training_args.do_predict, "test", data_args.max_predict_samples, "--do_predict", data_args.test_file),
    ]

    prepared = {}
    for do_split, split_name, max_samples, flag, split_file in split_configs:
        if not do_split:
            continue
        if split_name not in raw_datasets:
            raise ValueError(f"{flag} requires a {split_name} split.")
        with training_args.main_process_first(desc=f"{split_name} tokenisation"):
            prepared[split_name] = _prepare_split(
                raw_datasets, split_name, max_samples, preprocess_function, map_kwargs,
                cast_label_int=not is_multi_label,
            )

    train_dataset = prepared.get("train")
    eval_dataset = prepared.get("validation")
    predict_dataset = prepared.get("test")

    if training_args.do_train and data_args.shuffle_train_dataset:
        logger.info("Shuffling the training dataset")
        train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)

    # Log a few random samples from the training set
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # ── Data collator ─────────────────────────────────────────────────────
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # ── Class weights (from train set) ────────────────────────────────────
    # Single-label: per-class inverse-frequency weights for cross-entropy.
    # Multi-label: per-class pos_weight = #neg/#pos for BCE, up-weighting rare classes.
    class_weights = None
    pos_weight = None
    if training_args.do_train and train_dataset is not None:
        if is_multi_label:
            labels_matrix = np.asarray(train_dataset["label"], dtype=np.float64)  # (N, C)
            pos = labels_matrix.sum(axis=0)
            neg = labels_matrix.shape[0] - pos
            # Guard against divide-by-zero for a class with no positives in train.
            weights = neg / np.clip(pos, 1.0, None)
            pos_weight = torch.tensor(weights, dtype=torch.float)
            logger.info(
                f"Multi-label pos_weight: { {label_list[i]: round(float(w), 4) for i, w in enumerate(weights)} }"
            )
        else:
            label_counts = Counter(train_dataset["label"])
            total = sum(label_counts.values())
            weights = [total / (num_labels * label_counts[i]) for i in range(num_labels)]
            class_weights = torch.tensor(weights, dtype=torch.float)
            logger.info(f"Class weights: { {label_list[i]: round(w, 4) for i, w in enumerate(weights)} }")

    # ── Loss functions ────────────────────────────────────────────────────
    def weighted_loss_fn(outputs, labels, num_items_in_batch=None):
        """Class-weighted cross-entropy loss to offset label imbalance (single-label)."""
        logits = outputs.logits.float()
        weight = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
        return torch.nn.CrossEntropyLoss(weight=weight)(logits, labels)

    def multilabel_loss_fn(outputs, labels, num_items_in_batch=None):
        """pos_weight BCEWithLogitsLoss to offset per-class imbalance (multi-label)."""
        logits = outputs.logits.float()
        pw = pos_weight.to(device=logits.device, dtype=logits.dtype) if pos_weight is not None else None
        return torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels.float())

    if is_multi_label:
        compute_metrics_fn = make_multilabel_compute_metrics(label_list, data_args.multi_label_threshold)
        compute_loss_func = multilabel_loss_fn
    else:
        compute_metrics_fn = make_compute_metrics(label_list)
        compute_loss_func = weighted_loss_fn

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        compute_loss_func=compute_loss_func,
    )

    # ── Training ──────────────────────────────────────────────────────────
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ── Evaluation ────────────────────────────────────────────────────────
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ── Prediction ────────────────────────────────────────────────────────
    if training_args.do_predict:
        logger.info("*** Predict ***")
        metrics = trainer.evaluate(eval_dataset=predict_dataset, metric_key_prefix="predict")
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if "label" in predict_dataset.features:
            label_column = predict_dataset['label']
            predict_dataset = predict_dataset.remove_columns("label")
        raw_predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        raw_predictions = raw_predictions[0] if isinstance(raw_predictions, tuple) else raw_predictions

        if trainer.is_world_process_zero():
            output_path = os.path.join(training_args.output_dir, "predict_results.txt")
            with open(output_path, "w") as fh:
                logger.info("***** Predict results *****")
                if is_multi_label:
                    probs = 1.0 / (1.0 + np.exp(-raw_predictions))
                    preds = (probs > data_args.multi_label_threshold).astype(int)
                    # 'None of the above' is the sole answer when it fires; empty rows default to it.
                    preds = _apply_none_exclusivity(preds, _find_none_index(label_list))
                    gold = np.array(label_column).astype(int)
                    # One 0/1 column per class, plus readable ';'-joined predicted / gold label sets.
                    fh.write("index\t" + "\t".join(label_list) + "\tprediction_labels\tgold_labels\n")
                    for index, (prow, grow) in enumerate(zip(preds, gold)):
                        pred_names = ";".join(l for l, v in zip(label_list, prow) if v) or "-"
                        gold_names = ";".join(l for l, v in zip(label_list, grow) if v) or "-"
                        bits = "\t".join(str(int(v)) for v in prow)
                        fh.write(f"{index}\t{bits}\t{pred_names}\t{gold_names}\n")
                else:
                    predictions = np.argmax(raw_predictions, axis=1)
                    fh.write("index\tprediction\tgold\n")
                    for index, item in enumerate(zip(predictions, np.array(label_column))):
                        fh.write(f"{index}\t{label_list[item[0]]}\t{label_list[item[1]]}\n")
            logger.info(f"Predictions saved to {output_path}")

    # ── Model card / Hub push ─────────────────────────────────────────────
    hub_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        hub_kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            hub_kwargs["dataset_args"] = data_args.dataset_config_name
            hub_kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            hub_kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**hub_kwargs)
    else:
        trainer.create_model_card(**hub_kwargs)


if __name__ == "__main__":
    main()
