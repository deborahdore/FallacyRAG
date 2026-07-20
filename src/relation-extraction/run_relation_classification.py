"""
Fine-tuning BERT-style models for relation classification, augmented with retrieved
knowledge-base context (RAG).

Task
----
Multi-class (single-label) classification over three relation labels:

    support | attack | no_relation

A softmax head is trained with class-weighted cross-entropy and decoded with argmax.

Architecture
------------
For each example the tokenizer receives:

    [CLS] sentence [SEP] context_1 … context_n [SEP]   (with RAG)
    [CLS] sentence [SEP]                                 (without RAG)

The [CLS] token representation is used for classification. BERT's bidirectional
self-attention propagates knowledge-base signal into the sentence representation.

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
        default="relation_label",
        metadata={"help": "Column containing the relation classification label "
                          "(one of: support, attack, no_relation)."},
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


def get_label_list(raw_datasets, split: str = "train", label_column: str = "label") -> list[str]:
    """Get the sorted list of unique labels from a dataset split."""
    label_list = raw_datasets[split].unique(label_column)
    # Treat labels as strings for consistency with model.config.label2id
    return [str(label) for label in label_list]


def make_compute_metrics(label_list: list[str]):
    """Return a compute_metrics function for multi-class sentence classification."""

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


def _prepare_split(raw_datasets, split_name, max_samples, tokenize_fn, map_kwargs):
    """Optionally slice a split and tokenize it.

    The single-label integer ``label`` is forced to int64 (string/NER-style sources can
    otherwise be inferred as list[int]).
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
    if "label" in tokenized.column_names:
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

    # ── Label setup (multi-class single-label) ────────────────────────────
    ref_split = "train" if training_args.do_train else "validation"
    label_column_name = data_args.label_column_name
    if label_column_name not in raw_datasets[ref_split].column_names:
        raise ValueError(
            f"Label column '{label_column_name}' not found in the data. "
            f"Available columns: {raw_datasets[ref_split].column_names}"
        )

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
    logger.info(f"Multi-class setup with {num_labels} classes: {label_list}")

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
    config.problem_type = "single_label_classification"
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
    # Always use the model's own capacity as the maximum sequence length
    # (e.g. 512 for BERT, 4096 for Longformer). Some tokenizers report a
    # huge sentinel value (e.g. 1e30) when no limit is configured, so fall
    # back to the config's positional-embedding size or a safe default.
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

        if "label" in examples:
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
    # Per-class inverse-frequency weights for cross-entropy, to offset label imbalance.
    class_weights = None
    if training_args.do_train and train_dataset is not None:
        label_counts = Counter(train_dataset["label"])
        total = sum(label_counts.values())
        weights = [total / (num_labels * label_counts[i]) for i in range(num_labels)]
        class_weights = torch.tensor(weights, dtype=torch.float)
        logger.info(f"Class weights: { {label_list[i]: round(w, 4) for i, w in enumerate(weights)} }")

    # ── Loss function ─────────────────────────────────────────────────────
    def weighted_loss_fn(outputs, labels, num_items_in_batch=None):
        """Class-weighted cross-entropy loss to offset label imbalance."""
        logits = outputs.logits.float()
        weight = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
        return torch.nn.CrossEntropyLoss(weight=weight)(logits, labels)

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
