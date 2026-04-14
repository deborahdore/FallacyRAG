"""
Fine-tuning BERT-style models for fallacy detection using BIO tags,
augmented with retrieved knowledge-base context (RAG).

Architecture
------------
For each example the tokenizer receives:

    [CLS] tok_1 … tok_n [SEP] summary_1 [SEP] summary_2 … [SEP]

Labels are assigned only to phrase tokens; context positions get –100
so CrossEntropyLoss ignores them.  BERT's bidirectional self-attention
propagates knowledge-base signal into every phrase-token representation.

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
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import numpy as np
import rootutils
import transformers
from datasets import ClassLabel, load_dataset
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedConfig,
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


@dataclass
class DataTrainingArguments:
    """Arguments for data loading, preprocessing, and retrieval-augmentation."""

    task_name: Optional[str] = field(
        default="ner",
        metadata={"help": "Task name (ner, pos, …)."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (via datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name for the dataset."},
    )
    train_file: Optional[str] = field(
        default="./dataset/detection/train.json",
        metadata={"help": "Training data file (CSV or JSON). Use the rag/ subdirectory when --use_rag is set."},
    )
    validation_file: Optional[str] = field(
        default="./dataset/detection/dev.json",
        metadata={"help": "Validation data file (CSV or JSON). Use the rag/ subdirectory when --use_rag is set."},
    )
    test_file: Optional[str] = field(
        default="./dataset/detection/test.json",
        metadata={
            "help": "Test / prediction data file (CSV or JSON). Use the rag/ subdirectory when --use_rag is set."},
    )
    text_column_name: Optional[str] = field(
        default="tokens",
        metadata={"help": "Column containing the pre-tokenised words."},
    )
    context_column_name: Optional[str] = field(
        default="context",
        metadata={"help": "Column containing the context tokens."},
    )
    label_column_name: Optional[str] = field(
        default="ner_tags",
        metadata={"help": "Column containing BIO label sequences."},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite cached tokenised datasets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for dataset.map()."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum total input length after tokenisation."},
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
    label_all_tokens: bool = field(
        default=True,
        metadata={"help": "Propagate word-level labels to all sub-word tokens."},
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
        self.task_name = self.task_name.lower()
        if self.max_seq_length is None:
            self.max_seq_length = 512
        # Convenience: if --use_rag is set and the default no_rag paths are still in place,
        # automatically redirect to the rag/ subdirectory.
        if self.use_rag:
            for attr in ("train_file", "validation_file", "test_file"):
                path = getattr(self, attr)
                if path is not None and "/no_rag/" in path:
                    setattr(self, attr, path.replace("/no_rag/", "/rag/"))


def make_compute_metrics(label_list: list[str], eval_do_concat_batches: bool):
    """Return a compute_metrics function for token classification."""

    def compute_metrics(p):
        predictions, labels = p

        if not eval_do_concat_batches:
            predictions = np.hstack(predictions)
            labels = np.hstack(labels)

        predictions = np.argmax(predictions, axis=2)

        # Filter out padding (-100) and flatten in one pass per tensor.
        flat_preds = list(chain.from_iterable(
            [label_list[pred] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ))
        flat_labels = list(chain.from_iterable(
            [label_list[lbl] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ))

        return {
            "accuracy": accuracy_score(y_true=flat_labels, y_pred=flat_preds),
            "precision": precision_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
            "recall": recall_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
        }

    return compute_metrics


def _prepare_split(raw_datasets, split_name, max_samples, tokenize_fn, map_kwargs):
    """Optionally slice a split and tokenize it."""
    ds = raw_datasets[split_name]
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    #  FOR DEBUG:
    # debug_batch = ds[:2]
    # debug_result = tokenize_fn(debug_batch)  # noqa: F841  ← set breakpoint here
    return ds.map(tokenize_fn, desc=f"Tokenising {split_name}", **map_kwargs)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file,
    }
    extension = (data_args.train_file or data_args.validation_file).rsplit(".", 1)[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    features = (
        raw_datasets["train"].features
        if training_args.do_train
        else raw_datasets["validation"].features
    )
    text_column_name = data_args.text_column_name
    label_column_name = data_args.label_column_name
    context_column_name = data_args.context_column_name

    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = sorted({lbl for seq in raw_datasets["train"][label_column_name] for lbl in seq})
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if config.model_type in {"bloom", "gpt2", "roberta", "deberta"}:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    if not (hasattr(tokenizer, "_tokenizer") or getattr(tokenizer, "is_fast", False)):
        raise TypeError(
            "This script requires a fast tokenizer. "
            "See https://huggingface.co/transformers/index.html#supported-frameworks."
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if model.config.label2id != PreTrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Model labels do not match dataset labels — ignoring model labels.\n"
                f"  model:   {sorted(model.config.label2id.keys())}\n"
                f"  dataset: {sorted(label_list)}"
            )

    sorted_label_list = sorted(label_list)
    model.config.label2id = {l: i for i, l in enumerate(sorted_label_list)}
    model.config.id2label = dict(enumerate(sorted_label_list))

    # B-Xxx → I-Xxx map for sub-word token propagation.
    b_to_i_label = [
        label_list.index(lbl.replace("B-", "I-"))
        if lbl.startswith("B-") and lbl.replace("B-", "I-") in label_list
        else idx
        for idx, lbl in enumerate(label_list)
    ]

    def tokenize_and_align_labels(examples):
        phrase_tokenised = tokenizer(
            examples[text_column_name],
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True,
            max_length=data_args.max_seq_length,
            is_split_into_words=True,
            add_special_tokens=True,
        )

        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_token_type_ids: list[list[int]] = []
        all_labels: list[list[int]] = []

        for i, word_labels in enumerate(examples[label_column_name]):
            word_ids = phrase_tokenised.word_ids(batch_index=i)
            phrase_input_ids = phrase_tokenised["input_ids"][i]

            # Align BIO labels to sub-word tokens.
            phrase_label_ids: list[int] = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    phrase_label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    phrase_label_ids.append(label_to_id[word_labels[word_idx]])
                else:
                    phrase_label_ids.append(
                        b_to_i_label[label_to_id[word_labels[word_idx]]]
                        if data_args.label_all_tokens else -100
                    )
                prev_word_idx = word_idx

            if data_args.use_rag:
                context_str = examples[context_column_name][i]
                ctx_ids = tokenizer(context_str, add_special_tokens=False, truncation=True,
                                    max_length=data_args.max_seq_length - len(phrase_input_ids) - 1)["input_ids"]

                sep_id = tokenizer.sep_token_id
                # Layout: [CLS] phrase… [SEP] context… [SEP]
                # Segment 0 = phrase, Segment 1 = context (standard BERT pair encoding).
                combined_ids = phrase_input_ids[:-1] + [sep_id] + ctx_ids + [sep_id]
                combined_labels = phrase_label_ids[:-1] + [-100] * (len(ctx_ids) + 2)
                combined_types = [0] * len(phrase_input_ids[:-1]) + [1] * (len(ctx_ids) + 2)
                combined_attn = [1] * len(combined_ids)

                all_input_ids.append(combined_ids)
                all_attention_masks.append(combined_attn)
                all_token_type_ids.append(combined_types)
                all_labels.append(combined_labels)
            else:
                all_input_ids.append(phrase_input_ids)
                all_attention_masks.append(phrase_tokenised["attention_mask"][i])
                all_token_type_ids.append(
                    phrase_tokenised["token_type_ids"][i]
                    if "token_type_ids" in phrase_tokenised
                    else [0] * len(phrase_input_ids)
                )
                all_labels.append(phrase_label_ids)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "token_type_ids": all_token_type_ids,
            "labels": all_labels,
        }

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
            prepared[split_name] = _prepare_split(raw_datasets,
                                                  split_name,
                                                  max_samples,
                                                  tokenize_and_align_labels,
                                                  map_kwargs)

    train_dataset = prepared.get("train")
    eval_dataset = prepared.get("validation")
    predict_dataset = prepared.get("test")

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_list, training_args.eval_do_concat_batches),
    )

    # ── Training ──────────────────────────────────────────────────────────
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ── Evaluation ────────────────────────────────────────────────────────
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ── Prediction ────────────────────────────────────────────────────────
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[pred] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            output_path = os.path.join(training_args.output_dir, "predictions.txt")
            with open(output_path, "w") as fh:
                fh.writelines(" ".join(pred_seq) + "\n" for pred_seq in true_predictions)
            logger.info(f"Predictions saved to {output_path}")

    # ── Model card / Hub push ─────────────────────────────────────────────
    hub_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
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
