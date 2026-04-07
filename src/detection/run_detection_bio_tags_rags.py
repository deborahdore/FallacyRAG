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
from typing import Optional

import datasets
import numpy as np
import rootutils
import torch
import transformers
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedConfig,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed, AutoModelForTokenClassification,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from perform_retrieval import SEARCH_TYPES, load_bm25_retriever, load_vector_store

check_min_version("4.57.0.dev0")
require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

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
        default="./dataset/detection/bio_tags/train.json",
        metadata={"help": "Training data file (CSV or JSON)."},
    )
    validation_file: Optional[str] = field(
        default="./dataset/detection/bio_tags/dev.json",
        metadata={"help": "Validation data file (CSV or JSON)."},
    )
    test_file: Optional[str] = field(
        default="./dataset/detection/bio_tags/test.json",
        metadata={"help": "Test / prediction data file (CSV or JSON)."},
    )
    text_column_name: Optional[str] = field(
        default="tokens",
        metadata={"help": "Column containing the pre-tokenised words."},
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
        metadata={"help": "Augment BERT input with retrieved knowledge-base context."},
    )
    search_type: str = field(
        default="similarity",
        metadata={"help": f"Retrieval strategy. One of: {list(SEARCH_TYPES.keys())}."},
    )
    embeddings_model: str = field(
        default="google-bert/bert-base-uncased",
        metadata={"help": "Encoder for dense retrieval embeddings."},
    )
    persist_directory_path: str = field(
        default="./chromadb",
        metadata={"help": "ChromaDB persistence directory for dense search."},
    )
    chunk_size: int = field(
        default=800,
        metadata={"help": "Document chunk size for BM25 retriever."},
    )
    chunk_overlap: int = field(
        default=80,
        metadata={"help": "Chunk overlap for BM25 retriever."},
    )
    top_k: int = field(
        default=3,
        metadata={"help": "Number of documents to retrieve per example."},
    )
    summarization_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "LLM used to summarise retrieved passages before appending to BERT input. "
                "Supported: 'meta-llama/Llama-3.1-8B-Instruct', "
                "'mistralai/Mistral-7B-Instruct-v0.3', 'gpt-4o-mini'. "
                "If None, passages are appended verbatim (truncated to fit the budget)."
            )
        },
    )
    context_budget: int = field(
        default=128,
        metadata={"help": "Maximum tokens reserved for retrieved context (per example)."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Provide either --dataset_name or --train_file / --validation_file.")
        for attr in ("train_file", "validation_file"):
            path = getattr(self, attr)
            if path is not None:
                ext = path.rsplit(".", 1)[-1]
                if ext not in ("csv", "json"):
                    raise ValueError(f"{attr} must be a .csv or .json file.")
        self.task_name = self.task_name.lower()
        if self.max_seq_length is None:
            self.max_seq_length = 512


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def search_knowledge_base(
        query: str,
        *,
        search_type: str,
        embeddings_model: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        persist_directory_path: str,
) -> list[str]:
    """Return a list of raw document strings for *query*."""
    if search_type == "bm25":
        retriever = load_bm25_retriever(chunk_size, chunk_overlap, top_k)
    else:
        search_config = SEARCH_TYPES[search_type]
        vector_store = load_vector_store(embeddings_model, persist_directory_path)
        search_kwargs = search_config["search_kwargs"]({"top_k": top_k})
        retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    docs = retriever.invoke(query)
    # LangChain documents expose .page_content; plain strings pass through.
    return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]


def summarize_docs(docs: list[str], summarization_model: str) -> list[str]:
    """Summarise each retrieved passage with the chosen model."""
    prompt_template = "Summarize the following text in a concise paragraph:\n\n{}"
    summaries: list[str] = []

    LOCAL_MODELS = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }

    if summarization_model in LOCAL_MODELS:
        _tokenizer = AutoTokenizer.from_pretrained(summarization_model)
        _model = AutoModelForCausalLM.from_pretrained(
            summarization_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer)
        for doc in docs:
            messages = [{"role": "user", "content": prompt_template.format(doc)}]
            formatted = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            output = pipe(formatted, max_new_tokens=256, do_sample=False, return_full_text=False)
            summaries.append(output[0]["generated_text"].strip())

    elif summarization_model == "gpt-4o-mini":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        for doc in docs:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_template.format(doc)}],
                max_tokens=256,
            )
            summaries.append(response.choices[0].message.content.strip())

    else:
        raise ValueError(
            f"Unknown summarization model '{summarization_model}'. "
            "Choose from: 'meta-llama/Llama-3.1-8B-Instruct', "
            "'mistralai/Mistral-7B-Instruct-v0.3', 'gpt-4o-mini'."
        )

    return summaries


def build_rag_context(
        phrase_text: str,
        *,
        data_args: DataTrainingArguments,
        summarization_model: Optional[str],
) -> str:
    """
    Retrieve and (optionally) summarise documents for *phrase_text*.
    Returns a single string that will be appended after the phrase [SEP].
    """
    passages = search_knowledge_base(
        phrase_text,
        search_type=data_args.search_type,
        embeddings_model=data_args.embeddings_model,
        chunk_size=data_args.chunk_size,
        chunk_overlap=data_args.chunk_overlap,
        top_k=data_args.top_k,
        persist_directory_path=data_args.persist_directory_path,
    )

    if not passages:
        return ""

    if summarization_model:
        passages = summarize_docs(passages, summarization_model)

    return " [SEP] ".join(passages)


def make_tokenize_fn(
        tokenizer,
        text_column_name: str,
        label_column_name: str,
        label_to_id: dict,
        b_to_i_label: list[int],
        padding,
        data_args: DataTrainingArguments,
):
    """
    Return a batched map-function that:
      1. Tokenizes the word-level phrase.
      2. Optionally retrieves and appends context tokens (RAG).
      3. Aligns BIO labels with sub-word tokens, masking context with –100.
    """

    def tokenize_and_align_labels(examples):
        phrase_texts = [" ".join(toks) for toks in examples[text_column_name]]

        phrase_tokenised = tokenizer(
            examples[text_column_name],
            padding=False,
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

            phrase_label_ids: list[int] = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    phrase_label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    phrase_label_ids.append(label_to_id[word_labels[word_idx]])
                else:
                    if data_args.label_all_tokens:
                        phrase_label_ids.append(
                            b_to_i_label[label_to_id[word_labels[word_idx]]]
                        )
                    else:
                        phrase_label_ids.append(-100)
                prev_word_idx = word_idx

            if data_args.use_rag:
                context_str = build_rag_context(
                    phrase_texts[i],
                    data_args=data_args,
                    summarization_model=data_args.summarization_model,
                )

                if context_str:
                    # Budget: total budget minus what the phrase already uses.
                    phrase_len = len(phrase_input_ids)
                    budget = min(
                        data_args.context_budget,
                        data_args.max_seq_length - phrase_len - 1,  # –1 for final [SEP]
                    )

                    if budget > 0:
                        ctx_enc = tokenizer(
                            context_str,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=budget,
                        )
                        ctx_ids = ctx_enc["input_ids"]

                        # Remove the trailing [SEP] from the phrase, append
                        # context, then re-add [SEP].
                        # phrase_input_ids = [CLS] t1 … tn [SEP]
                        sep_id = tokenizer.sep_token_id
                        combined_ids = (
                                phrase_input_ids[:-1]  # drop trailing [SEP]
                                + [sep_id]  # [SEP] between phrase and context
                                + ctx_ids
                                + [sep_id]  # final [SEP]
                        )
                        # Context tokens all get –100; phrase labels unchanged.
                        ctx_label_mask = [-100] * (len(ctx_ids) + 2)  # +2 for the two extra [SEP]s
                        combined_labels = phrase_label_ids[:-1] + ctx_label_mask

                        # Token type ids: 0 for phrase, 1 for context.
                        phrase_type_ids = [0] * len(phrase_input_ids[:-1])
                        ctx_type_ids = [1] * (len(ctx_ids) + 2)
                        combined_type_ids = phrase_type_ids + ctx_type_ids

                        combined_attention = [1] * len(combined_ids)

                        all_input_ids.append(combined_ids)
                        all_attention_masks.append(combined_attention)
                        all_token_type_ids.append(combined_type_ids)
                        all_labels.append(combined_labels)
                        continue  # skip the non-RAG path below

            # ── Non-RAG path (or empty context) ──────────────────────────
            all_input_ids.append(phrase_input_ids)
            all_attention_masks.append(phrase_tokenised["attention_mask"][i])
            if "token_type_ids" in phrase_tokenised:
                all_token_type_ids.append(phrase_tokenised["token_type_ids"][i])
            else:
                all_token_type_ids.append([0] * len(phrase_input_ids))
            all_labels.append(phrase_label_ids)

        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "token_type_ids": all_token_type_ids,
            "labels": all_labels,
            "text": phrase_texts,
        }
        return result

    return tokenize_and_align_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(label_list: list[str], eval_do_concat_batches: bool):
    def flatten(xss):
        return [x for xs in xss for x in xs]

    def compute_metrics(p):
        predictions, labels = p

        if not eval_do_concat_batches:
            predictions = np.hstack(predictions)
            labels = np.hstack(labels)

        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[pred] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lbl] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
            for row_pred, row_lbl in zip(predictions, labels)
        ]

        flat_preds = flatten(true_predictions)
        flat_labels = flatten(true_labels)

        return {
            "accuracy": accuracy_score(y_true=flat_labels, y_pred=flat_preds),
            "precision": precision_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
            "recall": recall_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true=flat_labels, y_pred=flat_preds, average="macro", zero_division=0),
        }

    return compute_metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Logging setup ─────────────────────────────────────────────────────
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

    def get_label_list(labels):
        return sorted({lbl for seq in labels for lbl in seq})

    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
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

    model.config.label2id = {l: i for i, l in enumerate(sorted(label_list))}
    model.config.id2label = dict(enumerate(label_list))

    # B-Xxx → I-Xxx map for sub-word token propagation.
    b_to_i_label = [
        label_list.index(lbl.replace("B-", "I-"))
        if lbl.startswith("B-") and lbl.replace("B-", "I-") in label_list
        else idx
        for idx, lbl in enumerate(label_list)
    ]

    padding = "max_length" if data_args.pad_to_max_length else False

    tokenize_fn = make_tokenize_fn(
        tokenizer=tokenizer,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        label_to_id=label_to_id,
        b_to_i_label=b_to_i_label,
        padding=padding,
        data_args=data_args,
    )

    map_kwargs = dict(
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    train_dataset = eval_dataset = predict_dataset = None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train split.")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        with training_args.main_process_first(desc="train tokenisation"):
            train_dataset = train_dataset.map(tokenize_fn, desc="Tokenising train", **map_kwargs)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation split.")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
        with training_args.main_process_first(desc="validation tokenisation"):
            eval_dataset = eval_dataset.map(tokenize_fn, desc="Tokenising validation", **map_kwargs)

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test split.")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))
        with training_args.main_process_first(desc="test tokenisation"):
            predict_dataset = predict_dataset.map(tokenize_fn, desc="Tokenising test", **map_kwargs)

    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(
            label_list, training_args.eval_do_concat_batches
        ),
    )

    # ── Training ──────────────────────────────────────────────────────────
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
