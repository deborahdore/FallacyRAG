"""LLM-based fallacy classification: zero-shot, few-shot, and LoRA/QLoRA fine-tuning.
Each example becomes an instruction-following chat prompt, carrying the pre-built "context"
column when ``--use_rag``, and the free-text response is parsed back into canonical labels by
fuzzy alias matching. Single- vs multi-label and the system prompt are both derived from the
dataset. Backends: local HuggingFace generate() or the OpenAI chat API."""

from __future__ import annotations

import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import rootutils
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils.dataset_prompts import get_classification_prompt, get_label_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

ALL_LABELS: list[str] = []
LABEL_TO_LLM: dict[str, str] = {}
ALIAS_TO_LABEL: dict[str, str] = {}
_LABEL_MENU: str = ""

IS_MULTI_LABEL: bool = False

SINGLE_LABEL_NONE: Optional[str] = None

NONE_LABEL = "None of the above"
NONE_ALIASES = ["None of the above", "none of the above", "none", "no fallacy", "nofallacy"]

LLM_NONE = "None"

METADATA_COLUMNS = {
    "id", "component_id", "year", "sentence", "speech_turn", "speaker",
    "relations", "relations_type", "relations_text", "relations_label",
    "fallacy_text", "fallacy_labels", "components_labels", "component_labels",
    "component_label", "rag_context", "retrieval_query", "has_relations", "text",
}


def _normalize(text: str) -> str:
    """Lowercase and strip every non-alphanumeric character (spaces, hyphens, plurals' punctuation)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


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
    A present ``--label_column_name`` gives single-label; two or more binary columns give multi-label."""
    columns = raw_datasets[ref_split].column_names
    if data_args.label_column_name in columns:
        return False, [data_args.label_column_name]
    candidates = [
        c for c in columns
        if c not in METADATA_COLUMNS and _looks_binary(raw_datasets[ref_split].unique(c))
    ]
    if len(candidates) >= 2:
        return True, candidates
    return False, [data_args.label_column_name]


def apply_none_rules(labels: list[str]) -> list[str]:
    """Multi-label 'None of the above' handling: it is exclusive, and an empty set defaults to it.
    No-op for single-label datasets, whose label space does not contain it."""
    if not IS_MULTI_LABEL:
        return labels
    if NONE_LABEL in labels:
        return [NONE_LABEL]
    if not labels:
        return [NONE_LABEL]
    return labels


def _apply_label_config(train_file) -> None:
    """Populate the dataset-aware label globals from the training file path."""
    global ALL_LABELS, LABEL_TO_LLM, ALIAS_TO_LABEL, _LABEL_MENU, SINGLE_LABEL_NONE
    cfg = get_label_config(train_file)
    ALL_LABELS = cfg["labels"]
    LABEL_TO_LLM = cfg["label_to_llm"]
    ALIAS_TO_LABEL = cfg["alias_to_label"]
    _LABEL_MENU = cfg["label_menu"]
    SINGLE_LABEL_NONE = cfg.get("none_label")


def _register_none_class() -> None:
    """Register the explicit 'None of the above' class in the label globals (multi-label only),
    since ``get_label_config`` lists only the positive fallacy classes."""
    global ALL_LABELS
    if NONE_LABEL not in ALL_LABELS:
        ALL_LABELS = ALL_LABELS + [NONE_LABEL]
    LABEL_TO_LLM[NONE_LABEL] = NONE_LABEL
    for alias in NONE_ALIASES:
        norm = _normalize(alias)
        if norm:
            ALIAS_TO_LABEL[norm] = NONE_LABEL


SYSTEM_PROMPT = None


@dataclass
class ModelArguments:
    """Arguments for the LLM backbone."""

    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "HuggingFace model identifier or local path (local backend), "
                          "or OpenAI model name (openai backend)."},
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
        metadata={"help": "HuggingFace auth token for private / gated models."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow execution of remote code from Hub repos."},
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
        metadata={"help": "Training data file (CSV or JSON)."},
    )
    validation_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/classification/dev.csv",
        metadata={"help": "Validation data file (CSV or JSON)."},
    )
    test_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/classification/test.csv",
        metadata={"help": "Test / prediction data file (CSV or JSON)."},
    )
    text_column_name: Optional[str] = field(
        default="sentence",
        metadata={"help": "Column containing the input sentence."},
    )
    speech_turn_column_name: Optional[str] = field(
        default="speech_turn",
        metadata={"help": "Column containing the full speech turn."},
    )
    context_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "Column containing the RAG context (used when --use_rag)."},
    )
    label_column_name: Optional[str] = field(
        default="fallacy_label",
        metadata={"help": "Column containing the gold fallacy classification label. "
                          "Parsing tolerates comma / semicolon / pipe separated cells, "
                          "but the task itself is single-label: exactly one fallacy "
                          "type per example."},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite cached datasets."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training set (useful for debugging)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate evaluation set."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate prediction set."},
    )
    shuffle_train_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Random seed for shuffling."},
    )
    use_rag: bool = field(
        default=False,
        metadata={"help": "Append the pre-built 'context' field to each prompt."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Provide either --dataset_name or --train_file / --validation_file.")
        for attr in ("train_file", "validation_file"):
            path = getattr(self, attr)
            if path is not None and path.rsplit(".", 1)[-1] not in ("csv", "json"):
                raise ValueError(f"{attr} must be a .csv or .json file.")
        if self.use_rag:
            for attr in ("train_file", "validation_file", "test_file"):
                path = getattr(self, attr)
                if path is not None and "/no_rag/" in path:
                    setattr(self, attr, path.replace("/no_rag/", "/rag/"))


@dataclass
class LLMExperimentArguments:
    """Arguments specific to the LLM experiment (mode, backend, LoRA, etc.)."""

    mode: str = field(
        default="zero_shot",
        metadata={"help": "Experiment mode: zero_shot | few_shot | fine_tune."},
    )
    backend: str = field(
        default="local",
        metadata={"help": "Inference backend: local (HuggingFace) | openai."},
    )
    num_shots: int = field(
        default=1,
        metadata={"help": "Number of labelled examples per fallacy class for few-shot prompting "
                          "(total prompt examples ≈ num_shots * number_of_classes). "
                          "Fewer shots means shorter prompts and faster inference; raise it "
                          "if accuracy matters more than speed."},
    )
    few_shot_seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting few-shot examples."},
    )
    max_new_tokens: int = field(
        default=32,
        metadata={"help": "Maximum number of new tokens to generate per example "
                          "(needs to fit a single fallacy type name)."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature (0 = greedy)."},
    )
    inference_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for local inference (not used for openai backend)."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit NF4 (QLoRA). Requires bitsandbytes."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit. Requires bitsandbytes."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA scaling factor (alpha)."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules for LoRA. "
                          "If None, PEFT auto-detects suitable modules."},
    )
    merge_adapter: bool = field(
        default=False,
        metadata={"help": "After fine-tuning, merge LoRA weights into the base model "
                          "and save the full model to --output_dir/merged."},
    )

    def __post_init__(self):
        valid_modes = {"zero_shot", "few_shot", "fine_tune"}
        if self.mode not in valid_modes:
            raise ValueError(f"--mode must be one of {valid_modes}, got '{self.mode}'.")
        valid_backends = {"local", "openai"}
        if self.backend not in valid_backends:
            raise ValueError(f"--backend must be one of {valid_backends}, got '{self.backend}'.")
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot set both --load_in_4bit and --load_in_8bit.")


_GOLD_SPLIT_RE = re.compile(r"[,;|/]+")


def parse_gold_labels(raw: object) -> list[str]:
    """Parse a raw gold-label cell into a sorted list of canonical labels.
    The task is single-label, so this normally returns a single-item list."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or _normalize(text) in {"", "nan"}:
        return []
    labels = set()
    for token in _GOLD_SPLIT_RE.split(text):
        token = token.strip()
        if not token:
            continue
        canon = ALIAS_TO_LABEL.get(_normalize(token))
        if canon:
            labels.add(canon)
        else:
            logger.warning(f"Unrecognised gold label token: {token!r} (in {text!r}); skipping.")
    return sorted(labels, key=ALL_LABELS.index)


def parse_llm_response(response: str) -> list[str]:
    """Extract the canonical fallacy label(s) mentioned in a raw LLM response.
    Single-label keeps the earliest alias match; multi-label keeps every match in canonical order."""
    norm = _normalize(response)
    if not norm:
        return []
    matches = []
    for alias_norm, canon in ALIAS_TO_LABEL.items():
        if not alias_norm:
            continue
        pos = norm.find(alias_norm)
        if pos != -1:
            matches.append((pos, -len(alias_norm), canon))
    if not matches:
        return []
    matches.sort()
    if not IS_MULTI_LABEL:
        return [matches[0][2]]
    distinct = {canon for _, _, canon in matches}
    return sorted(distinct, key=ALL_LABELS.index)


def labels_to_llm_target(labels: list[str]) -> str:
    """Render canonical label(s) as the target string for the assistant turn.
    Multi-label joins them with ", "; single-label raises on an empty set."""
    if IS_MULTI_LABEL:
        canon = apply_none_rules(sorted(set(labels), key=ALL_LABELS.index))
        return ", ".join(LABEL_TO_LLM[c] for c in canon)
    if not labels:
        if SINGLE_LABEL_NONE:
            return LABEL_TO_LLM[SINGLE_LABEL_NONE]
        raise ValueError(
            "Expected exactly one fallacy label; empty label sets are not valid "
            "under the single-label task (\"None\" is not permitted)."
        )
    canon = sorted(set(labels), key=ALL_LABELS.index)[0]
    return LABEL_TO_LLM[canon]


def build_user_message(
        sentence: str,
        speech_turn: str,
        context: Optional[str] = None,
) -> str:
    """Build the user-turn text for a single example."""
    parts = [
        f'Sentence: "{sentence}"',
        f"Debate turn context:\n{speech_turn}",
    ]
    if context:
        parts.append(f"Retrieved context:\n{context}")
    if IS_MULTI_LABEL:
        parts.append(
            "Which fallacy type(s) does this sentence contain? "
            f"Answer with every type that applies, drawn only from: {_LABEL_MENU}, "
            "separated by commas. "
            f"If the sentence contains no fallacy, answer exactly \"{NONE_LABEL}\". "
            "Do not invent new types."
        )
    elif SINGLE_LABEL_NONE:
        none_readable = LABEL_TO_LLM.get(SINGLE_LABEL_NONE, "No Fallacy")
        parts.append(
            "Which fallacy type does this sentence contain, if any? "
            f"Answer with exactly one option drawn only from: {_LABEL_MENU}. "
            f"If the sentence contains no fallacy, answer exactly \"{none_readable}\". "
            "Do not list more than one option."
        )
    else:
        parts.append(
            "Which single fallacy type does this sentence contain? "
            f"Answer with exactly one type drawn only from: {_LABEL_MENU}. "
            "Every sentence contains exactly one of these; do not answer \"None\" and "
            "do not list more than one type."
        )
    return "\n\n".join(parts)


def build_zero_shot_messages(
        sentence: str,
        speech_turn: str,
        context: Optional[str] = None,
) -> list[dict]:
    """Chat messages list for zero-shot inference."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentence, speech_turn, context)},
    ]


def build_few_shot_messages(
        sentence: str,
        speech_turn: str,
        examples: list[dict],
        context: Optional[str] = None,
) -> list[dict]:
    """Chat messages list with few-shot examples interleaved as turns."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": build_user_message(
            ex["sentence"], ex["speech_turn"], ex.get("context")
        )})
        messages.append({"role": "assistant", "content": ex["llm_response"]})
    messages.append({"role": "user", "content": build_user_message(sentence, speech_turn, context)})
    return messages


def format_for_sft(
        example: dict,
        tokenizer,
        text_col: str,
        speech_turn_col: str,
        context_col: Optional[str],
        use_rag: bool,
) -> dict:
    """Format one training example as a complete chat string for SFTTrainer ({"text": ...})."""
    sentence = example[text_col] or ""
    speech_turn = example[speech_turn_col] or ""
    context = example[context_col] if (use_rag and context_col and context_col in example) else None
    llm_target = labels_to_llm_target(parse_gold_labels(example.get("label")))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentence, speech_turn, context)},
        {"role": "assistant", "content": llm_target},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def sample_few_shot_examples(
        dataset,
        n_per_class: int,
        text_col: str,
        speech_turn_col: str,
        context_col: Optional[str],
        use_rag: bool,
        seed: int = 42,
) -> list[dict]:
    """Sample up to ``n_per_class`` examples per fallacy label and return a flat shuffled list.
    De-duplicates examples chosen for several classes."""
    rng = random.Random(seed)
    examples = []
    seen = set()
    for label in ALL_LABELS:
        subset = [
            (i, ex) for i, ex in enumerate(dataset)
            if label in parse_gold_labels(ex.get("label"))
        ]
        chosen = rng.sample(subset, min(n_per_class, len(subset)))
        for idx, ex in chosen:
            if idx in seen:
                continue
            seen.add(idx)
            gold = parse_gold_labels(ex.get("label"))
            record = {
                "sentence": ex[text_col] or "",
                "speech_turn": ex[speech_turn_col] or "",
                "labels": gold,
                "llm_response": labels_to_llm_target(gold),
            }
            if use_rag and context_col and context_col in ex:
                record["context"] = ex[context_col]
            examples.append(record)
    rng.shuffle(examples)
    return examples


def compute_metrics(predictions: list[list[str]], references: list[list[str]]) -> dict:
    """Compute classification metrics over a fixed indicator space built from ALL_LABELS.
    Reports exact-match accuracy, Hamming loss, micro/macro/weighted/samples F1, precision and recall."""
    mlb = MultiLabelBinarizer(classes=ALL_LABELS)
    y_true = mlb.fit_transform(references)
    y_pred = mlb.transform(predictions)

    metrics = {
        "subset_accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        "hamming_loss": hamming_loss(y_true=y_true, y_pred=y_pred),
        "f1_micro": f1_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0),
        "f1_samples": f1_score(y_true=y_true, y_pred=y_pred, average="samples", zero_division=0),
        "precision_micro": precision_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
    }
    per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
    for label, score in zip(ALL_LABELS, per_class):
        metrics[f"f1_{label}"] = float(score)
    return metrics


def _is_prequantized(model_name_or_path: str, **config_kwargs) -> bool:
    """Return True if the checkpoint already ships its own quantization_config (e.g. Mxfp4)."""
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
        return getattr(cfg, "quantization_config", None) is not None
    except Exception:
        return False


def _build_load_kwargs(
        model_name_or_path: str,
        llm_args: "LLMExperimentArguments",
        base_kwargs: dict,
        torch_module,
) -> dict:
    """Populate quantization/dtype load kwargs, deferring to a checkpoint's built-in quant config."""
    kwargs = dict(base_kwargs)

    config_keys = {k: base_kwargs[k] for k in ("cache_dir", "token", "trust_remote_code")
                   if k in base_kwargs}
    if _is_prequantized(model_name_or_path, **config_keys):
        if llm_args.load_in_4bit or llm_args.load_in_8bit:
            logger.warning(
                f"Model '{model_name_or_path}' already has a built-in quantization config. "
                "Ignoring --load_in_4bit / --load_in_8bit to avoid conflicting quant configs."
            )
        kwargs["torch_dtype"] = "auto"
        return kwargs

    try:
        from transformers import BitsAndBytesConfig
        _bnb_available = True
    except ImportError:
        _bnb_available = False

    if llm_args.load_in_4bit and _bnb_available:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_module.bfloat16,
        )
    elif llm_args.load_in_8bit and _bnb_available:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch_module.bfloat16

    return kwargs


def _generate_batched(
        model,
        tokenizer,
        messages_list,
        batch_size,
        max_new_tokens,
        temperature,
        torch_module,
        max_length: int = 2048,
) -> list[str]:
    """Length-sorted, left-padded batched generation: prompts are sorted short to long so each batch
    wastes little compute on padding, then results are re-ordered back to the input order."""
    encodings = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        encodings.append(tokenizer(text, truncation=True, max_length=max_length))

    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]))

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False

    responses: list[Optional[str]] = [None] * len(encodings)
    n_batches = (len(order) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(order), batch_size), total=n_batches,
                  desc="Generating", unit="batch"):
        idxs = order[i: i + batch_size]
        inputs = tokenizer.pad(
            [encodings[j] for j in idxs],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch_module.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        for k, out in enumerate(output_ids):
            decoded = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            responses[idxs[k]] = decoded

    return responses


class LocalLLMInference:
    """Wrap a HuggingFace causal LM for batched, chat-templated inference."""

    def __init__(
            self,
            model_name_or_path: str,
            llm_args: LLMExperimentArguments,
            model_args: ModelArguments,
    ):
        """Load the tokenizer and model with the requested quantization settings."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading tokenizer from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_kwargs = dict(
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        load_kwargs = _build_load_kwargs(model_name_or_path, llm_args, base_kwargs, torch)

        is_quantized = "quantization_config" in load_kwargs
        if torch.cuda.is_available():
            load_kwargs.setdefault("device_map", "auto")

        logger.info(f"Loading model from {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

        if not is_quantized and "device_map" not in load_kwargs:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            else:
                logger.warning(
                    "No CUDA/MPS device detected — running on CPU. Generation with a large "
                    "model will be slow; use a GPU, --load_in_4bit, a smaller model, or the "
                    "OpenAI backend."
                )
        self.model.eval()
        logger.info(f"Model device: {next(self.model.parameters()).device}")

        self.max_new_tokens = llm_args.max_new_tokens
        self.temperature = llm_args.temperature
        self.batch_size = llm_args.inference_batch_size

    def predict_batch(self, messages_list: list[list[dict]]) -> list[str]:
        """Run length-sorted batched inference over chat-message lists."""
        import torch

        return _generate_batched(
            self.model,
            self.tokenizer,
            messages_list,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            torch_module=torch,
        )


class OpenAIInference:
    """Wraps the OpenAI chat completions API for inference."""

    def __init__(self, model_name: str, llm_args: LLMExperimentArguments):
        """Initialize the OpenAI client from the OPENAI_API_KEY environment variable."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai>=1.0: pip install openai") from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_new_tokens = llm_args.max_new_tokens
        self.temperature = llm_args.temperature

    def predict_batch(self, messages_list: list[list[dict]]) -> list[str]:
        """Run chat completion on each message list and return the raw text responses."""
        responses = []
        for messages in tqdm(messages_list, desc="Generating", unit="ex"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            responses.append(response.choices[0].message.content or "")
        return responses


def run_inference(
        split_name: str,
        dataset,
        inference_engine,
        data_args: DataTrainingArguments,
        llm_args: LLMExperimentArguments,
        few_shot_examples: Optional[list[dict]] = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Build prompts, run inference, and return (predictions, references) as label-set lists.
    ``references`` is empty when the split has no gold labels."""
    text_col = data_args.text_column_name
    speech_col = data_args.speech_turn_column_name
    ctx_col = data_args.context_column_name

    messages_list = []
    for ex in dataset:
        sentence = ex[text_col] or ""
        speech_turn = ex[speech_col] or ""
        context = ex[ctx_col] if (data_args.use_rag and ctx_col and ctx_col in ex) else None

        if llm_args.mode == "few_shot" and few_shot_examples:
            msgs = build_few_shot_messages(sentence, speech_turn, few_shot_examples, context)
        else:
            msgs = build_zero_shot_messages(sentence, speech_turn, context)
        messages_list.append(msgs)

    logger.info(f"Running inference on {len(messages_list)} {split_name} examples …")
    raw_responses = inference_engine.predict_batch(messages_list)

    predictions = [apply_none_rules(parse_llm_response(r)) for r in raw_responses]

    references: list[list[str]] = []
    if "label" in dataset.features:
        references = [parse_gold_labels(ex["label"]) for ex in dataset]

    return predictions, references


def run_fine_tuning(
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        llm_args: LLMExperimentArguments,
        training_args: TrainingArguments,
        train_dataset,
        eval_dataset,
) -> tuple:
    """Fine-tune with LoRA/QLoRA via TRL SFTTrainer; return (model, tokenizer) ready for inference."""
    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info("=== Fine-tuning phase ===")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    load_kwargs = _build_load_kwargs(model_args.model_name_or_path, llm_args, base_kwargs, torch)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **load_kwargs,
    )

    target_modules = (
        [m.strip() for m in llm_args.lora_target_modules.split(",")]
        if llm_args.lora_target_modules
        else "all-linear"
    )
    lora_config = LoraConfig(
        r=llm_args.lora_r,
        lora_alpha=llm_args.lora_alpha,
        lora_dropout=llm_args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    def _fmt(ex):
        """Format one example as an SFT chat string."""
        return format_for_sft(
            ex,
            tokenizer=tokenizer,
            text_col=data_args.text_column_name,
            speech_turn_col=data_args.speech_turn_column_name,
            context_col=data_args.context_column_name,
            use_rag=data_args.use_rag,
        )

    train_sft = train_dataset.map(_fmt, desc="Formatting train for SFT")
    eval_sft = eval_dataset.map(_fmt, desc="Formatting eval for SFT") if eval_dataset else None

    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        eval_strategy=training_args.eval_strategy if eval_sft else "no",
        load_best_model_at_end=training_args.load_best_model_at_end if eval_sft else False,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        seed=training_args.seed,
        report_to=training_args.report_to,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_sft,
        eval_dataset=eval_sft,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training …")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()

    if llm_args.merge_adapter:
        logger.info("Merging LoRA adapter into base model …")
        merged_model = trainer.model.merge_and_unload()
        merged_path = os.path.join(training_args.output_dir, "merged")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        logger.info(f"Merged model saved to {merged_path}")
        return merged_model, tokenizer

    return trainer.model, tokenizer


def load_raw_datasets(data_args: DataTrainingArguments, model_args: ModelArguments):
    """Load the train/validation/test CSV or JSON splits into a DatasetDict."""
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file,
    }
    extension = (data_args.train_file or data_args.validation_file).rsplit(".", 1)[-1]
    return load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)


def _build_gold_label_column(ds, label_columns: list[str]):
    """Add a ';'-joined 'label' string built from the 0/1 indicator columns (multi-label)."""

    def _mk(ex):
        names = [c for c in label_columns if int(ex[c]) == 1]
        return {"label": ";".join(names)}

    return ds.map(_mk, desc="Building multi-label gold labels")


def prepare_split(raw_datasets, split_name, label_column, max_samples=None,
                  multilabel_columns: Optional[list[str]] = None):
    """Materialise a 'label' column, optionally truncate, and return a plain HF Dataset.
    Multi-label builds it from the 0/1 indicators; single-label renames the categorical column."""
    ds = raw_datasets[split_name]
    if multilabel_columns:
        ds = _build_gold_label_column(ds, multilabel_columns)
    elif label_column != "label" and label_column in ds.column_names:
        ds = ds.rename_column(label_column, "label")
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def main():
    """Run the chosen mode (zero-shot, few-shot, or fine-tune) and evaluate/predict."""
    global SYSTEM_PROMPT, IS_MULTI_LABEL

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LLMExperimentArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, llm_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, llm_args, training_args = parser.parse_args_into_dataclasses()

    SYSTEM_PROMPT = get_classification_prompt(data_args.train_file)
    _apply_label_config(data_args.train_file)

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)

    logger.info(
        f"Mode: {llm_args.mode} | Backend: {llm_args.backend} | "
        f"Model: {model_args.model_name_or_path} | RAG: {data_args.use_rag}"
    )
    set_seed(training_args.seed)

    raw_datasets = load_raw_datasets(data_args, model_args)

    ref_split = "train" if "train" in raw_datasets else next(iter(raw_datasets))
    IS_MULTI_LABEL, label_columns = detect_label_columns(raw_datasets, data_args, ref_split)
    multilabel_columns = None
    if IS_MULTI_LABEL:
        _register_none_class()
        multilabel_columns = label_columns
        logger.info(f"Detected MULTI-LABEL classification over columns: {label_columns}")
    else:
        logger.info("Detected SINGLE-LABEL classification")
    logger.info(f"Fallacy label space ({len(ALL_LABELS)}): {ALL_LABELS}")

    label_column = data_args.label_column_name
    train_dataset, eval_dataset, predict_dataset = None, None, None

    if training_args.do_train or llm_args.mode in ("few_shot",):
        train_dataset = prepare_split(
            raw_datasets, "train", label_column, data_args.max_train_samples,
            multilabel_columns=multilabel_columns,
        )
        if data_args.shuffle_train_dataset:
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        logger.info(f"Train split: {len(train_dataset)} examples")

    if training_args.do_eval:
        eval_dataset = prepare_split(
            raw_datasets, "validation", label_column, data_args.max_eval_samples,
            multilabel_columns=multilabel_columns,
        )
        logger.info(f"Validation split: {len(eval_dataset)} examples")

    if training_args.do_predict:
        predict_dataset = prepare_split(
            raw_datasets, "test", label_column, data_args.max_predict_samples,
            multilabel_columns=multilabel_columns,
        )
        logger.info(f"Test split: {len(predict_dataset)} examples")

    few_shot_examples = None
    if llm_args.mode == "few_shot":
        if train_dataset is None:
            raise ValueError("few_shot mode requires a training split for example selection.")
        few_shot_examples = sample_few_shot_examples(
            dataset=train_dataset,
            n_per_class=llm_args.num_shots,
            text_col=data_args.text_column_name,
            speech_turn_col=data_args.speech_turn_column_name,
            context_col=data_args.context_column_name,
            use_rag=data_args.use_rag,
            seed=llm_args.few_shot_seed,
        )
        logger.info(
            f"Few-shot pool: {len(few_shot_examples)} examples "
            f"(up to {llm_args.num_shots} per class)"
        )

    inference_model = None
    inference_tokenizer = None

    if llm_args.mode == "fine_tune":
        if llm_args.backend != "local":
            raise ValueError("fine_tune mode is only supported with --backend local.")
        if train_dataset is None:
            raise ValueError("fine_tune mode requires --do_train and a training split.")

        inference_model, inference_tokenizer = run_fine_tuning(
            model_args=model_args,
            data_args=data_args,
            llm_args=llm_args,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    engine = None
    if training_args.do_eval or training_args.do_predict:
        if llm_args.backend == "openai":
            engine = OpenAIInference(
                model_name=model_args.model_name_or_path,
                llm_args=llm_args,
            )
        else:
            if llm_args.mode == "fine_tune" and inference_model is not None:
                engine = _FineTunedLocalEngine(
                    model=inference_model,
                    tokenizer=inference_tokenizer,
                    llm_args=llm_args,
                )
            else:
                engine = LocalLLMInference(
                    model_name_or_path=model_args.model_name_or_path,
                    llm_args=llm_args,
                    model_args=model_args,
                )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        predictions, references = run_inference(
            split_name="validation",
            dataset=eval_dataset,
            inference_engine=engine,
            data_args=data_args,
            llm_args=llm_args,
            few_shot_examples=few_shot_examples,
        )
        if references:
            metrics = compute_metrics(predictions, references)
            metrics["eval_samples"] = len(eval_dataset)
            logger.info(f"Eval metrics: {metrics}")
            _save_metrics(metrics, os.path.join(training_args.output_dir, "eval_results.json"))

    if training_args.do_predict and predict_dataset is not None:
        logger.info("*** Predict ***")
        predictions, references = run_inference(
            split_name="test",
            dataset=predict_dataset,
            inference_engine=engine,
            data_args=data_args,
            llm_args=llm_args,
            few_shot_examples=few_shot_examples,
        )
        if references:
            metrics = compute_metrics(predictions, references)
            metrics["predict_samples"] = len(predict_dataset)
            logger.info(f"Predict metrics: {metrics}")
            _save_metrics(metrics, os.path.join(training_args.output_dir, "predict_results_metrics.json"))

        output_path = os.path.join(training_args.output_dir, "predict_results.txt")
        with open(output_path, "w") as fh:
            fh.write("index\tprediction\tgold\n")
            for idx, pred in enumerate(predictions):
                pred_str = ",".join(pred) if pred else LLM_NONE
                gold_str = (",".join(references[idx]) if references[idx] else LLM_NONE) if references else "N/A"
                fh.write(f"{idx}\t{pred_str}\t{gold_str}\n")
        logger.info(f"Predictions saved to {output_path}")


class _FineTunedLocalEngine:
    """Adapt an in-memory fine-tuned model to the LocalLLMInference predict_batch() interface."""

    def __init__(self, model, tokenizer, llm_args: LLMExperimentArguments):
        """Store the fine-tuned model/tokenizer and switch to left-padding for inference."""
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.max_new_tokens = llm_args.max_new_tokens
        self.temperature = llm_args.temperature
        self.batch_size = llm_args.inference_batch_size
        self._torch = torch

    def predict_batch(self, messages_list: list[list[dict]]) -> list[str]:
        """Run length-sorted batched generation over chat-message lists."""
        return _generate_batched(
            self.model,
            self.tokenizer,
            messages_list,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            torch_module=self._torch,
        )


def _save_metrics(metrics: dict, path: str):
    """Write a metrics dict to *path* as indented JSON."""
    import json
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved to {path}")


if __name__ == "__main__":
    main()
