"""
LLM-based relation classification: zero-shot, few-shot, and fine-tuning.

Task
----
Given a pair of sentences (sentence1, sentence2), assign the single argumentative
relation that holds between them. The task is single-label: exactly one relation
per pair. Two datasets are supported, each with its own 3-class label space:

  - LIARArg      : support | attack | no_relation
  - DEBAGREEMENT : support | attack | neutral

The dataset (and thus prompt wording and label space) is auto-detected from the
training file path.

Architecture
------------
Each pair is formatted as an instruction-following chat prompt:

    [System]   You are an expert in argumentation analysis …
    [User]     Sentence 1: <sentence1>
               Sentence 2: <sentence2>
               What relation does Sentence 2 hold toward Sentence 1?
               Answer with exactly one of: Support, Attack, No Relation.
    [Assistant] Attack

Post-processing
---------------
The raw LLM response is parsed back into a canonical dataset label via fuzzy
alias matching (e.g. "attacks" -> "attack", "no relation" -> "no_relation").
The earliest-occurring match is kept.

Modes (--mode)
--------------
  zero_shot  : direct prompt inference, no labelled examples
  few_shot   : --num_shots balanced examples per class prepended to the prompt
  fine_tune  : LoRA/QLoRA supervised fine-tuning via TRL SFTTrainer,
               followed by evaluation / prediction

Backend (--backend)
-------------------
  local   : HuggingFace generate() — works with any causal LM
  openai  : OpenAI chat completions (requires OPENAI_API_KEY env var)
"""

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
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Relation label configuration (dataset-aware) ────────────────────────────────
# Canonical labels match the CSV "label_value" column; the LLM emits the readable name.

_LABELS_BY_DATASET: dict[str, list[str]] = {
    "liararg": ["support", "attack", "no_relation"],
    "debagreement": ["support", "attack", "neutral"],
}

_LABEL_TO_LLM_BY_DATASET: dict[str, dict[str, str]] = {
    "liararg": {"support": "Support", "attack": "Attack", "no_relation": "No Relation"},
    "debagreement": {"support": "Support", "attack": "Attack", "neutral": "Neutral"},
}

# Extra alias variants a model tends to emit, mapped to the canonical label.
_EXTRA_ALIASES_BY_DATASET: dict[str, dict[str, list[str]]] = {
    "liararg": {
        "support": ["supports", "supporting", "agree", "agreement", "in favor", "pro"],
        "attack": ["attacks", "attacking", "disagree", "disagreement", "against",
                   "rebut", "counter", "con"],
        "no_relation": ["no relation", "unrelated", "none", "neither", "no-relation"],
    },
    "debagreement": {
        "support": ["supports", "supporting", "agree", "agreement", "in favor", "pro"],
        "attack": ["attacks", "attacking", "disagree", "disagreement", "against",
                   "rebut", "counter", "con"],
        "neutral": ["no stance", "undecided", "neither", "unrelated", "none"],
    },
}

# Dataset-aware system prompts.
_SYSTEM_PROMPTS: dict[str, str] = {
    "liararg": (
        "You are an expert in argumentation analysis. You are given two sentences "
        "drawn from a fact-checking argumentation corpus: a claim (Sentence 1) and a "
        "candidate piece of evidence (Sentence 2). "
        "Your task is to identify the argumentative relation that Sentence 2 holds "
        "toward Sentence 1. Consider only these three relations:\n"
        "- Support: Sentence 2 provides evidence or reasoning in favour of Sentence 1.\n"
        "- Attack: Sentence 2 provides evidence or reasoning against Sentence 1.\n"
        "- No Relation: Sentence 2 is not argumentatively related to Sentence 1.\n"
        "Answer with exactly one of those three relation names, and nothing else."
    ),
    "debagreement": (
        "You are an expert in argumentation analysis. You are given two posts from an "
        "online debate (Reddit): a parent comment (Sentence 1) and a reply (Sentence 2). "
        "Your task is to identify the (dis)agreement relation that the reply holds "
        "toward the parent comment. Consider only these three relations:\n"
        "- Support: the reply agrees with or supports the parent comment.\n"
        "- Attack: the reply disagrees with or argues against the parent comment.\n"
        "- Neutral: the reply neither clearly agrees nor disagrees with the parent comment.\n"
        "Answer with exactly one of those three relation names, and nothing else."
    ),
}

# Populated at runtime in main() via _apply_relation_config().
RELATION_LABELS: list[str] = []
LABEL_TO_LLM: dict[str, str] = {}
ALIAS_TO_LABEL: dict[str, str] = {}
LABEL_MENU: str = ""
SYSTEM_PROMPT: Optional[str] = None


def _normalize(text: str) -> str:
    """Lowercase and strip every non-alphanumeric character for alias matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def identify_dataset(train_file: Optional[str]) -> str:
    """Identify the dataset ('liararg' | 'debagreement') from the training file path."""
    if train_file is None:
        return "liararg"
    path = str(train_file).lower()
    if "liararg" in path or "liar" in path:
        return "liararg"
    if "debag" in path:
        return "debagreement"
    logger.warning(f"Could not identify dataset from '{train_file}'; defaulting to LIARArg.")
    return "liararg"


def _apply_relation_config(train_file: Optional[str]) -> None:
    """Populate the dataset-aware relation-label and prompt globals from the train path."""
    global RELATION_LABELS, LABEL_TO_LLM, ALIAS_TO_LABEL, LABEL_MENU, SYSTEM_PROMPT
    dataset = identify_dataset(train_file)
    RELATION_LABELS = list(_LABELS_BY_DATASET[dataset])
    LABEL_TO_LLM = dict(_LABEL_TO_LLM_BY_DATASET[dataset])
    SYSTEM_PROMPT = _SYSTEM_PROMPTS[dataset]

    alias_to_label: dict[str, str] = {}

    def _add(canon: str, *variants: str):
        for v in variants:
            n = _normalize(v)
            if n:
                alias_to_label[n] = canon

    # Register the canonical label and its readable name, then hand-written variants.
    for canon in RELATION_LABELS:
        _add(canon, canon, LABEL_TO_LLM[canon])
    for canon, variants in _EXTRA_ALIASES_BY_DATASET[dataset].items():
        _add(canon, *variants)

    ALIAS_TO_LABEL = alias_to_label
    LABEL_MENU = ", ".join(LABEL_TO_LLM[lab] for lab in RELATION_LABELS)
    logger.info(f"Dataset '{dataset}' | relations ({len(RELATION_LABELS)}): {RELATION_LABELS}")


# ── Dataclasses ────────────────────────────────────────────────────────────────

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
    """Arguments for data loading and preprocessing."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (via datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name for the dataset."},
    )
    train_file: Optional[str] = field(
        default="./datasets/LIARArg-dataset/relation-classification/train.csv",
        metadata={"help": "Training data file (CSV or JSON)."},
    )
    validation_file: Optional[str] = field(
        default="./datasets/LIARArg-dataset/relation-classification/dev.csv",
        metadata={"help": "Validation data file (CSV or JSON)."},
    )
    test_file: Optional[str] = field(
        default="./datasets/LIARArg-dataset/relation-classification/test.csv",
        metadata={"help": "Test / prediction data file (CSV or JSON)."},
    )
    text_column_name: Optional[str] = field(
        default="sentence1",
        metadata={"help": "Column containing the first sentence of the pair."},
    )
    speech_turn_column_name: Optional[str] = field(
        default="sentence2",
        metadata={"help": "Column containing the second sentence of the pair."},
    )
    label_column_name: Optional[str] = field(
        default="label_value",
        metadata={"help": "Column containing the gold relation label "
                          "(support / attack / no_relation / neutral)."},
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

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Provide either --dataset_name or --train_file / --validation_file.")
        for attr in ("train_file", "validation_file"):
            path = getattr(self, attr)
            if path is not None and path.rsplit(".", 1)[-1] not in ("csv", "json"):
                raise ValueError(f"{attr} must be a .csv or .json file.")


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
    # ── Few-shot ──────────────────────────────────────────────────────────────
    num_shots: int = field(
        default=1,
        metadata={"help": "Number of labelled examples per relation class for few-shot "
                          "prompting (total examples ≈ num_shots * number_of_classes)."},
    )
    few_shot_seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting few-shot examples."},
    )
    # ── Inference ─────────────────────────────────────────────────────────────
    max_new_tokens: int = field(
        default=16,
        metadata={"help": "Maximum number of new tokens to generate per example "
                          "(needs to fit a single relation name)."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature (0 = greedy)."},
    )
    inference_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for local inference (not used for openai backend)."},
    )
    # ── Quantization ──────────────────────────────────────────────────────────
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit NF4 (QLoRA). Requires bitsandbytes."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit. Requires bitsandbytes."},
    )
    # ── LoRA (fine_tune mode) ─────────────────────────────────────────────────
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


# ── Label (de)serialisation ────────────────────────────────────────────────────

def parse_gold_label(raw: object) -> Optional[str]:
    """Parse a raw gold-label cell into its canonical relation label (or None)."""
    if raw is None:
        return None
    canon = ALIAS_TO_LABEL.get(_normalize(str(raw)))
    if canon is None:
        logger.warning(f"Unrecognised gold label: {raw!r}; skipping.")
    return canon


def parse_llm_response(response: str) -> Optional[str]:
    """Extract the canonical relation mentioned in a raw LLM response.

    Returns the earliest-occurring alias match (ties broken toward the longer,
    more specific alias), or None if no known relation is found.
    """
    norm = _normalize(response)
    if not norm:
        return None
    matches = []
    for alias_norm, canon in ALIAS_TO_LABEL.items():
        pos = norm.find(alias_norm)
        if pos != -1:
            matches.append((pos, -len(alias_norm), canon))
    if not matches:
        return None
    matches.sort()
    return matches[0][2]


def label_to_llm_target(label: Optional[str]) -> str:
    """Render a canonical label as the readable target string for the assistant turn."""
    if label is None:
        raise ValueError("Expected exactly one relation label; got an empty/unknown label.")
    return LABEL_TO_LLM[label]


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def build_user_message(sentence1: str, sentence2: str) -> str:
    """Build the user-turn text for a single sentence pair."""
    parts = [
        f'Sentence 1: "{sentence1}"',
        f'Sentence 2: "{sentence2}"',
        "What single relation does Sentence 2 hold toward Sentence 1? "
        f"Answer with exactly one relation drawn only from: {LABEL_MENU}. "
        "Do not answer with more than one relation.",
    ]
    return "\n\n".join(parts)


def build_zero_shot_messages(sentence1: str, sentence2: str) -> list[dict]:
    """Chat messages list for zero-shot inference."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentence1, sentence2)},
    ]


def build_few_shot_messages(sentence1: str, sentence2: str, examples: list[dict]) -> list[dict]:
    """Chat messages list with few-shot examples interleaved as turns."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": build_user_message(
            ex["sentence1"], ex["sentence2"]
        )})
        messages.append({"role": "assistant", "content": ex["llm_response"]})
    messages.append({"role": "user", "content": build_user_message(sentence1, sentence2)})
    return messages


def format_for_sft(example: dict, tokenizer, text_col: str, speech_turn_col: str) -> dict:
    """Format one training example as a complete chat string for SFTTrainer ({"text": ...})."""
    sentence1 = example[text_col] or ""
    sentence2 = example[speech_turn_col] or ""
    # Translate gold label -> readable target so the model learns to emit the name.
    llm_target = label_to_llm_target(parse_gold_label(example.get("label")))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentence1, sentence2)},
        {"role": "assistant", "content": llm_target},
    ]
    # apply_chat_template adds the EOS token automatically.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ── Few-shot example selection ─────────────────────────────────────────────────

def sample_few_shot_examples(
        dataset,
        n_per_class: int,
        text_col: str,
        speech_turn_col: str,
        seed: int = 42,
) -> list[dict]:
    """Sample up to `n_per_class` examples per relation class; return a flat shuffled list."""
    rng = random.Random(seed)
    examples = []
    seen = set()
    for label in RELATION_LABELS:
        subset = [
            (i, ex) for i, ex in enumerate(dataset)
            if parse_gold_label(ex.get("label")) == label
        ]
        chosen = rng.sample(subset, min(n_per_class, len(subset)))
        for idx, ex in chosen:
            if idx in seen:
                continue
            seen.add(idx)
            gold = parse_gold_label(ex.get("label"))
            examples.append({
                "sentence1": ex[text_col] or "",
                "sentence2": ex[speech_turn_col] or "",
                "label": gold,
                # Pre-rendered target string used in the assistant turn of the prompt.
                "llm_response": label_to_llm_target(gold),
            })
    rng.shuffle(examples)
    return examples


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[Optional[str]], references: list[Optional[str]]) -> dict:
    """Compute multi-class single-label metrics over the fixed relation label space.

    Unparseable predictions are kept as-is (an out-of-space token) so they count as
    wrong. Reports accuracy, macro precision/recall/F1, and per-class F1.
    """
    y_pred = [p if p is not None else "unparsed" for p in predictions]
    y_true = [r if r is not None else "unparsed" for r in references]

    metrics = {
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        "precision_macro": precision_score(
            y_true=y_true, y_pred=y_pred, labels=RELATION_LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(
            y_true=y_true, y_pred=y_pred, labels=RELATION_LABELS, average="macro", zero_division=0),
        "f1_macro": f1_score(
            y_true=y_true, y_pred=y_pred, labels=RELATION_LABELS, average="macro", zero_division=0),
    }
    # Per-class F1 for error analysis.
    per_class = f1_score(
        y_true=y_true, y_pred=y_pred, labels=RELATION_LABELS, average=None, zero_division=0)
    for label, score in zip(RELATION_LABELS, per_class):
        metrics[f"f1_{label}"] = float(score)
    return metrics


# ── Model loading helpers ──────────────────────────────────────────────────────

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
        # Let Transformers handle dtype automatically for pre-quantized checkpoints.
        kwargs["torch_dtype"] = "auto"
        return kwargs

    # Standard path — apply BitsAndBytes if requested.
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


# ── Batched generation helper ──────────────────────────────────────────────────

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
    """Length-sorted, left-padded batched generation.

    Each chat prompt is tokenised once, examples are sorted short→long to minimise
    padding within batches, then results are re-ordered back to the original order.
    """
    # 1. Render + tokenise each chat prompt once (no padding yet).
    encodings = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        encodings.append(tokenizer(text, truncation=True, max_length=max_length))

    # 2. Process short prompts first so padding within each batch stays minimal.
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
        # tokenizer.pad re-pads the pre-tokenised examples (left padding set on the tokenizer).
        inputs = tokenizer.pad(
            [encodings[j] for j in idxs],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch_module.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens; map each result back to its original slot.
        prompt_len = inputs["input_ids"].shape[1]
        for k, out in enumerate(output_ids):
            decoded = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            responses[idxs[k]] = decoded

    return responses


# ── Local HuggingFace backend ──────────────────────────────────────────────────

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
            padding_side="left",  # causal LM: pad on the left
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


# ── OpenAI backend ─────────────────────────────────────────────────────────────

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


# ── Inference runner ───────────────────────────────────────────────────────────

def run_inference(
        split_name: str,
        dataset,
        inference_engine,
        data_args: DataTrainingArguments,
        llm_args: LLMExperimentArguments,
        few_shot_examples: Optional[list[dict]] = None,
) -> tuple[list[Optional[str]], list[Optional[str]]]:
    """Build prompts, run inference, and return (predictions, references) as label lists.

    Each prediction / reference is a canonical relation label (or None). `references`
    is empty when the split has no gold labels.
    """
    text_col = data_args.text_column_name
    speech_col = data_args.speech_turn_column_name

    messages_list = []
    for ex in dataset:
        sentence1 = ex[text_col] or ""
        sentence2 = ex[speech_col] or ""
        if llm_args.mode == "few_shot" and few_shot_examples:
            msgs = build_few_shot_messages(sentence1, sentence2, few_shot_examples)
        else:
            msgs = build_zero_shot_messages(sentence1, sentence2)
        messages_list.append(msgs)

    logger.info(f"Running inference on {len(messages_list)} {split_name} examples …")
    raw_responses = inference_engine.predict_batch(messages_list)

    predictions = [parse_llm_response(r) for r in raw_responses]
    references: list[Optional[str]] = []
    if "label" in dataset.features:
        references = [parse_gold_label(ex["label"]) for ex in dataset]

    return predictions, references


# ── Fine-tuning ────────────────────────────────────────────────────────────────

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

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",  # right-padding for SFT
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model ───────────────────────────────────────────────────────
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

    # ── LoRA config ───────────────────────────────────────────────────────────
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

    # ── Format datasets for SFT ───────────────────────────────────────────────
    def _fmt(ex):
        """Format one example as an SFT chat string."""
        return format_for_sft(
            ex,
            tokenizer=tokenizer,
            text_col=data_args.text_column_name,
            speech_turn_col=data_args.speech_turn_column_name,
        )

    train_sft = train_dataset.map(_fmt, desc="Formatting train for SFT")
    eval_sft = eval_dataset.map(_fmt, desc="Formatting eval for SFT") if eval_dataset else None

    # ── SFT config ────────────────────────────────────────────────────────────
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

    # Optionally merge adapter into base model.
    if llm_args.merge_adapter:
        logger.info("Merging LoRA adapter into base model …")
        merged_model = trainer.model.merge_and_unload()
        merged_path = os.path.join(training_args.output_dir, "merged")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        logger.info(f"Merged model saved to {merged_path}")
        return merged_model, tokenizer

    return trainer.model, tokenizer


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_raw_datasets(data_args: DataTrainingArguments, model_args: ModelArguments):
    """Load the train/validation/test CSV or JSON splits into a DatasetDict."""
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file,
    }
    extension = (data_args.train_file or data_args.validation_file).rsplit(".", 1)[-1]
    return load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)


def prepare_split(raw_datasets, split_name, label_column, max_samples=None):
    """Rename the gold-label column to 'label', optionally truncate, and return a Dataset."""
    ds = raw_datasets[split_name]
    if label_column != "label" and label_column in ds.column_names:
        # drop stale 'label' so the rename target is free
        if "label" in ds.column_names:
            ds = ds.remove_columns("label")
        ds = ds.rename_column(label_column, "label")
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Run the chosen mode (zero-shot, few-shot, or fine-tune) and evaluate/predict."""
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LLMExperimentArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, llm_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, llm_args, training_args = parser.parse_args_into_dataclasses()

    # ── Set the system prompt AND relation label space based on the dataset ─────
    _apply_relation_config(data_args.train_file)

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)

    logger.info(
        f"Mode: {llm_args.mode} | Backend: {llm_args.backend} | "
        f"Model: {model_args.model_name_or_path}"
    )
    set_seed(training_args.seed)

    # ── Load datasets ─────────────────────────────────────────────────────────
    raw_datasets = load_raw_datasets(data_args, model_args)
    label_column = data_args.label_column_name
    train_dataset, eval_dataset, predict_dataset = None, None, None

    if training_args.do_train or llm_args.mode in ("few_shot",):
        train_dataset = prepare_split(
            raw_datasets, "train", label_column, data_args.max_train_samples,
        )
        if data_args.shuffle_train_dataset:
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        logger.info(f"Train split: {len(train_dataset)} examples")

    if training_args.do_eval:
        eval_dataset = prepare_split(
            raw_datasets, "validation", label_column, data_args.max_eval_samples,
        )
        logger.info(f"Validation split: {len(eval_dataset)} examples")

    if training_args.do_predict:
        predict_dataset = prepare_split(
            raw_datasets, "test", label_column, data_args.max_predict_samples,
        )
        logger.info(f"Test split: {len(predict_dataset)} examples")

    # ── Few-shot example pool (sampled once from training set) ─────────────────
    few_shot_examples = None
    if llm_args.mode == "few_shot":
        if train_dataset is None:
            raise ValueError("few_shot mode requires a training split for example selection.")
        few_shot_examples = sample_few_shot_examples(
            dataset=train_dataset,
            n_per_class=llm_args.num_shots,
            text_col=data_args.text_column_name,
            speech_turn_col=data_args.speech_turn_column_name,
            seed=llm_args.few_shot_seed,
        )
        logger.info(
            f"Few-shot pool: {len(few_shot_examples)} examples "
            f"(up to {llm_args.num_shots} per class)"
        )

    # ── Fine-tuning ────────────────────────────────────────────────────────────
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

    # ── Build inference engine ─────────────────────────────────────────────────
    engine = None
    if training_args.do_eval or training_args.do_predict:
        if llm_args.backend == "openai":
            engine = OpenAIInference(
                model_name=model_args.model_name_or_path,
                llm_args=llm_args,
            )
        else:
            # For fine_tune, reuse the fine-tuned model/tokenizer.
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

    # ── Evaluation ─────────────────────────────────────────────────────────────
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

    # ── Prediction ─────────────────────────────────────────────────────────────
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
                pred_str = pred if pred else "unparsed"
                gold_str = (references[idx] or "unparsed") if references else "N/A"
                fh.write(f"{idx}\t{pred_str}\t{gold_str}\n")
        logger.info(f"Predictions saved to {output_path}")


# ── Thin wrapper to reuse fine-tuned model for inference ──────────────────────

class _FineTunedLocalEngine:
    """Adapt an in-memory fine-tuned model to the LocalLLMInference predict_batch() interface."""

    def __init__(self, model, tokenizer, llm_args: LLMExperimentArguments):
        """Store the fine-tuned model/tokenizer and switch to left-padding for inference."""
        import torch
        self.model = model
        self.tokenizer = tokenizer
        # Switch to left-padding for inference.
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


# ── Utility ────────────────────────────────────────────────────────────────────

def _save_metrics(metrics: dict, path: str):
    """Write a metrics dict to *path* as indented JSON."""
    import json
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved to {path}")


if __name__ == "__main__":
    main()
