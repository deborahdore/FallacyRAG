"""
LLM-based argument detection: zero-shot, few-shot, and fine-tuning.

Architecture
------------
Each example is formatted as an instruction-following chat prompt:

    [System]   You are an expert in argumentation analysis …
    [User]     Sentence: <sentence>
               Speech turn: <speech_turn>
               [Retrieved context: <context>]   (when --use_rag)
               Does this sentence contain a fallacy?
               Answer with exactly one word: "Yes" or "No".
    [Assistant] Yes | No

Post-processing
---------------
The raw LLM response ("Yes" / "No") is translated to dataset labels
before metrics and file output:
    "Yes"  → "Fallacy"
    "No"   → "NoFallacy"

Modes (--mode)
--------------
  zero_shot  : direct prompt inference, no labelled examples
  few_shot   : --num_shots balanced examples prepended to the prompt
  fine_tune  : LoRA/QLoRA supervised fine-tuning via TRL SFTTrainer,
               followed by evaluation / prediction

Backend (--backend)
-------------------
  local   : HuggingFace generate() — works with any causal LM
  openai  : OpenAI chat completions (requires OPENAI_API_KEY env var)

Retrieval-augmentation (--use_rag)
-----------------------------------
  Reads the pre-built "context" column from the dataset and appends it
  to each prompt, exactly mirroring the encoder counterpart.

Fine-tuning details
--------------------
  - LoRA / QLoRA (4-bit NF4) via peft + bitsandbytes
  - TRL SFTTrainer handles response-only loss masking automatically
  - Saves adapter weights to --output_dir; merges + saves full model
    if --merge_adapter is set
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import rootutils
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import HfArgumentParser, TrainingArguments, set_seed

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Label constants ────────────────────────────────────────────────────────────

# Dataset-level labels (used for metrics, output files, and gold references)
LABEL_FALLACY = "Fallacy"
LABEL_NO_FALLACY = "NoFallacy"
ALL_LABELS = [LABEL_FALLACY, LABEL_NO_FALLACY]

# LLM-level response tokens (what the model is asked to produce)
LLM_YES = "Yes"
LLM_NO = "No"

# Bidirectional mappings between LLM responses and dataset labels
LABEL_TO_LLM: dict[str, str] = {
    LABEL_FALLACY: LLM_YES,
    LABEL_NO_FALLACY: LLM_NO,
}
LLM_TO_LABEL: dict[str, str] = {
    LLM_YES: LABEL_FALLACY,
    LLM_NO: LABEL_NO_FALLACY,
}

SYSTEM_PROMPT = (
    "You are an expert in argumentation theory and political discourse analysis. "
    "Your task is to determine whether a given sentence from a political debate contains a fallacy. "
    "A fallacy is a deceptive argumentative move that appears valid but violates principles of correct reasoning. "
    'Answer with exactly one word: "Yes" if the sentence contains a fallacy, '
    'or "No" if it does not.'
)


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
        default="./dataset/detection/train.csv",
        metadata={"help": "Training data file (CSV or JSON)."},
    )
    validation_file: Optional[str] = field(
        default="./dataset/detection/dev.csv",
        metadata={"help": "Validation data file (CSV or JSON)."},
    )
    test_file: Optional[str] = field(
        default="./dataset/detection/test.csv",
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
        metadata={"help": "Column containing the binary classification label."},
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
    # ── Few-shot ──────────────────────────────────────────────────────────────
    num_shots: int = field(
        default=1,
        metadata={"help": "Number of labelled examples per class for few-shot prompting "
                          "(total prompt examples = 2 * num_shots for balanced selection). "
                          "Fewer shots means shorter prompts and faster inference; raise it "
                          "if accuracy matters more than speed."},
    )
    few_shot_seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting few-shot examples."},
    )
    # ── Inference ─────────────────────────────────────────────────────────────
    max_new_tokens: int = field(
        default=4,
        metadata={"help": "Maximum number of new tokens to generate per example. "
                          "The answer is a single word (\"Yes\"/\"No\"), so a small cap "
                          "avoids wasted decode steps; raise it only for verbose models."},
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


# ── Prompt helpers ─────────────────────────────────────────────────────────────

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
    parts.append(
        'Does this sentence contain a fallacy? '
        'Answer with exactly one word: "Yes" or "No".'
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
        # Few-shot demonstrations use the LLM response token (Yes/No), not the dataset label
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
    # Translate dataset label → LLM response token so the model learns to say "Yes"/"No"
    llm_target = LABEL_TO_LLM.get(str(example["label"]), LLM_NO)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentence, speech_turn, context)},
        {"role": "assistant", "content": llm_target},
    ]
    # apply_chat_template adds the EOS token automatically
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ── Output parsing ─────────────────────────────────────────────────────────────

def parse_llm_response(response: str, fallback: str = LLM_NO) -> str:
    """Extract the canonical "Yes"/"No" token from a raw LLM response, else return *fallback*."""
    response_clean = response.strip()
    lowered = response_clean.lower()
    # Check "yes" before "no" so a response containing both resolves to "Yes".
    if "yes" in lowered:
        return LLM_YES
    if "no" in lowered:
        return LLM_NO
    logger.warning(
        f"Could not parse Yes/No from response: {response_clean!r}. "
        f"Using fallback '{fallback}'."
    )
    return fallback


def llm_response_to_label(llm_response: str) -> str:
    """Translate an LLM response token to a dataset label ("Yes"→"Fallacy", "No"→"NoFallacy")."""
    return LLM_TO_LABEL.get(llm_response, LABEL_NO_FALLACY)


# ── Few-shot example selection ─────────────────────────────────────────────────

def sample_few_shot_examples(
        dataset,
        n_per_class: int,
        text_col: str,
        speech_turn_col: str,
        context_col: Optional[str],
        use_rag: bool,
        seed: int = 42,
) -> list[dict]:
    """Sample `n_per_class` balanced examples per label; return a flat shuffled list of dicts."""
    rng = random.Random(seed)
    examples = []
    for label in ALL_LABELS:
        subset = [ex for ex in dataset if str(ex["label"]) == label]
        chosen = rng.sample(subset, min(n_per_class, len(subset)))
        for ex in chosen:
            record = {
                "sentence": ex[text_col] or "",
                "speech_turn": ex[speech_turn_col] or "",
                "label": label,
                # Pre-translated response token used in the assistant turn of the prompt
                "llm_response": LABEL_TO_LLM[label],
            }
            if use_rag and context_col and context_col in ex:
                record["context"] = ex[context_col]
            examples.append(record)
    rng.shuffle(examples)
    return examples


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute accuracy and macro precision/recall/F1 from string predictions and references."""
    metrics = {
        "accuracy": accuracy_score(y_true=references, y_pred=predictions),
        "precision": precision_score(y_true=references, y_pred=predictions, average="macro", zero_division=0),
        "recall": recall_score(y_true=references, y_pred=predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true=references, y_pred=predictions, average="macro", zero_division=0),
    }
    # Per-class F1 for error analysis
    per_class = f1_score(
        y_true=references, y_pred=predictions, labels=ALL_LABELS, average=None, zero_division=0
    )
    for label, score in zip(ALL_LABELS, per_class):
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
        # Let Transformers handle dtype automatically for pre-quantized checkpoints
        kwargs["torch_dtype"] = "auto"
        return kwargs

    # Standard path — apply BitsAndBytes if requested
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

    Each chat prompt is tokenised once, then examples are sorted short→long so that
    every padded batch groups similar-length sequences together. This minimises the
    compute wasted on padding tokens (a plain in-order pass pads every batch to the
    longest example it happens to contain). Results are re-ordered back to the
    original input order before returning, so callers see no difference in output —
    only in speed. This helps both zero-shot and, especially, few-shot inference,
    where prompt lengths vary the most.
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

        # Place the model on an accelerator. Plain from_pretrained leaves an unquantized
        # model on CPU, where generation for a multi-billion-parameter model is extremely
        # slow — the dominant cause of slow inference. Quantized models are placed by
        # device_map at load time; unquantized ones are moved after loading.
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
) -> tuple[list[str], list[str]]:
    """Build prompts, run inference, and return (predictions, references); references may be empty."""
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

    # Step 1 – parse the raw text into "Yes" / "No"
    llm_responses = [parse_llm_response(r) for r in raw_responses]
    # Step 2 – post-process: translate to dataset labels "Fallacy" / "NoFallacy"
    predictions = [llm_response_to_label(r) for r in llm_responses]

    references = []
    if "label" in dataset.features:
        references = [str(ex["label"]) for ex in dataset]

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
            context_col=data_args.context_column_name,
            use_rag=data_args.use_rag,
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

    # Optionally merge adapter into base model
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
    """Rename label column, optionally truncate, and return a plain HF Dataset."""
    ds = raw_datasets[split_name]
    if label_column != "label" and label_column in ds.column_names:
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

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)

    logger.info(
        f"Mode: {llm_args.mode} | Backend: {llm_args.backend} | "
        f"Model: {model_args.model_name_or_path} | RAG: {data_args.use_rag}"
    )
    set_seed(training_args.seed)

    # ── Load datasets ─────────────────────────────────────────────────────────
    raw_datasets = load_raw_datasets(data_args, model_args)

    label_column = data_args.label_column_name
    train_dataset, eval_dataset, predict_dataset = None, None, None

    if training_args.do_train or llm_args.mode in ("few_shot",):
        train_dataset = prepare_split(
            raw_datasets, "train", label_column, data_args.max_train_samples
        )
        if data_args.shuffle_train_dataset:
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        logger.info(f"Train split: {len(train_dataset)} examples")

    if training_args.do_eval:
        eval_dataset = prepare_split(
            raw_datasets, "validation", label_column, data_args.max_eval_samples
        )
        logger.info(f"Validation split: {len(eval_dataset)} examples")

    if training_args.do_predict:
        predict_dataset = prepare_split(
            raw_datasets, "test", label_column, data_args.max_predict_samples
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
            context_col=data_args.context_column_name,
            use_rag=data_args.use_rag,
            seed=llm_args.few_shot_seed,
        )
        logger.info(
            f"Few-shot pool: {len(few_shot_examples)} examples "
            f"({llm_args.num_shots} per class)"
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
            # For fine_tune, reuse the fine-tuned model/tokenizer
            if llm_args.mode == "fine_tune" and inference_model is not None:
                # Wrap the already-loaded fine-tuned model
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
                gold = references[idx] if references else "N/A"
                fh.write(f"{idx}\t{pred}\t{gold}\n")
        logger.info(f"Predictions saved to {output_path}")

# ── Thin wrapper to reuse fine-tuned model for inference ──────────────────────

class _FineTunedLocalEngine:
    """Adapt an in-memory fine-tuned model to the LocalLLMInference predict_batch() interface."""

    def __init__(self, model, tokenizer, llm_args: LLMExperimentArguments):
        """Store the fine-tuned model/tokenizer and switch to left-padding for inference."""
        import torch
        self.model = model
        self.tokenizer = tokenizer
        # Switch to left-padding for inference
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
