"""LLM-based fallacy detection: zero-shot, few-shot, and LoRA/QLoRA fine-tuning.
Each example becomes an instruction-following chat prompt answered with "Yes" or "No", carrying
the pre-built "context" column when ``--use_rag``, then translated to dataset labels:
Fallacy/NoFallacy or propaganda/non-propaganda. The system prompt and label pair are selected
from the dataset path. Backends: local HuggingFace generate() or the OpenAI chat API.
The templated prompt never exceeds MAX_PROMPT_TOKENS (2048)."""

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils.dataset_prompts import get_detection_label_config, get_detection_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

LLM_YES = "Yes"
LLM_NO = "No"

LABEL_FALLACY = "Fallacy"
LABEL_NO_FALLACY = "NoFallacy"
ALL_LABELS = [LABEL_FALLACY, LABEL_NO_FALLACY]

LABEL_TO_LLM: dict[str, str] = {LABEL_FALLACY: LLM_YES, LABEL_NO_FALLACY: LLM_NO}
LLM_TO_LABEL: dict[str, str] = {LLM_YES: LABEL_FALLACY, LLM_NO: LABEL_NO_FALLACY}

SYSTEM_PROMPT = None

USER_QUESTION = 'Does the sentence contain a fallacy? Answer "Yes" or "No".'


def configure_detection_labels(train_file: Optional[str]) -> None:
    """Set the dataset-aware label and question globals from the detection config."""
    global LABEL_FALLACY, LABEL_NO_FALLACY, ALL_LABELS, LABEL_TO_LLM, LLM_TO_LABEL
    global USER_QUESTION
    cfg = get_detection_label_config(train_file)
    LABEL_FALLACY = cfg["positive_label"]
    LABEL_NO_FALLACY = cfg["negative_label"]
    ALL_LABELS = [LABEL_FALLACY, LABEL_NO_FALLACY]
    LABEL_TO_LLM = {LABEL_FALLACY: LLM_YES, LABEL_NO_FALLACY: LLM_NO}
    LLM_TO_LABEL = {LLM_YES: LABEL_FALLACY, LLM_NO: LABEL_NO_FALLACY}
    target = "propaganda technique" if cfg["dataset"] == "propaganda" else "fallacy"
    USER_QUESTION = f'Does the sentence contain a {target}? Answer "Yes" or "No".'


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
        default="./datasets/elecdeb60to20-dataset/detection/train.csv",
        metadata={"help": "Training data file (CSV or JSON)."},
    )
    validation_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/detection/dev.csv",
        metadata={"help": "Validation data file (CSV or JSON)."},
    )
    test_file: Optional[str] = field(
        default="./datasets/elecdeb60to20-dataset/detection/test.csv",
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
    few_shot_pool_size: int = field(
        default=10,
        metadata={"help": "Few-shot demonstrations are drawn at random among the N shortest training "
                          "examples of each class (sentence + speech turn + context), to save context."},
    )
    max_new_tokens: int = field(
        default=4,
        metadata={"help": "Maximum number of new tokens to generate per example. "
                          "The answer is a single word (\"Yes\"/\"No\"), so a small cap "
                          "avoids wasted decode steps; raise it only for verbose models."},
    )
    reasoning_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Generation cap for models with a reasoning channel (gpt-oss), which "
                          "must finish reasoning before writing the answer."},
    )
    max_prompt_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on the templated prompt, never above 2048 and never above the model's "
                          "context window minus the generation cap (default: that bound). Speech turns "
                          "and retrieved contexts are clipped only if the prompt exceeds it; the "
                          "prompt itself is never truncated."},
    )
    speech_turn_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Optional token budget for the speech turn of the target example "
                          "(default: no limit beyond --max_prompt_tokens)."},
    )
    context_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Optional token budget for the retrieved context of the target example "
                          "(default: no limit beyond --max_prompt_tokens)."},
    )
    shot_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Optional token budget for the speech turn and the context of each "
                          "few-shot demonstration (default: no limit beyond --max_prompt_tokens)."},
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


MAX_PROMPT_TOKENS = 2048  # hard ceiling on the templated prompt, to fit in memory
MIN_CLIP_TOKENS = 64  # floor for clipping, so a RAG context is shortened but never dropped


def render_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Apply the chat template with thinking off (Qwen) and low reasoning effort (gpt-oss).
    Templates that do not use these variables ignore them."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False, reasoning_effort="low",
    )


def clip_text(text: Optional[str], max_tokens: int, tokenizer, anchor: Optional[str] = None):
    """Keep at most max_tokens tokens of text, centred on anchor when it occurs in text
    (otherwise the head is kept). Cuts are marked with "[...]"."""
    if not text or tokenizer is None:
        return text
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if len(offsets) <= max_tokens:
        return text
    pos = text.find(anchor) if anchor else -1
    first = sum(1 for s, _ in offsets if s < pos) if pos > 0 else 0
    start = max(0, min(first - max_tokens // 2, len(offsets) - max_tokens))
    end = start + max_tokens
    clipped = text[offsets[start][0]:offsets[end - 1][1]]
    return ("[...] " if start else "") + clipped + (" [...]" if end < len(offsets) else "")


def build_user_message(
        sentence: str,
        speech_turn: str,
        context: Optional[str] = None,
) -> str:
    """Build the user-turn text for a single example."""
    parts = [f'Sentence: "{sentence}"']
    if speech_turn:
        parts.append(f"Surrounding text:\n{speech_turn}")
    if context:
        parts.append(f"Retrieved context:\n{context}")
    parts.append(USER_QUESTION)
    return "\n\n".join(parts)


def build_messages(
        sentence: str,
        speech_turn: str,
        context: Optional[str],
        examples: Optional[list[dict]],
        tokenizer,
        llm_args: "LLMExperimentArguments",
) -> list[dict]:
    """Chat messages for one example, with optional few-shot demonstrations as prior turns.
    Speech turns and contexts are clipped, and budgets halved until the prompt fits max_prompt_tokens."""
    cap = min(llm_args.max_prompt_tokens or MAX_PROMPT_TOKENS, MAX_PROMPT_TOKENS)
    turn_n = llm_args.speech_turn_max_tokens or cap
    ctx_n = llm_args.context_max_tokens or cap
    shot_n = llm_args.shot_max_tokens or cap
    while True:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in examples or []:
            messages.append({"role": "user", "content": build_user_message(
                ex["sentence"],
                clip_text(ex["speech_turn"], shot_n, tokenizer, ex["sentence"]),
                clip_text(ex.get("context"), shot_n, tokenizer),
            )})
            messages.append({"role": "assistant", "content": ex["llm_response"]})
        messages.append({"role": "user", "content": build_user_message(
            sentence,
            clip_text(speech_turn, turn_n, tokenizer, sentence),
            clip_text(context, ctx_n, tokenizer),
        )})
        if tokenizer is None or max(turn_n, ctx_n, shot_n) <= MIN_CLIP_TOKENS:
            return messages
        n_tokens = len(tokenizer(render_chat(tokenizer, messages), add_special_tokens=False)["input_ids"])
        if n_tokens <= cap:
            return messages
        turn_n, ctx_n, shot_n = (max(MIN_CLIP_TOKENS, n // 2) for n in (turn_n, ctx_n, shot_n))


def format_for_sft(
        example: dict,
        tokenizer,
        text_col: str,
        speech_turn_col: str,
        context_col: Optional[str],
        use_rag: bool,
        llm_args: "LLMExperimentArguments",
) -> dict:
    """Tokenize one training example; the loss is computed on the answer tokens only."""
    sentence = example[text_col] or ""
    speech_turn = (example[speech_turn_col] if speech_turn_col in example else "") or ""
    context = example[context_col] if (use_rag and context_col and context_col in example) else None
    llm_target = LABEL_TO_LLM.get(str(example["label"]), LLM_NO)

    messages = build_messages(sentence, speech_turn, context, None, tokenizer, llm_args)
    prompt = render_chat(tokenizer, messages)
    # answer + end-of-turn exactly as the template writes it, probed without a system prompt
    # (Mistral attaches the system prompt to the last user turn, so the full chat is no prefix match)
    probe = [{"role": "user", "content": "Q"}]
    head = render_chat(tokenizer, probe)
    full = render_chat(tokenizer, probe + [{"role": "assistant", "content": "ANSWER"}], False)
    completion = (full[len(head):].replace("ANSWER", llm_target) if full.startswith(head)
                  else llm_target + tokenizer.eos_token)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def parse_llm_response(response: str, fallback: str = LLM_NO) -> str:
    """Return the first standalone "Yes"/"No" word in a raw LLM response, else *fallback*."""
    response_clean = response.strip()
    match = re.search(r"\b(yes|no)\b", response_clean.lower())  # whole words: "not"/"know" don't count
    if match:
        return LLM_YES if match.group(1) == "yes" else LLM_NO
    logger.warning(
        f"Could not parse Yes/No from response: {response_clean!r}. "
        f"Using fallback '{fallback}'."
    )
    return fallback


def llm_response_to_label(llm_response: str) -> str:
    """Translate an LLM response token to a dataset label ("Yes"→"Fallacy", "No"→"NoFallacy")."""
    return LLM_TO_LABEL.get(llm_response, LABEL_NO_FALLACY)


def sample_few_shot_examples(
        dataset,
        n_per_class: int,
        text_col: str,
        speech_turn_col: str,
        context_col: Optional[str],
        use_rag: bool,
        seed: int = 42,
        pool_size: int = 10,
) -> list[dict]:
    """Per label, draw `n_per_class` examples at random among its `pool_size` shortest ones
    (length of the fields that enter the prompt); return a flat shuffled list of dicts."""
    rng = random.Random(seed)
    fields = [text_col, speech_turn_col] + ([context_col] if use_rag and context_col else [])
    examples = []
    for label in ALL_LABELS:
        subset = sorted((ex for ex in dataset if str(ex["label"]) == label),
                        key=lambda ex: sum(len(ex.get(c) or "") for c in fields))
        pool = subset[:max(pool_size, n_per_class)]
        chosen = rng.sample(pool, min(n_per_class, len(pool)))
        for ex in chosen:
            record = {
                "sentence": ex[text_col] or "",
                "speech_turn": (ex[speech_turn_col] if speech_turn_col in ex else "") or "",
                "label": label,
                "llm_response": LABEL_TO_LLM[label],
            }
            if use_rag and context_col and context_col in ex:
                record["context"] = ex[context_col]
            examples.append(record)
    rng.shuffle(examples)
    return examples


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute accuracy and macro precision/recall/F1 from string predictions and references."""
    metrics = {
        "accuracy": accuracy_score(y_true=references, y_pred=predictions),
        "precision": precision_score(y_true=references, y_pred=predictions, average="macro", zero_division=0),
        "recall": recall_score(y_true=references, y_pred=predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true=references, y_pred=predictions, average="macro", zero_division=0),
    }
    per_class = f1_score(
        y_true=references, y_pred=predictions, labels=ALL_LABELS, average=None, zero_division=0
    )
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
        reasoning_max_new_tokens: int = 512,
) -> list[str]:
    """Length-sorted, left-padded batched generation: prompts are sorted short to long so each batch
    wastes little compute on padding, then results are re-ordered back to the input order."""
    # the chat template already holds the BOS token, so none is added here and nothing is truncated
    encodings = [tokenizer(render_chat(tokenizer, msgs), add_special_tokens=False)
                 for msgs in messages_list]
    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]))
    if order:
        logger.info(f"Longest prompt: {len(encodings[order[-1]]['input_ids'])} tokens")

    # gpt-oss reasons on an analysis channel before answering on the final channel
    has_reasoning = "<|channel|>" in str(tokenizer.chat_template or "")
    # stop at the end of the assistant turn, which can differ from the tokenizer eos (Qwen base: <|im_end|>)
    vocab = tokenizer.get_vocab()
    gen_eos = model.generation_config.eos_token_id
    stop_ids = {tokenizer.eos_token_id, *(gen_eos if isinstance(gen_eos, list) else [gen_eos]),
                *(vocab[tok] for tok in ("<|im_end|>", "<|eot_id|>") if tok in vocab)} - {None}
    gen_kwargs = dict(
        max_new_tokens=reasoning_max_new_tokens if has_reasoning else max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=sorted(stop_ids),
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
            decoded = tokenizer.decode(out[prompt_len:], skip_special_tokens=False)
            if "<|channel|>" in decoded:  # keep only the gpt-oss final channel, empty if never reached
                final = "<|channel|>final<|message|>"
                decoded = decoded.split(final)[1] if final in decoded else ""
            for tok in tokenizer.all_special_tokens:
                decoded = decoded.replace(tok, "")
            responses[idxs[k]] = re.sub(r"<\|[^|>]*\|>", "", decoded).strip()

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
        self.reasoning_max_new_tokens = llm_args.reasoning_max_new_tokens

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
            reasoning_max_new_tokens=self.reasoning_max_new_tokens,
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
) -> tuple[list[str], list[str]]:
    """Build prompts, run inference, and return (predictions, references); references may be empty."""
    text_col = data_args.text_column_name
    speech_col = data_args.speech_turn_column_name
    ctx_col = data_args.context_column_name

    tokenizer = getattr(inference_engine, "tokenizer", None)
    examples = few_shot_examples if llm_args.mode == "few_shot" else None
    messages_list = []
    for ex in dataset:
        sentence = ex[text_col] or ""
        speech_turn = (ex[speech_col] if speech_col in ex else "") or ""
        context = ex[ctx_col] if (data_args.use_rag and ctx_col and ctx_col in ex) else None
        messages_list.append(build_messages(sentence, speech_turn, context, examples, tokenizer, llm_args))

    logger.info(f"Running inference on {len(messages_list)} {split_name} examples …")
    raw_responses = inference_engine.predict_batch(messages_list)

    llm_responses = [parse_llm_response(r) for r in raw_responses]
    predictions = [llm_response_to_label(r) for r in llm_responses]

    references = []
    if "label" in dataset.features:
        references = [str(ex["label"]) for ex in dataset]

    return predictions, references


def run_fine_tuning(
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        llm_args: LLMExperimentArguments,
        training_args: TrainingArguments,
        train_dataset,
        eval_dataset,
) -> tuple:
    """Fine-tune with LoRA/QLoRA and an answer-only loss; return (model, tokenizer) ready for inference."""
    import dataclasses

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer

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
    if "quantization_config" in load_kwargs:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    def _fmt(ex):
        """Tokenize one example for SFT."""
        return format_for_sft(
            ex,
            tokenizer=tokenizer,
            text_col=data_args.text_column_name,
            speech_turn_col=data_args.speech_turn_column_name,
            context_col=data_args.context_column_name,
            use_rag=data_args.use_rag,
            llm_args=llm_args,
        )

    train_sft = train_dataset.map(_fmt, remove_columns=train_dataset.column_names,
                                  desc="Tokenizing train for SFT")
    eval_sft = (eval_dataset.map(_fmt, remove_columns=eval_dataset.column_names,
                                 desc="Tokenizing eval for SFT") if eval_dataset else None)
    if eval_sft is None:
        training_args = dataclasses.replace(training_args, eval_strategy="no", load_best_model_at_end=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_sft,
        eval_dataset=eval_sft,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
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


def prepare_split(raw_datasets, split_name, label_column, max_samples=None):
    """Rename label column, optionally truncate, and return a plain HF Dataset."""
    ds = raw_datasets[split_name]
    if label_column != "label" and label_column in ds.column_names:
        ds = ds.rename_column(label_column, "label")
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def main():
    """Run the chosen mode (zero-shot, few-shot, or fine-tune) and evaluate/predict."""
    global SYSTEM_PROMPT

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LLMExperimentArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, llm_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, llm_args, training_args = parser.parse_args_into_dataclasses()

    SYSTEM_PROMPT = get_detection_prompt(data_args.train_file)
    configure_detection_labels(data_args.train_file)

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)

    llm_args.max_prompt_tokens = min(llm_args.max_prompt_tokens or MAX_PROMPT_TOKENS, MAX_PROMPT_TOKENS)
    if llm_args.backend == "local":
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        window = (getattr(cfg, "max_position_embeddings", None)
                  or getattr(getattr(cfg, "text_config", None), "max_position_embeddings", None))
        if window:
            room = window - max(llm_args.max_new_tokens, llm_args.reasoning_max_new_tokens)
            llm_args.max_prompt_tokens = min(llm_args.max_prompt_tokens, room)
        logger.info(f"Model context window: {window} | prompt cap: {llm_args.max_prompt_tokens}")

    logger.info(
        f"Mode: {llm_args.mode} | Backend: {llm_args.backend} | "
        f"Model: {model_args.model_name_or_path} | RAG: {data_args.use_rag}"
    )
    set_seed(training_args.seed)

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
            pool_size=llm_args.few_shot_pool_size,
        )
        logger.info(
            f"Few-shot pool: {len(few_shot_examples)} examples "
            f"({llm_args.num_shots} per class)"
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
                gold = references[idx] if references else "N/A"
                fh.write(f"{idx}\t{pred}\t{gold}\n")
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
        self.reasoning_max_new_tokens = llm_args.reasoning_max_new_tokens
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
            reasoning_max_new_tokens=self.reasoning_max_new_tokens,
        )


def _save_metrics(metrics: dict, path: str):
    """Write a metrics dict to *path* as indented JSON."""
    import json
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved to {path}")


if __name__ == "__main__":
    main()