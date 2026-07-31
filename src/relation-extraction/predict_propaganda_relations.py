"""Predict propaganda relations with a LIARArg fine-tuned Llama (LoRA) checkpoint.
Runs the adapter produced by ``run_relation_classification_llm.py`` over the prediction pairs
built by ``prepare_propaganda.py``, generating Attack / Support / No Relation per pair. Prompt
and parser are imported from the training script so inference matches fine-tuning exactly.
There are no gold relations for these pairs, so this is pure inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_relation_classification_llm as rc_llm

RELATION2ID = {"attack": 0, "support": 1, "no_relation": 2}
ID2RELATION = {v: k for k, v in RELATION2ID.items()}

TEXT_COLUMN = "fallacy"
PAIR_COLUMN = "candidate_relation"

DEFAULT_MODEL_DIR = ROOT / "output" / "trained-models" / "LIARArg-Llama"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATA_FILE = ROOT / "datasets" / "propaganda-dataset" / "relation-classification" / "dataset.csv"
DEFAULT_OUTPUT_FILE = ROOT / "datasets" / "propaganda-dataset" / "relation-classification" / "predictions.csv"

LIARARG_TRAIN_HINT = "datasets/LIARArg-dataset/relation-classification/train.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="LoRA adapter directory, or a merged/full causal-LM checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model for the adapter. Read from adapter_config.json when omitted.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help="Propaganda prediction pairs CSV (from prepare_propaganda.py).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Where to write the predictions CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max prompt token length (prompts longer than this are truncated).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Tokens to generate per example (enough to fit one relation name).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--cache-dir", type=str, default="./cache/", help="HuggingFace cache directory.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token for gated models.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code from Hub repos.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit NF4.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load the base model in 8-bit.")
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


def resolve_base_model(model_dir: Path, override: str | None) -> str | None:
    """Return the adapter's base model, or None when *model_dir* is a full checkpoint."""
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.is_file():
        return None
    if override:
        return override
    with open(adapter_config) as fh:
        base = json.load(fh).get("base_model_name_or_path")
    return base or DEFAULT_BASE_MODEL


def load_model_and_tokenizer(args: argparse.Namespace, model_dir: Path, device: str):
    """Load the fine-tuned Llama (base + LoRA adapter, or a merged checkpoint) for generation."""
    base_model = resolve_base_model(model_dir, args.base_model)
    weights_source = base_model or str(model_dir)

    tokenizer_source = str(model_dir) if (model_dir / "tokenizer_config.json").is_file() else weights_source
    print(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_args = rc_llm.LLMExperimentArguments(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    base_kwargs = dict(
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    load_kwargs = rc_llm._build_load_kwargs(weights_source, quant_args, base_kwargs, torch)
    is_quantized = "quantization_config" in load_kwargs
    if device == "cuda":
        load_kwargs.setdefault("device_map", "auto")

    print(f"Loading weights from: {weights_source}")
    model = AutoModelForCausalLM.from_pretrained(weights_source, **load_kwargs)

    if base_model is not None:
        from peft import PeftModel

        print(f"Applying LoRA adapter from: {model_dir}")
        model = PeftModel.from_pretrained(model, str(model_dir))

    if not is_quantized and "device_map" not in load_kwargs:
        if device == "cpu":
            print("WARNING: running on CPU — generation will be slow. Use a GPU or --load-in-4bit.")
        model = model.to(device)

    model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    return model, tokenizer


def label_first_token_ids(tokenizer) -> dict[str, int]:
    """Map each canonical relation to the first token id of its readable name."""
    ids = {}
    for canon in rc_llm.RELATION_LABELS:
        readable = rc_llm.LABEL_TO_LLM[canon]
        token_ids = tokenizer.encode(readable, add_special_tokens=False)
        ids[canon] = token_ids[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"Relation names share a first token under this tokenizer: {ids}")
    return ids


@torch.no_grad()
def generate(
        frame: pd.DataFrame,
        model,
        tokenizer,
        batch_size: int,
        max_length: int,
        max_new_tokens: int,
        temperature: float,
) -> tuple[list[str], np.ndarray]:
    """Return the raw responses and an (N, 3) array of first-token relation probabilities."""
    first_token = label_first_token_ids(tokenizer)
    s1 = frame[TEXT_COLUMN].tolist()
    s2 = frame[PAIR_COLUMN].tolist()

    encodings = []
    for a, b in zip(s1, s2):
        messages = rc_llm.build_zero_shot_messages(a, b)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encodings.append(tokenizer(text, truncation=True, max_length=max_length))

    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]))

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False

    n = len(encodings)
    responses: list[str] = [""] * n
    probs = np.zeros((n, len(RELATION2ID)), dtype=np.float32)

    start = time.time()
    n_batches = (n + batch_size - 1) // batch_size
    for i in tqdm(range(0, n, batch_size), total=n_batches, desc="Generating", unit="batch"):
        idxs = order[i: i + batch_size]
        inputs = tokenizer.pad(
            [encodings[j] for j in idxs],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        out = model.generate(**inputs, **gen_kwargs)

        first_logits = out.scores[0].float()
        label_logits = torch.stack(
            [first_logits[:, first_token[rel]] for rel in ID2RELATION.values()], dim=-1
        )
        batch_probs = torch.softmax(label_logits, dim=-1).cpu().numpy()

        prompt_len = inputs["input_ids"].shape[1]
        for k, seq in enumerate(out.sequences):
            responses[idxs[k]] = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            probs[idxs[k]] = batch_probs[k]

        done = min(i + batch_size, n)
        rate = done / (time.time() - start + 1e-9)
        if done == n:
            print(f"  {done}/{n}  ({rate:.1f} rows/s)")

    return responses, probs


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"Project root: {ROOT}")
    print(f"Device: {device}")

    rc_llm._apply_relation_config(LIARARG_TRAIN_HINT)
    print(f"Applied relation map: {ID2RELATION}")

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {model_dir}\n"
            f"Pass the correct path with --model-dir (a LoRA adapter folder with "
            f"adapter_config.json + adapter_model.safetensors, or a merged full checkpoint)."
        )

    model, tokenizer = load_model_and_tokenizer(args, model_dir, device)

    df = pd.read_csv(args.data_file)
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[PAIR_COLUMN] = df[PAIR_COLUMN].fillna("").astype(str)

    work_df = df
    print(f"Predicting {len(work_df)} rows.")

    responses, probs = generate(
        work_df,
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    parsed = [rc_llm.parse_llm_response(r) for r in responses]
    n_unparsed = sum(p is None for p in parsed)
    relations = [
        p if p is not None else ID2RELATION[int(probs[i].argmax())]
        for i, p in enumerate(parsed)
    ]

    result = work_df.copy()
    result["predicted_relation"] = relations
    result["predicted_label"] = [RELATION2ID[r] for r in relations]
    result["raw_response"] = [r.strip() for r in responses]
    result["confidence"] = probs.max(axis=1)
    for i, name in ID2RELATION.items():
        result[f"prob_{name}"] = probs[:, i]

    print("\nPredicted relation distribution:")
    print(result["predicted_relation"].value_counts())
    print(f"Unparsed responses (fell back to token probabilities): {n_unparsed}")
    print(f"Mean confidence: {float(result['confidence'].mean()):.4f}")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
