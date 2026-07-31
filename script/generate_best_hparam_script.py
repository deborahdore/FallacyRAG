"""Emit sbatch scripts re-running each model's best hyperparameter configuration, once per seed.
Reads an ``output/one-run/<data>-<task>/<dataset>`` folder and picks the best run per model by
predict_accuracy (detection) or predict_f1_macro (otherwise), then clones the matching template:
encoders from ``03-fallacy_no_rag.sh``, LLMs from ``03-fallacy_no_rag_llm.sh``.
Usage: ``python script/generate_best_hparam_script.py <output_folder> [options]``."""

import argparse
import json
import logging
import os
import re

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "predict_results.json"
RESULTS_FILE_LLM = "predict_results_metrics.json"

CONF_RE = re.compile(r"^bs(?P<batch>\d+)_lr(?P<lr>[0-9.eE+-]+)_ep(?P<epoch>\d+)$")
MODE_FINE_TUNE_RE = re.compile(r"^mode-fine_tune_bs\d+_lr(?P<lr>[0-9.eE+-]+)_ep(?P<epoch>\d+)$")
MODE_SIMPLE_RE = re.compile(r"^mode-(?P<mode>zero_shot|few_shot)$")

DATA_TASK_RE = re.compile(r"^(?P<data>.+?)-(?P<task>relation-classification|detection|classification)$")

MODELS_BLOCK_RE = re.compile(r"MODELS=\((.*?)^\)", re.DOTALL | re.MULTILINE)
MODEL_LINE_RE = re.compile(r'"([^"]+)"')
SEED_LINE_RE = re.compile(r"^SEED=\d+\s*$")

ALT_METRIC_KEY = {
    "predict_accuracy": "accuracy",
    "predict_f1_macro": "f1_macro",
}

FALLACY_ENCODER_TEMPLATE = "script/03-fallacy_no_rag.sh"
FALLACY_LLM_TEMPLATE = "script/03-fallacy_no_rag_llm.sh"
RELATION_ENCODER_TEMPLATE = "script/05-relation_classification.sh"
RELATION_LLM_TEMPLATE = "script/05-relation_classification_llm.sh"


def parse_template_models(template_path: str) -> dict[str, str]:
    """Map basename(model_path) -> full model_path, from the template's MODELS array."""
    with open(template_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    m = MODELS_BLOCK_RE.search(content)
    if not m:
        raise ValueError(f"Could not find MODELS=(...) array in {template_path}")

    mapping = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lm = MODEL_LINE_RE.search(line)
        if lm:
            path = lm.group(1)
            mapping[os.path.basename(path)] = path
    return mapping


def extract_scalar(content: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}=(\S+)\s*$", content, re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find {name}=... in template")
    return m.group(1)


def extract_header(template_path: str) -> str:
    """Everything in the template up to (not including) the 'SEED=<n>' line."""
    with open(template_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    header_lines = []
    for line in lines:
        if SEED_LINE_RE.match(line):
            break
        header_lines.append(line)
    return "".join(header_lines).rstrip("\n") + "\n"


def parse_data_task(output_folder: str) -> tuple[str, str]:
    """Split the ``<data>-<task>`` folder name into its ``data`` (e.g. elecdeb60to20) and
    ``task`` (detection / classification) components."""
    data_task = os.path.basename(os.path.dirname(output_folder))
    m = DATA_TASK_RE.match(data_task)
    if not m:
        raise ValueError(
            f"Could not parse '<data>-<task>' from folder name {data_task!r} "
            f"(expected e.g. 'elecdeb60to20-detection'). Full path: {output_folder}"
        )
    return m.group("data"), m.group("task")


def tag_job_name(header: str, suffix: str) -> str:
    header = re.sub(r'(--job-name=)(\S+)', lambda m: f"{m.group(1)}{m.group(2)}{suffix}", header, count=1)
    header = re.sub(
        r'(--output=logs/)(\S+?)(_%j\.out)', lambda m: f"{m.group(1)}{m.group(2)}{suffix}{m.group(3)}", header, count=1,
    )
    header = re.sub(
        r'(--error=logs/)(\S+?)(_%j\.out)', lambda m: f"{m.group(1)}{m.group(2)}{suffix}{m.group(3)}", header, count=1,
    )
    return header



def find_best_encoder_configs(output_folder: str, metric: str) -> dict[str, tuple[str, str, str, float]]:
    """For each model dir with bs/lr/ep run folders, find the best (batch, lr, epoch, score)."""
    best = {}
    for model_name in sorted(os.listdir(output_folder)):
        model_path = os.path.join(output_folder, model_name)
        if not os.path.isdir(model_path):
            continue

        best_score, best_conf, seen_llm_style = None, None, False

        for run_name in sorted(os.listdir(model_path)):
            run_path = os.path.join(model_path, run_name)
            if not os.path.isdir(run_path):
                continue

            conf_m = CONF_RE.match(run_name)
            if not conf_m:
                if run_name.startswith("mode-"):
                    seen_llm_style = True
                continue

            results_path = os.path.join(run_path, RESULTS_FILE)
            if not os.path.isfile(results_path):
                continue
            try:
                with open(results_path, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
            except json.JSONDecodeError:
                logger.warning("Bad JSON in %s — skipping", results_path)
                continue

            score = metrics.get(metric)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_conf = (conf_m.group("batch"), conf_m.group("lr"), conf_m.group("epoch"))

        if best_conf is not None:
            best[model_name] = (*best_conf, best_score)
        elif seen_llm_style:
            logger.info("%s is LLM-style ('mode-...') — will be handled by the LLM template.", model_name)
        else:
            logger.warning("No usable %s runs found for %s — skipping.", RESULTS_FILE, model_name)

    return best


def build_encoder_script(output_folder: str, template_path: str, seeds: list[int]) -> tuple[str, int]:
    dataset = os.path.basename(output_folder)
    data, task = parse_data_task(output_folder)

    model_map = parse_template_models(template_path)
    metric = "predict_accuracy" if task == "detection" else "predict_f1_macro"
    best_configs = find_best_encoder_configs(output_folder, metric)

    entries = []
    for model_name, (batch, lr, epoch, score) in best_configs.items():
        if model_name not in model_map:
            logger.warning("%s has no matching entry in %s's MODELS array — skipping.", model_name, template_path)
            continue
        entries.append((model_map[model_name], batch, lr, epoch))
        logger.info("[encoder] %-30s best = bs%s_lr%s_ep%s (%s=%.4f)", model_name, batch, lr, epoch, metric, score)

    if not entries:
        return "", 0

    header = tag_job_name(extract_header(template_path), "_best_seeds")
    seeds_str = " ".join(str(s) for s in seeds)
    configs_str = "\n".join(f'    "{m}|{b}|{lr}|{e}"' for m, b, lr, e in entries)

    body = f"""wandb disabled

data="{data}"  # elecdeb60to20 | propaganda
task={task}
dataset="{dataset}"

export WANDB_PROJECT="fallacy_$task"

TRAIN_FILE="datasets/$data-dataset/$task/$dataset/train.csv"
VAL_FILE="datasets/$data-dataset/$task/$dataset/dev.csv"
TEST_FILE="datasets/$data-dataset/$task/$dataset/test.csv"

SEEDS=({seeds_str})

# Best hyperparameters per model, selected by {metric} from {output_folder}.
# Format: "<model_path>|<batch>|<lr>|<epoch>"
BEST_CONFIGS=(
{configs_str}
)

FAILED_LOG="logs/failed_runs_best_${{task}}.log"
mkdir -p logs
: > "$FAILED_LOG"  # truncate/create fresh each run

for SEED in "${{SEEDS[@]}}"; do
  export PYTHONHASHSEED=$SEED
  SEED_NUMBER=$SEED

  for entry in "${{BEST_CONFIGS[@]}}"; do
    IFS='|' read -r model batch lr epoch <<< "$entry"
    MODEL_NAME=$(basename "$model")
    CONF_ID="bs${{batch}}_lr${{lr}}_ep${{epoch}}_seed${{SEED}}"

    OUTPUT_DIR="output/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
    CACHE_DIR="cache/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
    mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

    echo "=== Running $MODEL_NAME | $CONF_ID ==="

    if ! python "src/$task/run_$task.py" \\
        --model_name_or_path "$model" \\
        --cache_dir "$CACHE_DIR" \\
        --train_file "$TRAIN_FILE" \\
        --validation_file "$VAL_FILE" \\
        --test_file "$TEST_FILE" \\
        --text_column_name "sentence" \\
        --speech_turn_column_name "speech_turn" \\
        --label_column_name "fallacy_label" \\
        --preprocessing_num_workers 8 \\
        --output_dir "$OUTPUT_DIR" \\
        --per_device_train_batch_size "$batch" \\
        --per_device_eval_batch_size "$batch" \\
        --num_train_epochs "$epoch" \\
        --learning_rate "$lr" \\
        --logging_steps 1000 \\
        --run_name "${{MODEL_NAME}}-bs${{batch}}-lr${{lr}}-ep${{epoch}}-seed${{SEED}}" \\
        --eval_strategy "epoch" \\
        --eval_on_start True \\
        --save_strategy "epoch" \\
        --save_total_limit 1 \\
        --seed $SEED \\
        --data_seed $SEED \\
        --dataloader_num_workers 8 \\
        --dataloader_prefetch_factor 2 \\
        --remove_unused_columns True \\
        --do_train True \\
        --do_eval True \\
        --do_predict True \\
        --trust_remote_code True \\
        --warmup_ratio 0.06 \\
        --use_rag True \\
        --context_column_name "rag_context"
    then
      echo "FAILED: $model $CONF_ID" >> "$FAILED_LOG"
      echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
    fi
  done
done

echo "Best-hyperparameter multi-seed sweep complete. Failed runs (if any) logged in $FAILED_LOG."
"""
    return header + "\n" + body, len(entries)



def find_best_llm_configs(output_folder: str, metric: str) -> dict[str, tuple[str, str | None, str | None, float]]:
    """For each model dir with 'mode-...' run folders, find the best (mode, lr, epoch, score)
    across fine_tune's bs/lr/ep sweep AND the zero_shot/few_shot single runs."""
    alt_key = ALT_METRIC_KEY[metric]
    best = {}
    for model_name in sorted(os.listdir(output_folder)):
        model_path = os.path.join(output_folder, model_name)
        if not os.path.isdir(model_path):
            continue

        best_score, best_conf = None, None

        for run_name in sorted(os.listdir(model_path)):
            run_path = os.path.join(model_path, run_name)
            if not os.path.isdir(run_path):
                continue

            mode, lr, epoch = None, None, None
            ft_m = MODE_FINE_TUNE_RE.match(run_name)
            simple_m = MODE_SIMPLE_RE.match(run_name)
            if ft_m:
                mode, lr, epoch = "fine_tune", ft_m.group("lr"), ft_m.group("epoch")
            elif simple_m:
                mode = simple_m.group("mode")
            else:
                continue

            results_path = os.path.join(run_path, RESULTS_FILE_LLM)
            if not os.path.isfile(results_path):
                continue
            try:
                with open(results_path, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
            except json.JSONDecodeError:
                logger.warning("Bad JSON in %s — skipping", results_path)
                continue

            score = metrics.get(alt_key)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_conf = (mode, lr, epoch)

        if best_conf is not None:
            best[model_name] = (*best_conf, best_score)

    return best


def build_llm_script(output_folder: str, template_path: str, seeds: list[int]) -> tuple[str, int]:
    dataset = os.path.basename(output_folder)
    data, task = parse_data_task(output_folder)

    with open(template_path, "r", encoding="utf-8") as fh:
        template_content = fh.read()

    model_map = parse_template_models(template_path)
    n_gpus = extract_scalar(template_content, "N_GPUS")
    grad_acc = extract_scalar(template_content, "GRAD_ACC")
    batch = extract_scalar(template_content, "BATCH")
    default_batch = extract_scalar(template_content, "DEFAULT_BATCH")

    metric = "predict_accuracy" if task == "detection" else "predict_f1_macro"
    best_configs = find_best_llm_configs(output_folder, metric)

    entries = []
    for model_name, (mode, lr, epoch, score) in best_configs.items():
        if model_name not in model_map:
            logger.warning("%s has no matching entry in %s's MODELS array — skipping.", model_name, template_path)
            continue
        entries.append((model_map[model_name], mode, lr or "", epoch or ""))
        if mode == "fine_tune":
            logger.info("[llm]     %-30s best = mode-fine_tune_lr%s_ep%s (%s=%.4f)", model_name, lr, epoch, metric,
                        score)
        else:
            logger.info("[llm]     %-30s best = mode-%s (%s=%.4f)", model_name, mode, metric, score)

    if not entries:
        return "", 0

    header = tag_job_name(extract_header(template_path), "_best_seeds")
    seeds_str = " ".join(str(s) for s in seeds)
    configs_str = "\n".join(f'    "{m}|{mode}|{lr}|{e}"' for m, mode, lr, e in entries)

    body = f"""wandb disabled

data="{data}"  # elecdeb60to20 | propaganda
task={task}
dataset="{dataset}"

export WANDB_PROJECT="fallacy_$task"

TRAIN_FILE="datasets/$data-dataset/$task/$dataset/train.csv"
VAL_FILE="datasets/$data-dataset/$task/$dataset/dev.csv"
TEST_FILE="datasets/$data-dataset/$task/$dataset/test.csv"

N_GPUS={n_gpus}
GRAD_ACC={grad_acc}
BATCH={batch}
DEFAULT_BATCH={default_batch}

SEEDS=({seeds_str})

# Best config per model (fine_tune sweep + zero_shot + few_shot), by {metric} from {output_folder}.
# Format: "<model_path>|<mode>|<lr>|<epoch>"  (lr/epoch empty for zero_shot/few_shot)
BEST_CONFIGS=(
{configs_str}
)

FAILED_LOG="logs/failed_runs_best_llm_${{task}}.log"
mkdir -p logs
: > "$FAILED_LOG"  # truncate/create fresh each run

for SEED in "${{SEEDS[@]}}"; do
  export PYTHONHASHSEED=$SEED
  SEED_NUMBER=$SEED

  for entry in "${{BEST_CONFIGS[@]}}"; do
    IFS='|' read -r model mode lr epoch <<< "$entry"
    MODEL_NAME=$(basename "$model")

    if [[ "$mode" == "fine_tune" ]]; then
      CONF_ID="mode-fine_tune_lr${{lr}}_ep${{epoch}}_seed${{SEED}}"
      OUTPUT_DIR="output/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      CACHE_DIR="cache/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

      echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

      if ! torchrun \\
          --nproc_per_node=${{N_GPUS}} \\
          --rdzv_backend=c10d \\
          --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\
          "src/${{task}}/run_${{task}}_llm.py" \\
          --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \\
          --cache_dir "$CACHE_DIR" \\
          --train_file "$TRAIN_FILE" \\
          --validation_file "$VAL_FILE" \\
          --test_file "$TEST_FILE" \\
          --text_column_name "sentence" \\
          --speech_turn_column_name "speech_turn" \\
          --label_column_name "fallacy_label" \\
          --mode "$mode" \\
          --output_dir "$OUTPUT_DIR" \\
          --per_device_train_batch_size "$BATCH" \\
          --per_device_eval_batch_size "$BATCH" \\
          --gradient_accumulation_steps $GRAD_ACC \\
          --num_train_epochs "$epoch" \\
          --learning_rate "$lr" \\
          --logging_steps 1000 \\
          --run_name "${{task}}-${{dataset}}-${{MODEL_NAME}}-bs${{BATCH}}-lr${{lr}}-ep${{epoch}}-mode-${{mode}}-seed${{SEED}}" \\
          --eval_strategy "epoch" \\
          --eval_on_start True \\
          --save_strategy "epoch" \\
          --save_total_limit 1 \\
          --seed $SEED \\
          --data_seed $SEED \\
          --dataloader_num_workers 8 \\
          --dataloader_prefetch_factor 2 \\
          --remove_unused_columns True \\
          --do_train True \\
          --do_eval True \\
          --do_predict True \\
          --trust_remote_code True \\
          --warmup_ratio 0.06 \\
          --use_rag True \\
          --context_column_name "rag_context"
      then
        echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
        echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
      fi

    else
      CONF_ID="mode-${{mode}}_seed${{SEED}}"
      OUTPUT_DIR="output/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      CACHE_DIR="cache/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

      echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

      if ! torchrun \\
          --nproc_per_node=${{N_GPUS}} \\
          --rdzv_backend=c10d \\
          --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\
          "src/${{task}}/run_${{task}}_llm.py" \\
          --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \\
          --cache_dir "$CACHE_DIR" \\
          --train_file "$TRAIN_FILE" \\
          --validation_file "$VAL_FILE" \\
          --test_file "$TEST_FILE" \\
          --text_column_name "sentence" \\
          --speech_turn_column_name "speech_turn" \\
          --label_column_name "fallacy_label" \\
          --mode "$mode" \\
          --output_dir "$OUTPUT_DIR" \\
          --per_device_eval_batch_size "$DEFAULT_BATCH" \\
          --run_name "${{task}}-${{dataset}}-${{MODEL_NAME}}-mode-${{mode}}-seed${{SEED}}" \\
          --seed $SEED \\
          --data_seed $SEED \\
          --dataloader_num_workers 8 \\
          --dataloader_prefetch_factor 2 \\
          --remove_unused_columns True \\
          --do_train False \\
          --do_eval True \\
          --do_predict True \\
          --trust_remote_code True \\
          --warmup_ratio 0.06 \\
          --use_rag True \\
          --context_column_name "rag_context"
      then
        echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
        echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
      fi
    fi
  done
done

echo "Best-hyperparameter multi-seed LLM sweep complete. Failed runs (if any) logged in $FAILED_LOG."
"""
    return header + "\n" + body, len(entries)



def build_relation_encoder_script(output_folder: str, template_path: str, seeds: list[int]) -> tuple[str, int]:
    dataset = os.path.basename(output_folder)
    data, task = parse_data_task(output_folder)

    model_map = parse_template_models(template_path)
    metric = "predict_f1_macro"
    best_configs = find_best_encoder_configs(output_folder, metric)

    entries = []
    for model_name, (batch, lr, epoch, score) in best_configs.items():
        if model_name not in model_map:
            logger.warning("%s has no matching entry in %s's MODELS array — skipping.", model_name, template_path)
            continue
        entries.append((model_map[model_name], batch, lr, epoch))
        logger.info("[relation] %-30s best = bs%s_lr%s_ep%s (%s=%.4f)", model_name, batch, lr, epoch, metric, score)

    if not entries:
        return "", 0

    header = tag_job_name(extract_header(template_path), "_best_seeds")
    seeds_str = " ".join(str(s) for s in seeds)
    configs_str = "\n".join(f'    "{m}|{b}|{lr}|{e}"' for m, b, lr, e in entries)

    body = f"""wandb disabled

data="{data}"
task={task}
dataset="{dataset}"

export WANDB_PROJECT="{task}"

# Sentence-pair CSVs live directly under the task dir (no per-config subfolder).
TRAIN_FILE="datasets/$data-dataset/$task/train.csv"
VAL_FILE="datasets/$data-dataset/$task/dev.csv"
TEST_FILE="datasets/$data-dataset/$task/test.csv"

SEEDS=({seeds_str})

# Best hyperparameters per model, selected by {metric} from {output_folder}.
# Format: "<model_path>|<batch>|<lr>|<epoch>"
BEST_CONFIGS=(
{configs_str}
)

FAILED_LOG="logs/failed_runs_best_${{task}}.log"
mkdir -p logs
: > "$FAILED_LOG"  # truncate/create fresh each run

for SEED in "${{SEEDS[@]}}"; do
  export PYTHONHASHSEED=$SEED
  SEED_NUMBER=$SEED

  for entry in "${{BEST_CONFIGS[@]}}"; do
    IFS='|' read -r model batch lr epoch <<< "$entry"
    MODEL_NAME=$(basename "$model")
    CONF_ID="bs${{batch}}_lr${{lr}}_ep${{epoch}}_seed${{SEED}}"

    OUTPUT_DIR="output/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
    CACHE_DIR="cache/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
    mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

    echo "=== Running $MODEL_NAME | $CONF_ID ==="

    if ! python src/relation-extraction/run_relation_classification.py \\
        --model_name_or_path "$model" \\
        --cache_dir "$CACHE_DIR" \\
        --train_file "$TRAIN_FILE" \\
        --validation_file "$VAL_FILE" \\
        --test_file "$TEST_FILE" \\
        --label_column_name "label" \\
        --do_train \\
        --do_eval \\
        --do_predict \\
        --per_device_train_batch_size "$batch" \\
        --per_device_eval_batch_size "$batch" \\
        --learning_rate "$lr" \\
        --num_train_epochs "$epoch" \\
        --run_name "${{MODEL_NAME}}-bs${{batch}}-lr${{lr}}-ep${{epoch}}-seed${{SEED}}" \\
        --seed $SEED \\
        --data_seed $SEED \\
        --output_dir "$OUTPUT_DIR"
    then
      echo "FAILED: $model $CONF_ID" >> "$FAILED_LOG"
      echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
    fi
  done
done

echo "Best-hyperparameter multi-seed relation sweep complete. Failed runs (if any) logged in $FAILED_LOG."
"""
    return header + "\n" + body, len(entries)


def build_relation_llm_script(output_folder: str, template_path: str, seeds: list[int]) -> tuple[str, int]:
    dataset = os.path.basename(output_folder)
    data, task = parse_data_task(output_folder)

    with open(template_path, "r", encoding="utf-8") as fh:
        template_content = fh.read()

    model_map = parse_template_models(template_path)
    n_gpus = extract_scalar(template_content, "N_GPUS")
    grad_acc = extract_scalar(template_content, "GRAD_ACC")
    batch = extract_scalar(template_content, "BATCH")
    default_batch = extract_scalar(template_content, "DEFAULT_BATCH")

    metric = "predict_f1_macro"
    best_configs = find_best_llm_configs(output_folder, metric)

    entries = []
    for model_name, (mode, lr, epoch, score) in best_configs.items():
        if model_name not in model_map:
            logger.warning("%s has no matching entry in %s's MODELS array — skipping.", model_name, template_path)
            continue
        entries.append((model_map[model_name], mode, lr or "", epoch or ""))
        if mode == "fine_tune":
            logger.info("[relation-llm] %-26s best = mode-fine_tune_lr%s_ep%s (%s=%.4f)", model_name, lr, epoch, metric,
                        score)
        else:
            logger.info("[relation-llm] %-26s best = mode-%s (%s=%.4f)", model_name, mode, metric, score)

    if not entries:
        return "", 0

    header = tag_job_name(extract_header(template_path), "_best_seeds")
    seeds_str = " ".join(str(s) for s in seeds)
    configs_str = "\n".join(f'    "{m}|{mode}|{lr}|{e}"' for m, mode, lr, e in entries)

    body = f"""wandb disabled

data="{data}"
task={task}
dataset="{dataset}"

export WANDB_PROJECT="{task}"

SCRIPT="src/relation-extraction/run_relation_classification_llm.py"

# Sentence-pair CSVs live directly under the task dir (no per-config subfolder).
TRAIN_FILE="datasets/$data-dataset/$task/train.csv"
VAL_FILE="datasets/$data-dataset/$task/dev.csv"
TEST_FILE="datasets/$data-dataset/$task/test.csv"

N_GPUS={n_gpus}
GRAD_ACC={grad_acc}
BATCH={batch}
DEFAULT_BATCH={default_batch}

SEEDS=({seeds_str})

# Best config per model (fine_tune sweep + zero_shot + few_shot), by {metric} from {output_folder}.
# Format: "<model_path>|<mode>|<lr>|<epoch>"  (lr/epoch empty for zero_shot/few_shot)
BEST_CONFIGS=(
{configs_str}
)

FAILED_LOG="logs/failed_runs_best_llm_${{task}}.log"
mkdir -p logs
: > "$FAILED_LOG"  # truncate/create fresh each run

for SEED in "${{SEEDS[@]}}"; do
  export PYTHONHASHSEED=$SEED
  SEED_NUMBER=$SEED

  for entry in "${{BEST_CONFIGS[@]}}"; do
    IFS='|' read -r model mode lr epoch <<< "$entry"
    MODEL_NAME=$(basename "$model")

    if [[ "$mode" == "fine_tune" ]]; then
      CONF_ID="mode-fine_tune_lr${{lr}}_ep${{epoch}}_seed${{SEED}}"
      OUTPUT_DIR="output/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      CACHE_DIR="cache/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

      echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

      if ! torchrun \\
          --nproc_per_node=${{N_GPUS}} \\
          --rdzv_backend=c10d \\
          --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\
          "$SCRIPT" \\
          --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \\
          --cache_dir "$CACHE_DIR" \\
          --train_file "$TRAIN_FILE" \\
          --validation_file "$VAL_FILE" \\
          --test_file "$TEST_FILE" \\
          --text_column_name "sentence1" \\
          --speech_turn_column_name "sentence2" \\
          --label_column_name "label_value" \\
          --mode "$mode" \\
          --output_dir "$OUTPUT_DIR" \\
          --per_device_train_batch_size "$BATCH" \\
          --per_device_eval_batch_size "$BATCH" \\
          --gradient_accumulation_steps $GRAD_ACC \\
          --num_train_epochs "$epoch" \\
          --learning_rate "$lr" \\
          --logging_steps 1000 \\
          --run_name "${{task}}-${{dataset}}-${{MODEL_NAME}}-bs${{BATCH}}-lr${{lr}}-ep${{epoch}}-mode-${{mode}}-seed${{SEED}}" \\
          --eval_strategy "epoch" \\
          --eval_on_start True \\
          --save_strategy "epoch" \\
          --save_total_limit 1 \\
          --seed $SEED \\
          --data_seed $SEED \\
          --dataloader_num_workers 8 \\
          --dataloader_prefetch_factor 2 \\
          --remove_unused_columns True \\
          --do_train True \\
          --do_eval True \\
          --do_predict True \\
          --trust_remote_code True
      then
        echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
        echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
      fi

    else
      CONF_ID="mode-${{mode}}_seed${{SEED}}"
      OUTPUT_DIR="output/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      CACHE_DIR="cache/five-runs/$data-$task/$dataset/${{MODEL_NAME}}/seed_${{SEED_NUMBER}}/${{CONF_ID}}"
      mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

      echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

      if ! torchrun \\
          --nproc_per_node=${{N_GPUS}} \\
          --rdzv_backend=c10d \\
          --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\
          "$SCRIPT" \\
          --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \\
          --cache_dir "$CACHE_DIR" \\
          --train_file "$TRAIN_FILE" \\
          --validation_file "$VAL_FILE" \\
          --test_file "$TEST_FILE" \\
          --text_column_name "sentence1" \\
          --speech_turn_column_name "sentence2" \\
          --label_column_name "label_value" \\
          --mode "$mode" \\
          --output_dir "$OUTPUT_DIR" \\
          --per_device_eval_batch_size "$DEFAULT_BATCH" \\
          --run_name "${{task}}-${{dataset}}-${{MODEL_NAME}}-mode-${{mode}}-seed${{SEED}}" \\
          --seed $SEED \\
          --data_seed $SEED \\
          --dataloader_num_workers 8 \\
          --dataloader_prefetch_factor 2 \\
          --remove_unused_columns True \\
          --do_train False \\
          --do_eval True \\
          --do_predict True \\
          --trust_remote_code True
      then
        echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
        echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
      fi
    fi
  done
done

echo "Best-hyperparameter multi-seed relation LLM sweep complete. Failed runs (if any) logged in $FAILED_LOG."
"""
    return header + "\n" + body, len(entries)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_folder",
                        help="e.g. output/one-run/elecdeb60to20-detection/temporal=false-retriever=false-summarizer=false")
    parser.add_argument("--template", default=None, help="Encoder template sbatch script.")
    parser.add_argument("--llm-template", default=None, help="LLM template sbatch script.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[14, 42, 55, 89, 3],
                        help="Seeds to run each best config with.")
    parser.add_argument("--output", default=None,
                        help="Path for the encoder script (default: script/06-best_hparams_<data>_<task>_<dataset>.sh).")
    parser.add_argument("--llm-output", default=None,
                        help="Path for the LLM script (default: script/06-best_hparams_llm_<data>_<task>_<dataset>.sh).")
    args = parser.parse_args()

    output_folder = args.output_folder.rstrip("/")
    if not os.path.isdir(output_folder):
        raise SystemExit(f"Output folder not found: {output_folder}")

    dataset = os.path.basename(output_folder)
    try:
        data, task = parse_data_task(output_folder)
    except ValueError as exc:
        raise SystemExit(str(exc))

    is_relation = task == "relation-classification"
    encoder_template = args.template or (RELATION_ENCODER_TEMPLATE if is_relation else FALLACY_ENCODER_TEMPLATE)
    llm_template = args.llm_template or (RELATION_LLM_TEMPLATE if is_relation else FALLACY_LLM_TEMPLATE)

    if not os.path.isfile(encoder_template):
        raise SystemExit(f"Encoder template not found: {encoder_template}")
    if not os.path.isfile(llm_template):
        raise SystemExit(f"LLM template not found: {llm_template}")

    encoder_dest = args.output or os.path.join("script", f"06-best_hparams_{data}_{task}_{dataset}.sh")
    llm_dest = args.llm_output or os.path.join("script", f"06-best_hparams_llm_{data}_{task}_{dataset}.sh")

    if is_relation:
        encoder_text, n_encoder = build_relation_encoder_script(output_folder, encoder_template, args.seeds)
        llm_text, n_llm = build_relation_llm_script(output_folder, llm_template, args.seeds)
    else:
        encoder_text, n_encoder = build_encoder_script(output_folder, encoder_template, args.seeds)
        llm_text, n_llm = build_llm_script(output_folder, llm_template, args.seeds)

    if not encoder_text and not llm_text:
        raise SystemExit("No matching best configurations found (encoder or LLM) — nothing to write.")

    if encoder_text:
        os.makedirs(os.path.dirname(encoder_dest), exist_ok=True)
        with open(encoder_dest, "w", encoding="utf-8") as fh:
            fh.write(encoder_text)
        os.chmod(encoder_dest, 0o755)
        logger.info(
            "Wrote encoder script (%d models x %d seeds = %d runs) → %s",
            n_encoder, len(args.seeds), n_encoder * len(args.seeds), encoder_dest,
        )

    if llm_text:
        os.makedirs(os.path.dirname(llm_dest), exist_ok=True)
        with open(llm_dest, "w", encoding="utf-8") as fh:
            fh.write(llm_text)
        os.chmod(llm_dest, 0o755)
        logger.info(
            "Wrote LLM script (%d models x %d seeds = %d runs) → %s",
            n_llm, len(args.seeds), n_llm * len(args.seeds), llm_dest,
        )


if __name__ == "__main__":
    main()
