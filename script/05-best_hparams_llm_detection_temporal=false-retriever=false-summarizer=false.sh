#!/bin/bash
#SBATCH --job-name=llm_fallacy_no_rag_best_seeds
#SBATCH --output=logs/llm_fallacy_no_rag_best_seeds_%j.out
#SBATCH --error=logs/llm_fallacy_no_rag_best_seeds_%j.out

#SBATCH --gpus=h100:2
#SBATCH --partition=gpu
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2

#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=FAIL
#SBATCH --time=36:00:00
#SBATCH --account=marianne

set -uo pipefail

module purge
module load miniconda/25.1.1
module load cuda/13.0
conda activate rags

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/home/ddore/.cache/huggingface"

export MASTER_ADDR=localhost
export MASTER_PORT=6000

wandb disabled

task=detection
dataset="temporal=false-retriever=false-summarizer=false"

export WANDB_PROJECT="fallacy_$task"

TRAIN_FILE="dataset/$task/$dataset/train.csv"
VAL_FILE="dataset/$task/$dataset/dev.csv"
TEST_FILE="dataset/$task/$dataset/test.csv"

N_GPUS=2
GRAD_ACC=2
BATCH=2
DEFAULT_BATCH=8

SEEDS=(14 42 55 89 3)

# Best configuration per model (across fine_tune sweep + zero_shot + few_shot),
# selected by predict_accuracy from:
#   output/detection/temporal=false-retriever=false-summarizer=false
# Format: "<model_path>|<mode>|<lr>|<epoch>"  (lr/epoch empty for zero_shot/few_shot)
BEST_CONFIGS=(
    "meta-llama/Llama-3.1-8B-Instruct|fine_tune|5e-4|3"
    "mistralai/Mistral-7B-Instruct-v0.3|zero_shot||"
    "Qwen/Qwen3.5-9B-Base|fine_tune|5e-4|3"
    "openai/gpt-oss-20b|fine_tune|5e-4|3"
)

FAILED_LOG="logs/failed_runs_best_llm_${task}.log"
mkdir -p logs
: > "$FAILED_LOG"  # truncate/create fresh each run

for SEED in "${SEEDS[@]}"; do
  export PYTHONHASHSEED=$SEED

  for entry in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r model mode lr epoch <<< "$entry"
    MODEL_NAME=$(basename "$model")

    if [[ "$mode" == "fine_tune" ]]; then
      CONF_ID="mode-fine_tune_lr${lr}_ep${epoch}_seed${SEED}"
      OUTPUT_DIR="output_5runs/$task/$dataset/${MODEL_NAME}/${CONF_ID}"
      mkdir -p "$OUTPUT_DIR"

      echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

      if ! torchrun \
          --nproc_per_node=${N_GPUS} \
          --rdzv_backend=c10d \
          --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
          "src/${task}/run_${task}_llm.py" \
          --model_name_or_path "$model" \
          --cache_dir "./cache/" \
          --train_file "$TRAIN_FILE" \
          --validation_file "$VAL_FILE" \
          --test_file "$TEST_FILE" \
          --text_column_name "sentence" \
          --speech_turn_column_name "speech_turn" \
          --label_column_name "fallacy_label" \
          --mode "$mode" \
          --output_dir "$OUTPUT_DIR" \
          --per_device_train_batch_size "$BATCH" \
          --per_device_eval_batch_size "$BATCH" \
          --gradient_accumulation_steps $GRAD_ACC \
          --num_train_epochs "$epoch" \
          --learning_rate "$lr" \
          --logging_steps 1000 \
          --run_name "${task}-${dataset}-${MODEL_NAME}-bs${BATCH}-lr${lr}-ep${epoch}-mode-${mode}-seed${SEED}" \
          --eval_strategy "epoch" \
          --eval_on_start True \
          --save_strategy "epoch" \
          --save_total_limit 1 \
          --seed $SEED \
          --data_seed $SEED \
          --dataloader_num_workers 8 \
          --dataloader_prefetch_factor 2 \
          --remove_unused_columns True \
          --do_train True \
          --do_eval True \
          --do_predict True \
          --trust_remote_code True \
          --warmup_ratio 0.06
      then
        echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
        echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
      fi

    else
      CONF_ID="mode-${mode}_seed${SEED}"
      OUTPUT_DIR="output_5runs/$task/$dataset/${MODEL_NAME}/${CONF_ID}"
      mkdir -p "$OUTPUT_DIR"

      echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

      if ! torchrun \
          --nproc_per_node=${N_GPUS} \
          --rdzv_backend=c10d \
          --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
          "src/${task}/run_${task}_llm.py" \
          --model_name_or_path "$model" \
          --cache_dir "./cache/" \
          --train_file "$TRAIN_FILE" \
          --validation_file "$VAL_FILE" \
          --test_file "$TEST_FILE" \
          --text_column_name "sentence" \
          --speech_turn_column_name "speech_turn" \
          --label_column_name "fallacy_label" \
          --mode "$mode" \
          --output_dir "$OUTPUT_DIR" \
          --per_device_eval_batch_size "$DEFAULT_BATCH" \
          --run_name "${task}-${dataset}-${MODEL_NAME}-mode-${mode}-seed${SEED}" \
          --seed $SEED \
          --data_seed $SEED \
          --dataloader_num_workers 8 \
          --dataloader_prefetch_factor 2 \
          --remove_unused_columns True \
          --do_train False \
          --do_eval True \
          --do_predict True \
          --trust_remote_code True \
          --warmup_ratio 0.06
      then
        echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
        echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
      fi
    fi
  done
done

echo "Best-hyperparameter multi-seed LLM sweep complete. Failed runs (if any) logged in $FAILED_LOG."
