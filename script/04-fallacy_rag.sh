#!/bin/bash
#SBATCH --job-name=encoders_rag
#SBATCH --output=logs/encoders_rag_%j.out
#SBATCH --error=logs/encoders_rag_%j.out

#SBATCH --time=36:00:00
#SBATCH --account=marianne
#SBATCH --gpus=h100:1
#SBATCH --partition=gpu

#SBATCH --mail-user=anonymous@anonymous.org
#SBATCH --mail-type=FAIL

module purge
module load miniconda
conda activate rags

set -uo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/anonymous/.cache/huggingface"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=14
export PYTHONHASHSEED=$SEED
wandb disabled

data="elecdeb60to20"  # elecdeb60to20 | propaganda
DATASETS=(
)

task=detection

export WANDB_PROJECT="fallacy_$task"

MODELS=(
    "microsoft/deberta-v3-base"
    "allenai/longformer-base-4096"
    "answerdotai/ModernBERT-base"
    "FacebookAI/roberta-base"
    "casehold/legalbert"
    "MARIANNE-INRIA/RooseBERT-cont-uncased"
    "MARIANNE-INRIA/RooseBERT-cont-cased"
    "MARIANNE-INRIA/RooseBERT-scr-uncased"
    "MARIANNE-INRIA/RooseBERT-scr-cased"
    "chandar-lab/NeoBERT"
)

BATCH_SIZES=(8 16)
LEARNING_RATES=(2e-5 3e-5 5e-5)
EPOCHS=(3 5)
GRAD_ACC=1
FAILED_LOG="logs/failed_runs_${task}.log"

for dataset in "${DATASETS[@]}"; do
  TRAIN_FILE="datasets/$data-dataset/$task/$dataset/train.csv"
  VAL_FILE="datasets/$data-dataset/$task/$dataset/dev.csv"
  TEST_FILE="datasets/$data-dataset/$task/$dataset/test.csv"

  for model in "${MODELS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
          MODEL_NAME=$(basename "$model")
          CONF_ID="bs${batch}_lr${lr}_ep${epoch}"

          OUTPUT_DIR="output/one-run/$data-$task/$dataset/${MODEL_NAME}/${CONF_ID}"
          CACHE_DIR="cache/one-run/$data-$task/$dataset/${MODEL_NAME}/${CONF_ID}"
          mkdir -p "$OUTPUT_DIR"

          echo "=== Running $MODEL_NAME | $CONF_ID ==="

          if ! python "src/$task/run_$task.py" \
              --model_name_or_path "$model" \
              --cache_dir "$CACHE_DIR" \
              --train_file "$TRAIN_FILE" \
              --validation_file "$VAL_FILE" \
              --test_file "$TEST_FILE" \
              --text_column_name "sentence" \
              --speech_turn_column_name "speech_turn" \
              --label_column_name "fallacy_label" \
              --preprocessing_num_workers 8 \
              --output_dir "$OUTPUT_DIR" \
              --per_device_train_batch_size "$((batch/GRAD_ACC))" \
              --per_device_eval_batch_size "$((batch/GRAD_ACC))" \
              --num_train_epochs "$epoch" \
              --learning_rate "$lr" \
              --logging_steps 1000 \
              --run_name "${MODEL_NAME}-bs${batch}-lr${lr}-ep${epoch}" \
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
              --warmup_ratio 0.06 \
              --use_rag True \
              --context_column_name "rag_context" \
              --gradient_accumulation_steps $GRAD_ACC
          then
            echo "FAILED: $model $CONF_ID, Dataset: $dataset" >> "$FAILED_LOG"
            echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
          fi
        done
      done
    done
  done
done
echo "Sweep complete. Failed runs (if any) logged in $FAILED_LOG."