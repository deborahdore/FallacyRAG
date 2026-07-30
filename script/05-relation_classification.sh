#!/bin/bash
#SBATCH --job-name=relation_classification
#SBATCH --output=logs/relation_classification_%j.out
#SBATCH --error=logs/relation_classification_%j.out

#SBATCH --time=36:00:00
#SBATCH --account=marianne
#SBATCH --gpus=h100:1
#SBATCH --partition=gpu

#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=FAIL

module purge
module load miniconda
conda activate rags

set -uo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=14
export PYTHONHASHSEED=$SEED
wandb disabled
export WANDB_PROJECT="relation_classification"

task=relation-classification

DATASETS=(
#    "LIARArg-dataset"
    "change-my-view-dataset"
)

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

FAILED_LOG="logs/failed_runs_${task}.log"
mkdir -p logs

for dataset in "${DATASETS[@]}"; do
  TRAIN_FILE="datasets/$dataset/$task/train.csv"
  VAL_FILE="datasets/$dataset/$task/dev.csv"
  TEST_FILE="datasets/$dataset/$task/test.csv"

  for batch in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for model in "${MODELS[@]}"; do
          MODEL_NAME=$(basename "$model")
          CONF_ID="bs${batch}_lr${lr}_ep${epoch}"
          OUTPUT_DIR="output/one-run/$dataset-$task/${MODEL_NAME}/${CONF_ID}"
          CACHE_DIR="cache/one-run/$dataset-$task/${MODEL_NAME}/${CONF_ID}"
          mkdir -p "$OUTPUT_DIR"

          echo "=== [$dataset] Running $MODEL_NAME | $CONF_ID ==="

          if ! python src/relation-extraction/run_relation_classification.py \
              --model_name_or_path "$model" \
              --cache_dir "$CACHE_DIR" \
              --train_file "$TRAIN_FILE" \
              --validation_file "$VAL_FILE" \
              --test_file "$TEST_FILE" \
              --label_column_name "label" \
              --do_train \
              --do_eval \
              --do_predict \
              --per_device_train_batch_size "$batch" \
              --per_device_eval_batch_size "$batch" \
              --learning_rate "$lr" \
              --num_train_epochs "$epoch" \
              --run_name "${dataset}-${MODEL_NAME}-bs${batch}-lr${lr}-ep${epoch}" \
              --seed $SEED \
              --data_seed $SEED \
              --output_dir "$OUTPUT_DIR"
          then
            echo "FAILED: $dataset $model $CONF_ID" >> "$FAILED_LOG"
            echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
          fi
        done
      done
    done
  done
done

echo "Sweep complete. Failed runs (if any) logged in $FAILED_LOG."
