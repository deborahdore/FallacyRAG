#!/bin/bash
#SBATCH --job-name=encoders_fallacy_detection
#SBATCH --output=logs/encoders_fallacy_detection.out
#SBATCH --error=logs/encoders_fallacy_detection.out
#SBATCH --gres=gpu:1
#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=FAIL

set -e
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface"
export WANDB_PROJECT="fallacy_detection"
export WANDB_PROJECT="fallacy_detection"

SEED=123
export PYTHONHASHSEED=$SEED

wandb disabled

dataset="agentic=false-temporal=false-retriever=false-summarizer=false"

TRAIN_FILE="dataset/detection/$dataset/train.csv"
VAL_FILE="dataset/detection/$dataset/dev.csv"
TEST_FILE="dataset/detection/$dataset/test.csv"

MODELS=(
#    "microsoft/deberta-v3-base"
#    "MARIANNE-INRIA/RooseBERT-cont-uncased"
#    "MARIANNE-INRIA/RooseBERT-cont-cased"
#    "MARIANNE-INRIA/RooseBERT-scr-uncased"
#    "MARIANNE-INRIA/RooseBERT-scr-cased"
#    "allenai/longformer-base-4096"
#    "answerdotai/ModernBERT-base"
#    "FacebookAI/roberta-base"
#    "FacebookAI/xlm-roberta-base"
#    "casehold/legalbert"
    "chandar-lab/NeoBERT"
)

BATCH_SIZES=(8 16 32)
LEARNING_RATES=(2e-5 3e-5 5e-5)
EPOCHS=(2 3 4)

for model in "${MODELS[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        MODEL_NAME=$(basename "$model")
        CONF_ID="bs${batch}_lr${lr}_ep${epoch}"

        OUTPUT_DIR="output/$dataset/${MODEL_NAME}/${CONF_ID}"
        mkdir -p "$OUTPUT_DIR"

        python src/detection/run_detection.py \
            --model_name_or_path "$model" \
            --cache_dir "./cache/" \
            --train_file "$TRAIN_FILE" \
            --validation_file "$VAL_FILE" \
            --test_file "$TEST_FILE" \
            --text_column_name "sentence" \
            --speech_turn_column_name "speech_turn" \
            --label_column_name "fallacy_label" \
            --preprocessing_num_workers 8 \
            --output_dir "$OUTPUT_DIR" \
            --per_device_train_batch_size "$batch" \
            --per_device_eval_batch_size "$batch" \
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
            --trust_remote_code True

      done
    done
  done
done
