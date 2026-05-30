#!/bin/bash
#SBATCH --job-name=fallacy_detection
#SBATCH --account=vqm@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/fallacy_detection_%j.out
#SBATCH --error=logs/fallacy_detection_%j.out
#SBATCH --hint=nomultithread

#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=BEGIN,FAIL

module purge
module load cuda/12.4.1
module load miniforge/24.9.0
conda activate rags

set -e
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/linkhome/rech/genzqh01/ubq61ty/.cache/huggingface/hub/"
wandb offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEED=14
export PYTHONHASHSEED=$SEED

export WANDB_PROJECT="fallacy_detection"

TRAIN_FILE="dataset/detection/train.csv"
VAL_FILE="dataset/detection/dev.csv"
TEST_FILE="dataset/detection/test.csv"

MODELS=(
    "microsoft/deberta-v3-base"
    "MARIANNE-INRIA/RooseBERT-cont-uncased"
    "MARIANNE-INRIA/RooseBERT-cont-cased"
    "MARIANNE-INRIA/RooseBERT-scr-uncased"
    "MARIANNE-INRIA/RooseBERT-scr-cased"
    "allenai/longformer-base-4096"
    "answerdotai/ModernBERT-base"
    "FacebookAI/roberta-base"
    "FacebookAI/xlm-roberta-base"
    "casehold/legalbert"
    "chandar-lab/NeoBERT"
)

BATCH_SIZES=(8 16 32)
LEARNING_RATES=(1e-5 2e-5 3e-5 5e-5 8e-5)
EPOCHS=(2 3 4 5)

SKIP_CONFS=(
    "bs16_lr1e-5_ep2"  "bs8_lr1e-5_ep2"   "bs8_lr1e-5_ep3"   "bs8_lr1e-5_ep4"   "bs8_lr1e-5_ep5"
    "bs8_lr2e-5_ep2"   "bs8_lr2e-5_ep3"   "bs8_lr2e-5_ep4"
    "bs8_lr3e-5_ep2"   "bs8_lr3e-5_ep3"   "bs8_lr3e-5_ep5"
    "bs8_lr5e-5_ep2"   "bs8_lr5e-5_ep3"   "bs8_lr5e-5_ep4"   "bs8_lr5e-5_ep5"
    "bs8_lr8e-5_ep2"   "bs8_lr8e-5_ep3"   "bs8_lr8e-5_ep4"   "bs8_lr8e-5_ep5"
)

for model in "${MODELS[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        MODEL_NAME=$(basename "$model")
        CONF_ID="bs${batch}_lr${lr}_ep${epoch}"

        # Skip already-completed combinations
        skip=false
        for s in "${SKIP_CONFS[@]}"; do
          if [[ "$CONF_ID" == "$s" ]]; then
            skip=true
            break
          fi
        done
        if $skip; then
          echo "Skipping $CONF_ID"
          continue
        fi

        OUTPUT_DIR="output_no_rag/${MODEL_NAME}/${CONF_ID}"
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