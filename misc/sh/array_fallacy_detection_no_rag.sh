#!/bin/bash
#SBATCH --job-name=fallacy_detection_sweep
#SBATCH --time=0-36:00:00
#SBATCH --account=marianne
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/fallacy_detection_%A_%a.out
#SBATCH --array=1-297

# 1. Environment Setup
module purge
module load cuda/13.0
module load miniconda/25.1.1
conda activate rags

set -e
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface"

# Determinism — set SEED to a fixed integer to reproduce runs
SEED=42

# WandB (optional)
export WANDB_PROJECT="fallacy_detection_sweep"

# Internal Paths
SCRIPT_PATH="src/detection/run_detection.py"
TRAIN_FILE="dataset/detection/train.csv"
VAL_FILE="dataset/detection/dev.csv"
TEST_FILE="dataset/detection/test.csv"
# All array task results are pooled into this single CSV (flock prevents race conditions)
MASTER_CSV="runs/fallacy_detection/master_results.csv"

# 2. Extract Parameters for this specific Task ID
# Generate parameters file if it doesn't exist
PARAMS_FILE="runs/fallacy_detection/params.txt"
if [ ! -f "$PARAMS_FILE" ]; then
    mkdir -p "$(dirname $PARAMS_FILE)"
    
    MODELS=("google-bert/bert-base-uncased"
            "google-bert/bert-base-cased"
            "FacebookAI/roberta-base"
            "microsoft/deberta-v3-base"
            "ddore14/RooseBERT-cont-uncased"
            "ddore14/RooseBERT-cont-cased"
            "ddore14/RooseBERT-scr-uncased"
            "ddore14/RooseBERT-scr-cased"
            "ddore14/DeRooseBERTa"
            "allenai/longformer-base-4096"
            "answerdotai/ModernBERT-base"
            )
    BATCH_SIZES=(8 16 32)
    LEARNING_RATES=(2e-5 3e-5 5e-5 2e-4 3e-4 5e-4 2e-3 3e-3 5e-3)
    EPOCHS=(2 3 4)
    
    # Generate all combinations
    > "$PARAMS_FILE"
    for model in "${MODELS[@]}"; do
        for batch in "${BATCH_SIZES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for epoch in "${EPOCHS[@]}"; do
                    echo "$model $batch $lr $epoch" >> "$PARAMS_FILE"
                done
            done
        done
    done
fi

# Extract the parameters for this task
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")
MODEL=$(echo $PARAMS | cut -d' ' -f1)
BATCH=$(echo $PARAMS | cut -d' ' -f2)
LR=$(echo $PARAMS | cut -d' ' -f3)
EPOCH=$(echo $PARAMS | cut -d' ' -f4)

MODEL_NAME=$(basename "$MODEL")
CONF_ID="bs${BATCH}_lr${LR}_ep${EPOCH}"
OUTPUT_DIR="output_no_rag/${MODEL_NAME}/${CONF_ID}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname $MASTER_CSV)"

# 3. Execution
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=$SEED

python "$SCRIPT_PATH" \
    --model_name_or_path "$MODEL" \
    --cache_dir "$OUTPUT_DIR/cache/" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --text_column_name "sentence" \
    --speech_turn_column_name "speech_turn" \
    --label_column_name "fallacy_label" \
    --preprocessing_num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH" \
    --per_device_eval_batch_size "$BATCH" \
    --num_train_epochs "$EPOCH" \
    --learning_rate "$LR" \
    --logging_steps 1000 \
    --run_name "${MODEL_NAME}-bs${BATCH}-lr${LR}-ep${EPOCH}" \
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
    --do_predict True

TRAIN_EXIT=$?

# 4. Extract Results to a local task CSV
EVAL_JSON="${OUTPUT_DIR}/eval_results.json"
TASK_CSV="${OUTPUT_DIR}/master_results.csv"

if [ $TRAIN_EXIT -ne 0 ]; then
    METRICS="0,0,0,TRAIN_FAILED"
elif [ ! -f "$EVAL_JSON" ]; then
    METRICS="0,0,0,NO_EVAL_JSON"
else
    METRICS=$(python3 -c "
import json
try:
    with open('$EVAL_JSON') as f:
        r = json.load(f)
    # Adjust these keys to match your actual eval output
    accuracy = r.get('eval_accuracy', '0')
    f1 = r.get('eval_f1', '0')
    precision = r.get('eval_precision', '0')
    recall = r.get('eval_recall', '0')
    print(f'{accuracy},{f1},{precision},{recall},COMPLETED')
except Exception as e:
    print(f'0,0,0,0,JSON_ERROR: {e}')
")
fi

# Write the headerless line for this specific run
echo "${SLURM_ARRAY_TASK_ID},${MODEL_NAME},${BATCH},${LR},${EPOCH},${METRICS}" > "$TASK_CSV"

# 5. Append to master CSV with file locking (prevents race conditions)
(
    flock 200
    if [ ! -f "$MASTER_CSV" ]; then
        echo "task_id,model_name,batch_size,learning_rate,epochs,accuracy,f1,precision,recall,status,output_dir" > "$MASTER_CSV"
    fi
    echo "${SLURM_ARRAY_TASK_ID},${MODEL_NAME},${BATCH},${LR},${EPOCH},${METRICS},${OUTPUT_DIR}" >> "$MASTER_CSV"
) 200>"${MASTER_CSV}.lock"

# 6. (Optional) Rebuild master CSV from all individual task CSVs
# This helps if some jobs miss the flock
(
    flock 200
    RUNS_ROOT=$(dirname "$MASTER_CSV")
    echo "task_id,model_name,batch_size,learning_rate,epochs,accuracy,f1,precision,recall,status,output_dir" > "$MASTER_CSV"
    find "$RUNS_ROOT" -mindepth 4 -maxdepth 4 -name "master_results.csv" | sort -V | while read task_csv; do
        run_dir=$(dirname "$task_csv")
        line=$(cat "$task_csv")
        echo "${line},${run_dir}"
    done >> "$MASTER_CSV"
) 200>"${MASTER_CSV}.lock"

echo "Task ${SLURM_ARRAY_TASK_ID} completed"
