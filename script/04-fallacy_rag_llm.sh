#!/bin/bash
#SBATCH --job-name=llm_fallacy_rag
#SBATCH --account=fuo@a100
#SBATCH -C a100
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

#SBATCH --time=10:00:00
#SBATCH --output=logs/llm_fallacy_rag_%j.out
#SBATCH --error=logs/llm_fallacy_rag_%j.out

#SBATCH --hint=nomultithread

#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=FAIL

module purge
module load arch/a100
module load cuda/13.0.3
module load miniforge/24.9.0

conda activate rags

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=localhost
export MASTER_PORT=6000

wandb disabled

SEED=14
export PYTHONHASHSEED=$SEED

data="elecdeb60to20"  # elecdeb60to20 | reddit
task=detection

export WANDB_PROJECT="fallacy_$task"

DATASETS=()

MODES=("fine_tune" "zero_shot" "few_shot")

MODELS=(
"meta-llama/Llama-3.1-8B-Instruct"
"Qwen/Qwen3.5-9B-Base"
"openai/gpt-oss-20b"
"mistralai/Mistral-7B-Instruct-v0.3"
)

N_GPUS=2
GRAD_ACC=2
BATCH=2

DEFAULT_BATCH=8

LEARNING_RATES=(1e-4 3e-4 5e-4)
EPOCHS=(1 2 3)

mkdir -p logs


FAILED_LOG="logs/failed_runs_$task.log"

for dataset in "${DATASETS[@]}"; do
  TRAIN_FILE="datasets/$data-dataset/$task/$dataset/train.csv"
  VAL_FILE="datasets/$data-dataset/$task/$dataset/dev.csv"
  TEST_FILE="datasets/$data-dataset/$task/$dataset/test.csv"

  for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
      MODEL_NAME=$(basename "$model")

      if [[ "$mode" == "zero_shot" || "$mode" == "few_shot" ]]; then
        CONF_ID="mode-${mode}"
        OUTPUT_DIR="output/one-run/$data-$task/$dataset/${MODEL_NAME}/${CONF_ID}"
        CACHE_DIR="cache/one-run/$data-$task/$dataset/${MODEL_NAME}/${CONF_ID}"
        mkdir -p "$OUTPUT_DIR"

        echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

        if ! torchrun \
            --nproc_per_node=${N_GPUS} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            "src/${task}/run_${task}_llm.py" \
            --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \
            --cache_dir "$CACHE_DIR" \
            --train_file "$TRAIN_FILE" \
            --validation_file "$VAL_FILE" \
            --test_file "$TEST_FILE" \
            --text_column_name "sentence" \
            --speech_turn_column_name "speech_turn" \
            --label_column_name "fallacy_label" \
            --mode "$mode" \
            --output_dir "$OUTPUT_DIR" \
            --per_device_eval_batch_size "$DEFAULT_BATCH" \
            --run_name "${task}-${dataset}-${MODEL_NAME}-mode-${mode}" \
            --seed $SEED \
            --data_seed $SEED \
            --dataloader_num_workers 8 \
            --dataloader_prefetch_factor 2 \
            --remove_unused_columns True \
            --do_train False \
            --do_eval True \
            --do_predict True \
            --trust_remote_code True \
            --use_rag True \
            --context_column_name "rag_context"
        then
          echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
          echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
        fi

      else
        for lr in "${LEARNING_RATES[@]}"; do
          for epoch in "${EPOCHS[@]}"; do
            CONF_ID="mode-${mode}_bs$((BATCH*N_GPUS*GRAD_ACC))_lr${lr}_ep${epoch}"
            OUTPUT_DIR="output/one-run/$data-$task/$dataset/${MODEL_NAME}/${CONF_ID}"
            CACHE_DIR="cache/one-run/$data-$task/$dataset/${MODEL_NAME}/${CONF_ID}"
            mkdir -p "$OUTPUT_DIR"

            echo "=== [$task] Running $MODEL_NAME | $CONF_ID ==="

            if ! torchrun \
                --nproc_per_node=${N_GPUS} \
                --rdzv_backend=c10d \
                --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                "src/${task}/run_${task}_llm.py" \
                --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \
                --cache_dir "$CACHE_DIR" \
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
                --run_name "${task}-${dataset}-${MODEL_NAME}-bs${BATCH}-lr${lr}-ep${epoch}-mode-${mode}" \
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
                --use_rag True \
                --context_column_name "rag_context"
            then
              echo "FAILED: $task $model $CONF_ID" >> "$FAILED_LOG"
              echo "!!! Run failed, see $FAILED_LOG. Continuing with next configuration."
            fi
          done
        done
      fi
    done
  done
done

echo "Task '$task' sweep complete. Failed runs (if any) logged in $FAILED_LOG."