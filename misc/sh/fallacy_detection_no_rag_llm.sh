#!/bin/bash
#SBATCH --job-name=llm_fallacy_detection
#SBATCH --output=logs/llm_fallacy_detection.out
#SBATCH --error=logs/llm_fallacy_detection.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=FAIL

set -e
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export WANDB_PROJECT="fallacy_detection"

SEED=123
export PYTHONHASHSEED=$SEED

wandb disabled

dataset="agentic=false-temporal=false-retriever=false-summarizer=false"

TRAIN_FILE="dataset/detection/$dataset/train.csv"
VAL_FILE="dataset/detection/$dataset/dev.csv"
TEST_FILE="dataset/detection/$dataset/test.csv"

MODES=("fine_tune" "zero_shot" "few_shot")

MODELS=(
#"mistralai/Mistral-7B-Instruct-v0.3"
#"meta-llama/Llama-3.1-8B-Instruct"
"Qwen/Qwen3.5-9B-Base"
"openai/gpt-oss-20b"
)

N_GPUS=2
grad_acc=4

BATCH_SIZES=(1 2 4)
LEARNING_RATES=(1e-4 3e-4 5e-4)
EPOCHS=(1 2 3)

DEFAULT_BATCH=8

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    MODEL_NAME=$(basename "$model")

    if [[ "$mode" == "zero_shot" || "$mode" == "few_shot" ]]; then
      CONF_ID="mode-${mode}"
      OUTPUT_DIR="output/$dataset/${MODEL_NAME}/${CONF_ID}"
      mkdir -p "$OUTPUT_DIR"

      torchrun \
          --nproc_per_node=${N_GPUS} \
          --rdzv_backend=c10d \
          --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
          src/detection/run_detection_llm.py \
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
          --run_name "${MODEL_NAME}-mode-${mode}" \
          --seed $SEED \
          --data_seed $SEED \
          --dataloader_num_workers 8 \
          --dataloader_prefetch_factor 2 \
          --remove_unused_columns True \
          --do_train False \
          --do_eval True \
          --do_predict True \
          --trust_remote_code True
    else
      for batch in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for epoch in "${EPOCHS[@]}"; do
            CONF_ID="mode-${mode}_bs$((batch*grad_acc*N_GPUS))_lr${lr}_ep${epoch}"
            OUTPUT_DIR="output/$dataset/${MODEL_NAME}/${CONF_ID}"
            mkdir -p "$OUTPUT_DIR"

            torchrun \
                --nproc_per_node=${N_GPUS} \
                --rdzv_backend=c10d \
                --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                src/detection/run_detection_llm.py \
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
                --per_device_train_batch_size "$batch" \
                --per_device_eval_batch_size "$batch" \
                --gradient_accumulation_steps $grad_acc \
                --num_train_epochs "$epoch" \
                --learning_rate "$lr" \
                --logging_steps 1000 \
                --run_name "${MODEL_NAME}-bs${batch}-lr${lr}-ep${epoch}-mode-${mode}" \
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
    fi
  done
done
