#!/bin/bash
#SBATCH --job-name=mistralai
#SBATCH --account=vqm@a100
#SBATCH -C a100
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=logs/mistralai_%j.out
#SBATCH --error=logs/mistralai_%j.out
#SBATCH --hint=nomultithread

#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=BEGIN,FAIL

module purge
module load arch/a100
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

export MASTER_PORT=6000

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

SEED=14
export PYTHONHASHSEED=$SEED

export WANDB_PROJECT="fallacy_detection"

TRAIN_FILE="dataset/detection/train.csv"
VAL_FILE="dataset/detection/dev.csv"
TEST_FILE="dataset/detection/test.csv"
MODES=("zero_shot" "few_shot" "fine_tune")

MODELS=(
"mistralai/Mistral-7B-Instruct-v0.3"
#"openai/gpt-oss-20b"
#"meta-llama/Llama-3.1-8B-Instruct"
#"Qwen/Qwen3.5-9B-Base"
)

N_GPUS=8
grad_acc=4

BATCH_SIZES=(2 4 8)
LEARNING_RATES=(1e-4 2e-4 5e-4 1e-3)
EPOCHS=(1 2 3)

# Fixed defaults used for zero_shot / few_shot (no training, so these don't matter)
DEFAULT_BATCH=8
DEFAULT_LR=1e-4
DEFAULT_EPOCH=1

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    MODEL_NAME=$(basename "$model")

    if [[ "$mode" == "zero_shot" || "$mode" == "few_shot" ]]; then
      # Inference-only: skip hyperparameter sweep, use fixed defaults
      CONF_ID="mode${mode}"
      OUTPUT_DIR="output_no_rag/${MODEL_NAME}/${CONF_ID}"
      mkdir -p "$OUTPUT_DIR"

      python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
          --node_rank=${SLURM_PROCID} \
          --master_addr=${MASTER_ADDR} \
          --master_port=${MASTER_PORT} \
          --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=c10d \
          src/detection/run_detection_llm.py \
          --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \
          --cache_dir "./cache/" \
          --train_file "$TRAIN_FILE" \
          --validation_file "$VAL_FILE" \
          --test_file "$TEST_FILE" \
          --text_column_name "sentence" \
          --speech_turn_column_name "speech_turn" \
          --label_column_name "fallacy_label" \
          --mode "$mode" \
          --load_in_4bit \
          --output_dir "$OUTPUT_DIR" \
          --per_device_eval_batch_size "$DEFAULT_BATCH" \
          --run_name "${MODEL_NAME}-mode${mode}" \
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
      # fine_tune: sweep over all hyperparameters
      for batch in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for epoch in "${EPOCHS[@]}"; do
            CONF_ID="mode${mode}_bs$((batch*grad_acc))_lr${lr}_ep${epoch}"
            OUTPUT_DIR="output_no_rag/${MODEL_NAME}/${CONF_ID}"
            mkdir -p "$OUTPUT_DIR"

            python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
                --node_rank=${SLURM_PROCID} \
                --master_addr=${MASTER_ADDR} \
                --master_port=${MASTER_PORT} \
                --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=c10d \
                src/detection/run_detection_llm.py \
                --model_name_or_path "/lustre/fsmisc/dataset/HuggingFace_Models/$model" \
                --cache_dir "./cache/" \
                --train_file "$TRAIN_FILE" \
                --validation_file "$VAL_FILE" \
                --test_file "$TEST_FILE" \
                --text_column_name "sentence" \
                --speech_turn_column_name "speech_turn" \
                --label_column_name "fallacy_label" \
                --mode "$mode" \
                --load_in_4bit \
                --output_dir "$OUTPUT_DIR" \
                --per_device_train_batch_size "$batch" \
                --per_device_eval_batch_size "$batch" \
                --gradient_accumulation_steps $grad_acc \
                --num_train_epochs "$epoch" \
                --learning_rate "$lr" \
                --logging_steps 1000 \
                --run_name "${MODEL_NAME}-bs${batch}-lr${lr}-ep${epoch}-mode${mode}" \
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