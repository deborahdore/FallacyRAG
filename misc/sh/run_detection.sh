#!/bin/bash
#SBATCH --job-name=fallacy_detection_no_rags
#SBATCH --output=logs/fallacy_detection_no_rags.out
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --account=marianne
#SBATCH --gpus=h100:1
#SBATCH --partition=gpu

module purge
module load cuda/13.0
module load miniconda/25.1.1
conda activate rags

set -e
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface"

MODELS=("google-bert/bert-base-uncased"
        "google-bert/bert-base-cased"
        "FacebookAI/roberta-base"
        "microsoft/deberta-v3-base"
        "ddore14/RooseBERT-cont-uncased"
        "ddore14/RooseBERT-cont-cased"
        "ddore14/RooseBERT-scr-uncased"
        "ddore14/RooseBERT-scr-cased"
        "ddore14/DeRooseBERTa"
        "answerdotai/ModernBERT-base"
        "allenai/longformer-base-4096"
        )
BATCH_SIZES=(8 16 32)
LEARNING_RATES=(2e-5 3e-5 5e-5 2e-4 3e-4 5e-4 2e-3 3e-3 5e-3)
EPOCHS=(2 3 4)

for model in "${MODELS[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for epoch in "${EPOCHS[@]}"; do

        run_name="${model##*/}-epochs=$epoch-batch=$batch-lr=$lr"
        output_dir="output/$run_name/"

        python src/detection/run_detection_bio_tags.py \
                  --model_name_or_path "$model" \
                  --cache_dir "$output_dir/cache/" \
                  --task_name "ner" \
                  --train_file "dataset/detection/dev.json" \
                  --validation_file "dataset/detection/dev.json" \
                  --test_file "dataset/detection/test.json" \
                  --text_column_name "tokens" \
                  --label_column_name "ner_tags" \
                  --preprocessing_num_workers 8 \
                  --output_dir $output_dir \
                  --per_device_train_batch_size "$batch" \
                  --per_device_eval_batch_size "$batch" \
                  --num_train_epochs "$epoch" \
                  --learning_rate "$lr" \
                  --logging_steps 1000 \
                  --run_name $run_name \
                  --eval_strategy "epoch" \
                  --eval_on_start True \
                  --save_strategy "epoch" \
                  --save_total_limit 1 \
                  --seed 42 \
                  --data_seed 42 \
                  --dataloader_num_workers 8 \
                  --dataloader_prefetch_factor 2 \
                  --remove_unused_columns True \
                  --do_train True \
                  --do_eval True \
                  --do_predict True
      done
    done
  done
done