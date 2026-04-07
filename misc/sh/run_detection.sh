#!/bin/bash
#SBATCH --job-name=fallacy_detection_without_rags_%j
#SBATCH --output=logs/fallacy_detection_without_rags_%j.out

#SBATCH --time=36:00:00
#SBATCH --account=marianne
#SBATCH --gpus=h100:1
#SBATCH --partition=gpu

module purge
module load miniconda
conda activate rags
set -e

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface"

models=(
"google-bert/bert-base-uncased"
"google-bert/bert-base-cased"
"FacebookAI/roberta-base"
"microsoft/deberta-v3-base"
"ddore14/RooseBERT-scr-cased"
"ddore14/RooseBERT-scr-uncased"
"ddore14/RooseBERT-cont-cased"
"ddore14/RooseBERT-cont-uncased"
)

batch_sizes=(8 16 32)
lrs=(2e-5 3e-5 5e-5)

epochs=15
for model in "${models[@]}"; do
  for batch_per_device in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do

      run_name="${model}-BATCH=${batch_per_device}-EPOCHS=${epochs}-LR=${lr}"

      python src/detection/run_detection_bio_tags.py \
          --model_name_or_path "$model" \
          --config_name "$model" \
          --tokenizer_name "$model" \
          --cache_dir "cache/" \
          --task_name "ner" \
          --train_file "dataset/detection/bio_tags/train.json" \
          --validation_file "dataset/detection/bio_tags/dev.json" \
          --test_file "dataset/detection/bio_tags/test.json" \
          --text_column_name "tokens" \
          --label_column_name "ner_tags" \
          --preprocessing_num_workers 6 \
          --label_all_tokens True \
          --output_dir "$run_name" \
          --per_device_train_batch_size "$batch_per_device" \
          --per_device_eval_batch_size "$batch_per_device" \
          --num_train_epochs "$epochs" \
          --learning_rate "$lr" \
          --logging_strategy "epochs" \
          --eval_strategy "epochs" \
          --log_level "detail" \
          --run_name "$run_name" \
          --eval_on_start True \
          --seed 42 \
          --data_seed 42 \
          --dataloader_drop_last True \
          --dataloader_num_workers 8 \
          --remove_unused_columns \
          --do_train \
          --do_eval \
          --do_predict \
          --load_best_model_at_end True \
          --metric_for_best_model "eval_f1_macro"

    done
  done
done


