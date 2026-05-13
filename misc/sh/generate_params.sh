#!/bin/bash
# Helper script to generate params.txt and manage your sweep

set -e

OUTPUT_DIR="runs/fallacy_detection"
PARAMS_FILE="${OUTPUT_DIR}/params.txt"

mkdir -p "$OUTPUT_DIR"

# Define your parameter space
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
echo "Generating parameter combinations..."
> "$PARAMS_FILE"
counter=0
for model in "${MODELS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for epoch in "${EPOCHS[@]}"; do
                counter=$((counter + 1))
                echo "$model $batch $lr $epoch" >> "$PARAMS_FILE"
            done
        done
    done
done

echo "Generated $counter parameter combinations"
echo "Total: ${#MODELS[@]} models × ${#BATCH_SIZES[@]} batch sizes × ${#LEARNING_RATES[@]} LRs × ${#EPOCHS[@]} epochs = $counter"
echo ""
echo "Parameters file: $PARAMS_FILE"
echo ""
echo "To submit the job array, run:"
echo "  sbatch fallacy_detection_array.sh"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all array jobs:"
echo "  scancel <job_id>"
echo ""
echo "To view results once complete:"
echo "  cat $OUTPUT_DIR/master_results.csv"
