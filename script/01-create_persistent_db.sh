#!/bin/bash
#SBATCH --job-name=create_persistent_db
#SBATCH --output=logs/create_persistent_db.out
#SBATCH --error=logs/create_persistent_db.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

RETRIEVERS=(
    "sentence-transformers/all-mpnet-base-v2"
    "anonymous/Sentence-RooseBERT"
    "BAAI/bge-large-en-v1.5"
)

BATCH_SIZE=1024

for retriever in "${RETRIEVERS[@]}"; do
        echo "===================================================="
        echo "Retriever:  $retriever"
        echo "===================================================="

        python -m src.create_persistent_db \
            --retriever   "$retriever"  \
            --batch-size  "$BATCH_SIZE"
done
