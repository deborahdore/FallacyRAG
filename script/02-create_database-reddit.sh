#!/bin/bash
#SBATCH --job-name=reddit-database
#SBATCH --output=logs/reddit-database_%j.out
#SBATCH --error=logs/reddit-database_%j.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

tasks=("detection" "classification")
retrievers=(
  "BAAI/bge-large-en-v1.5"
  "ddore14/Sentence-RooseBERT"
  "sentence-transformers/all-mpnet-base-v2"
)

for task in "${tasks[@]}"; do
  script="reddit-dataset/reddit-$task/create_db.py"

  # bm25
  python "$script" --retriever bm25 --search-type bm25

  for retriever in "${retrievers[@]}"; do
    python "$script" --retriever "$retriever" --search-type similarity
    python "$script" --retriever "$retriever" --search-type hybrid --rerank --fetch-k 30 --top-k 3
  done
done