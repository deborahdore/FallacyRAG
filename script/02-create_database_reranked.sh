#!/bin/bash
#SBATCH --job-name=create_rag_dataset
#SBATCH --output=logs/create_rag_dataset.out
#SBATCH --error=logs/create_rag_dataset.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

retrievers=(
    "BAAI/bge-large-en-v1.5"
    "bm25"
    "ddore14/Sentence-RooseBERT"
    "sentence-transformers/all-mpnet-base-v2"
)

summarization_models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3.5-9B-Base"
)

for retriever in "${retrievers[@]}"; do
    python dataset/detection/create_db.py --retriever "$retriever"  --search-type hybrid --rerank --fetch-k 30 --top-k 3

    for model in "${summarization_models[@]}"; do
        python dataset/detection/create_db.py --retriever "$retriever" \
        --search-type hybrid \
        --rerank \
        --fetch-k 30 \
        --top-k 3 \
        --summarization-model "$model"
    done
done