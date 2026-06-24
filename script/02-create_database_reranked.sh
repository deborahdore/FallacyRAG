#!/bin/bash
#SBATCH --job-name=create_rag_dataset
#SBATCH --output=logs/create_rag_dataset_%j.out
#SBATCH --error=logs/create_rag_dataset_%j.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

TEMPORAL_BOOL=False

if [[ "$TEMPORAL_BOOL" == "True" ]]; then
    TEMPORAL_FLAG="--temporal-rag"
else
    TEMPORAL_FLAG=""
fi

retrievers=(
    "BAAI/bge-large-en-v1.5"
    "ddore14/Sentence-RooseBERT"
    "sentence-transformers/all-mpnet-base-v2"
)

summarization_models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3.5-9B-Base"
)

for retriever in "${retrievers[@]}"; do
    echo "========================================================================"
    echo "Hybrid RAG enrichment"
    echo "Retriever: $retriever"
    echo "========================================================================"

    python dataset/detection/create_db.py --mode sentence --retriever "$retriever"\
      --search-type "hybrid" \
      --rerank \
      --fetch-k 30 \
      --top-k 3 \
      $TEMPORAL_FLAG

#    for model in "${summarization_models[@]}"; do
#        python dataset/detection/create_db.py --mode sentence --retriever "$retriever" \
#        --search-type "hybrid" \
#        --rerank \
#        --fetch-k 30 \
#        --top-k 5 \
#        --summarization-model "$model" \
#        $TEMPORAL_FLAG
#    done
done