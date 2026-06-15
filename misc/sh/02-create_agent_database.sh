#!/bin/bash
#SBATCH --job-name=create_agentic_dataset
#SBATCH --output=logs/create_agentic_dataset.out
#SBATCH --error=logs/create_agentic_dataset.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

RETRIEVERS=(
  "sentence-transformers/all-mpnet-base-v2"
  "ddore14/Sentence-RooseBERT"
  "BAAI/bge-large-en-v1.5"
)

AGENT_MODEL="ollama:command-r"

PERSIST_DIR="./chromadb"

SEARCH_TYPE="similarity"
TOP_K=5
FETCH_K=20

LAMBDA_MULT=0.5
SCORE_THRESHOLD=0.5

ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

sleep 30

for RETRIEVER in "${RETRIEVERS[@]}"; do
    echo "Running with retriever: $RETRIEVER"
    python dataset/detection/create_db_agents.py \
        --retriever "$RETRIEVER" \
        --agent-model "$AGENT_MODEL" \
        --persist-directory "$PERSIST_DIR" \
        --search-type "$SEARCH_TYPE" \
        --top-k $TOP_K \
        --fetch-k $FETCH_K \
        --lambda-mult $LAMBDA_MULT \
        --score-threshold $SCORE_THRESHOLD
done

kill $OLLAMA_PID