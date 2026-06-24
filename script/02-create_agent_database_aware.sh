#!/bin/bash
#SBATCH --job-name=create_agentic_dataset_aware
#SBATCH --output=logs/create_agentic_dataset_aware_%j.out
#SBATCH --error=logs/create_agentic_dataset_aware_%j.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

RETRIEVERS=(
  "BAAI/bge-large-en-v1.5"
  "ddore14/Sentence-RooseBERT"
  "sentence-transformers/all-mpnet-base-v2"
)

AGENT_MODEL="ollama:command-r"
PERSIST_DIR="./chromadb"
TOP_K=5
HISTORY_TURNS=3
TEMPORAL_BOOL=False

if [[ "$TEMPORAL_BOOL" == "True" ]]; then
    TEMPORAL_FLAG="--temporal-rag"
else
    TEMPORAL_FLAG=""
fi

ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!
sleep 30

for RETRIEVER in "${RETRIEVERS[@]}"; do
    echo "========================================================================"
    echo "Agent Aware RAG enrichment"
    echo "Retriever: $RETRIEVER"
    echo "========================================================================"

    echo "Running with retriever: $RETRIEVER"
    python dataset/detection/create_db_agents_aware.py \
        --retriever "$RETRIEVER" \
        --agent-model "$AGENT_MODEL" \
        --persist-directory "$PERSIST_DIR" \
        --top-k $TOP_K \
        --history-turns $HISTORY_TURNS \
        $TEMPORAL_FLAG
done

kill $OLLAMA_PID