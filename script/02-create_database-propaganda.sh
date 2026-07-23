#!/bin/bash
#SBATCH --job-name=create-db-propaganda
#SBATCH --output=logs/create-db-propaganda_%j.out
#SBATCH --error=logs/create-db-propaganda_%j.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

TEMPORAL_BOOLS=(True False)

DENSE_RETRIEVERS=(
    "BAAI/bge-large-en-v1.5"
    "ddore14/Sentence-RooseBERT"
    "sentence-transformers/all-mpnet-base-v2"
)
DENSE_SEARCH_TYPES=(similarity hybrid)
QUERY_MODES=(fallacy_relations relations_labeled)

FAILED_RUNS=()

run() {
    echo "------------------------------------------------------------------------"
    echo "python $*"
    echo "------------------------------------------------------------------------"
    if ! python "$@"; then
        echo "FAILED: python $*"
        FAILED_RUNS+=("python $*")
    fi
}

SCRIPT_PATH="datasets/propaganda-dataset/detection/create_db.py"

for TEMPORAL_BOOL in "${TEMPORAL_BOOLS[@]}"; do
  if [[ "$TEMPORAL_BOOL" == "True" ]]; then
      TEMPORAL_FLAG="--temporal-rag"
  else
      TEMPORAL_FLAG=""
  fi

  echo "========================================================================"
  echo "TASK=detection  TEMPORAL_RAG=$TEMPORAL_BOOL"
  echo "========================================================================"

  # --- No-RAG baseline: sentence mode, no retriever. --temporal-rag asserts a
  # retriever is set, so this combination only exists without it. ---
  if [[ "$TEMPORAL_BOOL" == "False" ]]; then
      run "$SCRIPT_PATH" --mode sentence
  fi

  # --- Dense retrievers x every dense search-type.
  # (bm25 is only valid paired with --search-type hybrid, covered below.)
  # similarity never reranks; hybrid always reranks. ---
  for RETRIEVER in "${DENSE_RETRIEVERS[@]}"; do
    for SEARCH_TYPE in "${DENSE_SEARCH_TYPES[@]}"; do
      if [[ "$SEARCH_TYPE" == "hybrid" ]]; then
          RERANK="--rerank"
      else
          RERANK=""
      fi

      run "$SCRIPT_PATH" --mode sentence --retriever "$RETRIEVER" --search-type "$SEARCH_TYPE" \
          --fetch-k 30 --top-k 5 $RERANK $TEMPORAL_FLAG

      for QUERY_MODE in "${QUERY_MODES[@]}"; do
          run "$SCRIPT_PATH" --mode relations --retriever "$RETRIEVER" --search-type "$SEARCH_TYPE" \
              --fetch-k 30 --top-k 5 $RERANK $TEMPORAL_FLAG --query-mode "$QUERY_MODE"
      done
    done
  done
done

echo "========================================================================"
echo "Done. ${#FAILED_RUNS[@]} run(s) failed."
for f in "${FAILED_RUNS[@]}"; do
    echo "  FAILED: $f"
done
echo "========================================================================"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    exit 1
fi
