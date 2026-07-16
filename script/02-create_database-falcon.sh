#!/bin/bash
#SBATCH --job-name=falcon-database
#SBATCH --output=logs/falcon-database_%j.out
#SBATCH --error=logs/falcon-database_%j.out
#SBATCH --gres=gpu:1

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# Usage: 02-create_database-falcon.sh [basic|relations|reranked]
#   basic      Basic RAG enrichment (comment mode, one retriever at a time)
#   reranked   Hybrid RAG enrichment (comment mode, hybrid+rerank)
#   relations  Relations-aware RAG enrichment (hybrid+rerank and similarity,
#              both query-modes)
# All three loop over --temporal-rag on/off, since the falcon dataset
# carries a year per row.
# ---------------------------------------------------------------------------
RUN_MODE="${1:-relations}"

TEMPORAL_BOOLS=(True False)
TASK=detection
SCRIPT="datasets/falcon-dataset/$TASK/create_db.py"

for TEMPORAL_BOOL in "${TEMPORAL_BOOLS[@]}"; do
  if [[ "$TEMPORAL_BOOL" == "True" ]]; then
      TEMPORAL_FLAG="--temporal-rag"
  else
      TEMPORAL_FLAG=""
  fi

#  summarization_models=(
#      "meta-llama/Llama-3.1-8B-Instruct"
#      "Qwen/Qwen3.5-9B-Base"
#  )

  case "$RUN_MODE" in
      basic)
          retrievers=(
              "BAAI/bge-large-en-v1.5"
              "bm25"
              "ddore14/Sentence-RooseBERT"
              "sentence-transformers/all-mpnet-base-v2"
          )

          for retriever in "${retrievers[@]}"; do
              echo "========================================================================"
              echo "Basic RAG enrichment"
              echo "Retriever: $retriever"
              echo "========================================================================"

              if [[ "$retriever" == "bm25" ]]; then
                  search_type="--search-type bm25"
              else
                  search_type="--search-type similarity"
              fi

              python "$SCRIPT" --mode comment --retriever "$retriever" $search_type $TEMPORAL_FLAG

  #            for model in "${summarization_models[@]}"; do
  #                python "$SCRIPT" --mode comment --retriever "$retriever" --summarization-model "$model" $search_type $TEMPORAL_FLAG
  #            done
          done
          ;;

      relations)
          retrievers=(
              "BAAI/bge-large-en-v1.5"
              "ddore14/Sentence-RooseBERT"
              "sentence-transformers/all-mpnet-base-v2"
          )

          for retriever in "${retrievers[@]}"; do
              echo "========================================================================"
              echo "Relations Aware RAG enrichment"
              echo "Retriever: $retriever"
              echo "========================================================================"

              python "$SCRIPT" \
                --mode "relations" \
                --retriever "$retriever"\
                --search-type "hybrid" \
                --rerank \
                --fetch-k 30 \
                --top-k 5 \
                $TEMPORAL_FLAG \
                --query-mode "relations_labeled"

              python "$SCRIPT" \
                    --mode "relations" \
                    --retriever "$retriever"\
                    --search-type "similarity" \
                    $TEMPORAL_FLAG \
                    --query-mode "relations_labeled"


              python "$SCRIPT" \
                --mode "relations" \
                --retriever "$retriever"\
                --search-type "hybrid" \
                --rerank \
                --fetch-k 30 \
                --top-k 5 \
                $TEMPORAL_FLAG \
                --query-mode "fallacy_relations"

              python "$SCRIPT" \
                    --mode "relations" \
                    --retriever "$retriever"\
                    --search-type "similarity" \
                    $TEMPORAL_FLAG \
                    --query-mode "fallacy_relations"

  #            for model in "${summarization_models[@]}"; do
  #                python "$SCRIPT" --mode relations --retriever "$retriever" \
  #                --query-mode "relations_labeled" \
  #                --search-type "hybrid" \
  #                --rerank \
  #                --fetch-k 30 \
  #                --top-k 5 \
  #                --summarization-model "$model" \
  #                $TEMPORAL_FLAG
  #            done
          done
          ;;

      reranked)
          retrievers=(
              "BAAI/bge-large-en-v1.5"
              "ddore14/Sentence-RooseBERT"
              "sentence-transformers/all-mpnet-base-v2"
          )

          for retriever in "${retrievers[@]}"; do
              echo "========================================================================"
              echo "Hybrid RAG enrichment"
              echo "Retriever: $retriever"
              echo "========================================================================"

              python "$SCRIPT" --mode comment --retriever "$retriever"\
                --search-type "hybrid" \
                --rerank \
                --fetch-k 30 \
                --top-k 3 \
                $TEMPORAL_FLAG

  #            for model in "${summarization_models[@]}"; do
  #                python "$SCRIPT" --mode comment --retriever "$retriever" \
  #                --search-type "hybrid" \
  #                --rerank \
  #                --fetch-k 30 \
  #                --top-k 5 \
  #                --summarization-model "$model" \
  #                $TEMPORAL_FLAG
  #            done
          done
          ;;

      *)
          echo "Unknown mode: $RUN_MODE"
          echo "Usage: $0 [basic|relations|reranked]"
          exit 1
          ;;
  esac
done
