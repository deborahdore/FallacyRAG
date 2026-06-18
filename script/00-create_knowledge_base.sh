#!/bin/bash
#SBATCH --job-name=create_knowledge_base
#SBATCH --output=logs/create_knowledge_base.out
#SBATCH --error=logs/create_knowledge_base.out

set -e
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python src/create_knowledge_base.py