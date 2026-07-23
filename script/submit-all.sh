#!/bin/bash
#!/bin/bash
#SBATCH --job-name=submit_all
#SBATCH --output=logs/submit_all.out
#SBATCH --error=logs/submit_all.out

#SBATCH --time=72:00:00
#SBATCH --account=marianne
#SBATCH --partition=cpucourt

#SBATCH --mail-user=deborah.dore@inria.fr
#SBATCH --mail-type=FAIL

module purge
module load miniconda
conda activate rags

CHUNK=199
TOTAL=240   # set to your actual total task count

task="detection"

for ((start=199; start<TOTAL; start+=CHUNK)); do
  end=$((start+CHUNK-1))
  (( end >= TOTAL )) && end=$((TOTAL-1))
  echo "Submitting ${start}-${end}, waiting for completion..."
  sbatch --wait --array=${start}-${end}%4 "$task.sh"
  echo "Batch ${start}-${end} done"
done


task="classification"

for ((start=0; start<TOTAL; start+=CHUNK)); do
  end=$((start+CHUNK-1))
  (( end >= TOTAL )) && end=$((TOTAL-1))
  echo "Submitting ${start}-${end}, waiting for completion..."
  sbatch --wait --array=${start}-${end}%4 "$task.sh"
  echo "Batch ${start}-${end} done"
done
