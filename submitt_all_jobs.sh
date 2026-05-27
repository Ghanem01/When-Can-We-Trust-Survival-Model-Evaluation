#!/bin/bash
# submit_all_jobs.sh
# ============================================================
# Submit all experiments for all datasets and censorship rates
# Example usage: bash submit_all_jobs.sh rsf
# ============================================================

model=${1:-rsf}

datasets=("nacd" "flchain" "gbmlgg" "metabric" "pbc" "support")
censor_rates=(10 20 30 50 70 90)

for dataset in "${datasets[@]}"; do
  for censorship in "${censor_rates[@]}"; do
    echo "Submitting job for dataset: $dataset | Model: $model | Censorship: $censorship%"
    qsub -v dataset=$dataset,model=$model,censorship=$censorship job_template.pbs
  done
done
