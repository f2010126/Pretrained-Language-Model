#!/bin/bash -l
#SBATCH --job-name=DDG
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced

#SBATCH -c 1 # number of cores
#SBATCH -a 0-1 # array size
#SBATCH --time=00:5:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/hp_parallel-%A-%a.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/hp_parallel-%A-%a.error


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test


echo "run script"

bohb_id=$(($SLURM_ARRAY_TASK_ID+0))
python3 -u hpband_parallel.py --bohb_id $bohb_id --n_workers 1

echo 'End of Script'