#!/bin/bash -l
#SBATCH --job-name=UpScale
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced

### Modify this according to your Ray workload.
# number of nodes, Ray will find and manage all resources, where each node runes one worker instance or the main.
#SBATCH --nodes=2

# So each ray instance gets the gpus needed, here 2 gpus per node?

#SBATCH --gres=gpu:4 # I only want 4 gpus per node right now var will be SLURM_GPUS_ON_NODE
#SBATCH --cpus-per-task=20
#SBATCH --time=00:5:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/hp_clusterduck-%A-%a.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/hp_clusterduck-%A-%a.error
#SBATCH --array=1-2

echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test

echo  'Code will take care of starting the master and workers'

if [ $SLURM_ARRAY_TASK_ID -eq 1 ];
   then python3 hp_cluster.py --run_id $SLURM_JOB_NAME --nic_name eth0 --shared_directory ./BOHB_Results --task_id $SLURM_ARRAY_TASK_ID
else
   python3 hp_cluster.py --run_id $SLURM_JOB_NAME --nic_name eth0  --shared_directory ./BOHB_Results --task_id $SLURM_ARRAY_TASK_ID --worker
fi

echo 'End of Script'

