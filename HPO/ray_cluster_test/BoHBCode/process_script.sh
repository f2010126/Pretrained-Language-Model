#!/bin/bash -l
#SBATCH --job-name=DDG
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --nodes=2 # number of nodes
#SBATCH --gres=gpu:4 # I only want 4 gpus per node right now var will be SLURM_GPUS_ON_NODE
#SBATCH -a 0-1 # array size
#SBATCH --time=00:5:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/node-%A-%a.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/node-%A-%a.error


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

if [ $SLURM_ARRAY_TASK_ID -eq 0 ];
   then srun python3 node_worker.py --run_id $SLURM_JOB_NAME --nic_name eth0 --shared_directory ./BOHB_Results --task_id $SLURM_ARRAY_TASK_ID
else
   srun python3 node_worker.py --run_id $SLURM_JOB_NAME --nic_name eth0  --shared_directory ./BOHB_Results --task_id $SLURM_ARRAY_TASK_ID --worker
fi

echo 'End of Script'





