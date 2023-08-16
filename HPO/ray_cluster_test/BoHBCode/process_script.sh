#!/bin/bash -l
#SBATCH --job-name=3x4x2xDDG_run
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --nodes=3 # number of nodes
#SBATCH --gres=gpu:4 # I only want 4 gpus per node right now var will be SLURM_GPUS_ON_NODE
#SBATCH -a 0-2 # array size
#SBATCH --time=00:15:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/node-%x-%a.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/node-%x-%a.error
#SBATCH --wait-all-nodes=1 # Do not begin execution until all nodes are ready for use.


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

echo "Task ID: $SLURM_ARRAY_TASK_ID"
if [ $SLURM_ARRAY_TASK_ID -eq 0 ];
   then srun python3 node_worker.py --run_id $SLURM_JOB_NAME --nic_name eth0 --shared_directory ./BOHB_Results/$SLURM_JOB_NAME --task_id $SLURM_ARRAY_TASK_ID
else
    srun python3 node_worker.py --run_id $SLURM_JOB_NAME --nic_name eth0  --shared_directory ./BOHB_Results/$SLURM_JOB_NAME --task_id $SLURM_ARRAY_TASK_ID --worker
fi

echo 'End of Script'





