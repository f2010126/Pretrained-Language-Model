#!/bin/bash -l
#SBATCH --job-name=AdapterTimeCheck
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/adapter_quick.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/adapter_quick.error


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate

echo 'Run Adapter Fine Tune'
cd $(ws_find zap_hpo_og)/TinyBert/Adapters
echo 'Run Adapter Tune'

echo 'Run Adapter Training'
srun torchrun --standalone --nproc_per_node=gpu adapter_quick.py --num_train_epochs 6

echo 'End of Script'