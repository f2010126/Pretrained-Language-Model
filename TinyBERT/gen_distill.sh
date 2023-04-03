#!/bin/bash -l
#SBATCH --job-name=TinyGenDistill
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/tinyjob.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/tinyjob.error
#SBATCH --array 0-4%1

export NCCL_DEBUG=INFO
echo 'Activate Environment'
source ~/tinybert_nlp/bin/activate
GPU=$(nvidia-smi  -L | wc -l)
echo 'Run Generate Distill Data'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
echo 'Start of Script'
srun torchrun --standalone --nproc_per_node=$GPU general_distill.py --eval_step 50
echo 'End of Script'