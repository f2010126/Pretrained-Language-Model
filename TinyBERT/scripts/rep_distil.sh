#!/bin/bash -l
#SBATCH --job-name=RepTinyGenDistill
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=20:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/rep_30.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/rep_30.error
#SBATCH --array 0-4%1


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
source ~/tinybert_nlp/bin/activate

echo 'Run Generate Distill Data with only Representation Loss'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
GPU=$(nvidia-smi  -L | wc -l)
echo 'Start of Script with GPU: '$GPU
echo 'Train for 30 epochs'
srun torchrun --standalone --nproc_per_node=gpu general_distill.py --eval_step 2 --num_train_epochs 30 --pregenerated_data 'data/ep_30_pretraining_data' --output_dir 'rep_gen_distil' --checkpoint-name 'rep_gen_distil' --exp_name 'TinyBERT-DE-Ablations' --group_name 'general-distillation-rep' --attn_scale 0.0 --rep_scale 1.0
echo 'End of Script'