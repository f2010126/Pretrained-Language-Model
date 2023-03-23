#!/bin/bash -l
#SBATCH --job-name=TinyGenDistill
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:04:00
#SBATCH --signal=USR1@30
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/tinyjob.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/tinyjob.error


echo 'Activate Environment'
source ~/tinybert_nlp/bin/activate

echo 'Run Generate Distill Data'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
srun python3

echo 'End of Script'