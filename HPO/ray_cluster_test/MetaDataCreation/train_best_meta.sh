#!/bin/bash -l
#SBATCH --job-name=TrainBestMeta
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/TrainBest.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/TrainBest.error


cd /work/dlclarge1/dsengupt-zap_hpo_og/TinyBert/HPO/ray_cluster_test/MetaDataCreation
source metatest_env/bin/activate

echo 'Run Surrogate Training for Regression'
python metamodel_train.py --batch_size 204 --seed 42 --loss_func regression --epochs 500 --patience 10

echo 'Run Surrogate Training for BPR'
python metamodel_train.py --batch_size 204 --seed 42 --loss_func bpr --epochs 500 --patience 10

echo 'Run Surrogate Training for Hinge'
python metamodel_train.py --batch_size 204 --seed 42 --loss_func hingeloss --epochs 500 --patience 10

echo 'End of Script'