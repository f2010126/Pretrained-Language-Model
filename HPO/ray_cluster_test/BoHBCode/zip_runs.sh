#!/bin/bash -l

#SBATCH --partition=mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --job-name=zip_dataset


### This script works for any number of nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/zip_dataset.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/zip_dataset.error

#SBATCH --time=5:10:00

echo "Start"
source ~/tinybert_nlp/bin/activate
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

zip -r trials_data.zip datasetruns

echo "End"
