#!/bin/bash -l
#SBATCH --job-name=DataAugDe
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=23:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/data_aug_de.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/data_aug_de.error


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate

echo 'Run Generate Distill Data'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
echo 'Run Data Augmentation'
python3 german_data_aug.py --pretrained_bert_model models/bert-base-german-uncased --glove_embs data/data_augmentation/de_glove_embeddings.txt --glue_dir data/data_augmentation/glue_data --task_name amazon_de --N 10
echo 'End of Script'





