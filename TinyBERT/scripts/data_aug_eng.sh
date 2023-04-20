#!/bin/bash -l
#SBATCH --job-name=EngDataAug
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/eng_data_aug.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/eng_data_aug.error

echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
source ~/tinybert_nlp/bin/activate

echo 'Run Data Augmentation'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
echo 'SST-2 Augmentation'
python3 data_augmentation.py --pretrained_bert_model 'models/bert-base-cased' --glove_embs './data/data_augmentation/glove.6B.300d.txt' --glue_dir 'data/data_augmentation/glue_data' --task_name 'SST-2'
echo 'CoLA Augmentation'
python3 data_augmentation.py --pretrained_bert_model 'models/bert-base-cased' --glove_embs './data/data_augmentation/glove.6B.300d.txt' --glue_dir 'data/data_augmentation/glue_data' --task_name 'CoLA'
echo 'MNLI Augmentation'
python3 data_augmentation.py --pretrained_bert_model 'models/bert-base-cased' --glove_embs './data/data_augmentation/glove.6B.300d.txt' --glue_dir 'data/data_augmentation/glue_data' --task_name 'MNLI'
echo 'QQP Augmentation'
python3 data_augmentation.py --pretrained_bert_model 'models/bert-base-cased' --glove_embs './data/data_augmentation/glove.6B.300d.txt' --glue_dir 'data/data_augmentation/glue_data' --task_name 'QQP'
echo 'RTE Augmentation'
python3 data_augmentation.py --pretrained_bert_model 'models/bert-base-cased' --glove_embs './data/data_augmentation/glove.6B.300d.txt' --glue_dir 'data/data_augmentation/glue_data' --task_name 'RTE'
echo 'QNLI Augmentation'
python3 data_augmentation.py --pretrained_bert_model 'models/bert-base-cased' --glove_embs './data/data_augmentation/glove.6B.300d.txt' --glue_dir 'data/data_augmentation/glue_data' --task_name 'QNLI'
echo 'End of Script'