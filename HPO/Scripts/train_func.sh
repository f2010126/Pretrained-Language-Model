#!/bin/bash -l
#SBATCH --job-name=HPOTrainingFunc
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/sst2_bert.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/sst2_bert.error


echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO
echo 'Run HPO Training'

srun python3 training.py --learning_rate 2e-05 --num_train_epochs 3 \
--eval_batch_size_gpu 8 --optimizer_name Adam --scheduler_name 'linear' \
--project_name PyLight --run_name SST2_BERT_3

echo 'End of Script'