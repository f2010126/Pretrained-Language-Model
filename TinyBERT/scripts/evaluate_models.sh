#!/bin/bash -l
#SBATCH --job-name=EvaluateModels
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:50:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/evaluate_model.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/evaluate_model.error



echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate

echo 'Evaluate Model on English GlUE Tasks'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
GPU=$(nvidia-smi  -L | wc -l)
echo 'Start of Script with GPU: '$GPU
accelerate launch --multi_gpu --gpu_ids "all" --num_processes $GPU evaluate_glue_task.py --model_name_or_path 'OpenVINO/bert-base-uncased-sst2-int8-unstructured80'
accelerate launch --multi_gpu --gpu_ids "all" --num_processes $GPU evaluate_glue_task.py --model_name_or_path 'JeremiahZ/bert-base-uncased-sst2'
echo 'End of Script'