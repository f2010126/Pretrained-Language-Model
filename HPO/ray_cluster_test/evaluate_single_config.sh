#!/bin/bash -l

#SBATCH --job-name=EvaluateOneConfig
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=14:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/LongEval.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/LongEval.error


export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=False
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
source ~/tinybert_nlp/bin/activate
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test

echo 'Run Evaluation for Best Config'
python3 evaluate_single_config.py
echo 'End of Script'