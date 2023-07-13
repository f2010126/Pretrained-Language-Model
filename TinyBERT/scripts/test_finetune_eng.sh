#!/bin/bash -l
#SBATCH --job-name=TestFineTuneFinalEng
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:50:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/test_eng_fine_tune.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/test_eng_fine_tune.error

echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate

cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
GPU=$(nvidia-smi  -L | wc -l)
echo 'Start of Script with GPU: '$GPU

echo 'Run Task specific Distillation, intermediate layers'
srun torchrun --standalone --nproc_per_node=gpu task_distill.py --teacher_model 'models/bert-base-uncased-sst2' \
--student_model 'models/GeneralDistilledModels/TinyBERT_General_6L_768D' \
--data_dir 'data/data_augmentation/glue_data/SST-2' --task_name 'SST-2' \
--output_dir 'models/InterMedDistill/TestQEngTinyBert' --max_seq_length 128 --train_batch_size 32 \
--num_train_epochs 20 --aug_train --do_lower_case


echo 'Run Task specific Distillation, Prediction layers'
srun torchrun --standalone --nproc_per_node=gpu task_distill.py --pred_distill \
--teacher_model 'models/bert-base-uncased-sst2' \
--student_model 'models/InterMedDistill/TestQEngTinyBert' \
--data_dir 'data/data_augmentation/glue_data/SST-2' \
--task_name 'SST-2' --output_dir 'models/FineTunedModels/TestQTinyBERTEng' \
--aug_train --do_lower_case --learning_rate 3e-5 --num_train_epochs  3 \
--eval_step 100 --max_seq_length 128 --train_batch_size 32


echo 'Evaluate'
srun torchrun --standalone --nproc_per_node=gpu task_distill.py --do_eval \
--student_model 'models/FineTunedModels/TestQTinyBERTEng' \
--data_dir 'data/data_augmentation/glue_data/SST-2' --task_name 'SST-2' \
--output_dir 'models/FineTunedModels/EvaluateTestQTinyBERTEng' \
--do_lower_case --eval_batch_size 32 --max_seq_length 128


echo 'End of Script'


