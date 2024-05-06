#!/bin/bash -l
#SBATCH --job-name=SurrogateUnHinge
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/HingeLossSuroogate.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/HingeLossSuroogate.error


cd /work/dlclarge1/dsengupt-zap_hpo_og/TinyBert/HPO/ray_cluster_test/MetaDataCreation
source metatest_env/bin/activate

echo 'Optimise Surrogate Training'

python metamodel_optimise.py --min_budget 100 --max_budget 200 --n_iterations 2 --n_workers 1 --run_id optimiseRegression1 \
--nic_name eth0 --shared_directory metaModelHPO --previous_run None --seed 42 --input_size 27 --output_size 1 \
--loss_func regression --batch_size 204
    
echo 'End of Script'