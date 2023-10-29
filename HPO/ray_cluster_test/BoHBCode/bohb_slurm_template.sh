#!/bin/bash -l

${PARTITION_OPTION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_NAME}.log
${GIVEN_NODE}

### This script works for any number of nodes
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gres=gpu:${NUM_GPUS_PER_NODE}
#SBATCH --cpus-per-task=20

#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/${JOB_NAME}_Main.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/${JOB_NAME}_Main.error

#SBATCH --time=RUN_FORREST_RUN


echo "Activate environment for Job ID $SLURM_JOB_ID"

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=False
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export HF_DATASETS_CACHE="$(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/HF_Cache"
source ~/tinybert_nlp/bin/activate
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

echo  'Code will take care of starting the master and workers'
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
python3 hpband_parallel.py --run_id ${SLURM_JOB_NAME} --nic_name eth0 --shared_directory datasetruns --task-name DATASET_TO_OPTIMSE --n_iterations NUMMER_TRIALS --n_workers NUM_WORKERS &

echo " Wait 60s before STARTING WORKERS"
sleep 60

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
  python3 hpband_parallel.py --run_id ${SLURM_JOB_NAME} --nic_name eth0 --shared_directory datasetruns --task-name DATASET_TO_OPTIMSE --n_iterations NUMMER_TRIALS --n_workers NUM_WORKERS --worker &
  sleep 20
done
wait
echo 'End of Script'

