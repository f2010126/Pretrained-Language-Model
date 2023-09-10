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

#SBATCH --time=00:05:00


echo "Activate environment for Job ID $SLURM_JOB_ID"
echo  "Print Environment Variables"

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

echo  'Code will take care of starting the master and workers'
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
--error="/work/dlclarge1/dsengupt/logs/${SLURM_JOB_NAME}_${node_1}.err" \
--output="/work/dlclarge1/dsengupt/logs/${SLURM_JOB_NAME}_${node_1}.out" \
python3 hpband_parallel.py --run_id ${SLURM_JOB_NAME} --nic_name eth0 --shared_directory ./datasetruns

echo " Wait 30s before STARTING WORKERS"
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
  --error="/work/dlclarge1/dsengupt/logs/${SLURM_JOB_NAME}_${node_i}.err" \
  --output="/work/dlclarge1/dsengupt/logs/${SLURM_JOB_NAME}_${node_i}.out" \
  python3 hpband_parallel.py --run_id ${SLURM_JOB_NAME} --nic_name eth0  --shared_directory ./datasetruns --worker
  sleep 5
done
wait
echo 'End of Script'

