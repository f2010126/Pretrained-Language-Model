#!/bin/bash -l

${PARTITION_OPTION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/${JOB_NAME}.log
${GIVEN_NODE}

### This script works for any number of nodes
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gres=gpu:${NUM_GPUS_PER_NODE}
#SBATCH --cpus-per-task=32

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
source ~/ray_env/bin/activate
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

# ray setup
echo "Ray setup"
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --redis-password="$redis_password" --block &
  sleep 5
done

echo "Ray setup complete"

# run the script. A runner function will start the master and workers with popen. Debug outputs will be saved in respective log files. 
python3 bohb_runner.py --max_budget MAX_BUDGET --n_iterations NUM_ITER \
--n_workers NUM_WORKER --run_id RUN_ID --shared_directory datasetruns \
--task DATASET_TO_OPTIMSE --eta bohb_eta --num_gpu GPU_WORKERS --prev_run PREV_RUN --aug AUGMENTATION
wait
echo 'End of Script'



