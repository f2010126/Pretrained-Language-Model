#!/bin/bash -l
#SBATCH --job-name=UpScale
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced

### Modify this according to your Ray workload.
# number of nodes, Ray will find and manage all resources, where each node runes one worker instance or the main.
#SBATCH --nodes=2

# So each ray instance gets the gpus needed, here 2 gpus per node?

#SBATCH --gres=gpu:4 # I only want 4 gpus per node right now var will be SLURM_GPUS_ON_NODE
#SBATCH --cpus-per-task=20
#SBATCH --time=00:5:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/hp_clusterduck-%A-%a.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/hp_clusterduck-%A-%a.error
#SBATCH --array=1-2

echo "Activate environment for Job ID $SLURM_JOB_ID"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

echo  'Code will take care of starting the master and workers'
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
  python3 hpband_parallel.py --run_id 'jobname' --nic_name eth0 --shared_directory ./bohb_runs --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" python3 hpband_parallel.py --run_id 'jobname' --nic_name eth0  --shared_directory ./bohb_runs --worker --block &
  sleep 5
done









if [ $SLURM_ARRAY_TASK_ID -eq 1 ];
   then python3 hp_cluster.py --run_id $SLURM_JOB_NAME --nic_name eth0 --shared_directory ./BOHB_Results --task_id $SLURM_ARRAY_TASK_ID
else
   python3 hp_cluster.py --run_id $SLURM_JOB_NAME --nic_name eth0  --shared_directory ./BOHB_Results --task_id $SLURM_ARRAY_TASK_ID --worker
fi

echo 'End of Script'

