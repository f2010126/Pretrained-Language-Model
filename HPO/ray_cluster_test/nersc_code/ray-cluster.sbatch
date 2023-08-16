#!/bin/bash

#SBATCH --job-name=RayCluster
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced

#SBATCH --time=00:10:00

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=2

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4 # for now

#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/${JOB_NAME}.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/${JOB_NAME}.error

echo "Activate environment for Job ID $SLURM_JOB_ID"
echo  "Print Environment Variables"
printenv

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/nersc_code


head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi
port=6379

echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"
srun --nodes=1 --ntasks=1 -w $head_node start-head.sh $head_node_ip &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $head_node_ip:$port &
sleep 5
##############################################################################################

#### call your code below
python examples/mnist_pytorch_trainable.py --cuda

exit