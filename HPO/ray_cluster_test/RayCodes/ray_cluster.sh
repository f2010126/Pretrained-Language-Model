#!/bin/bash -l
#SBATCH --job-name=RayCluster
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced

### Modify this according to your Ray workload.
# number of nodes, Ray will find and manage all resources, where each node runes one worker instance or the main.
#SBATCH --nodes=2

# So each ray instance gets the gpus needed, here 2 gpus per node?
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:8 # I only want 2 gpus per node right now var will be SLURM_GPUS_ON_NODE
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/ray_clusterduck.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/ray_clusterduck.error

# when running on server, do ray start --head
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
source ~/tinybert_nlp/bin/activate
export TOKENIZERS_PARALLELISM=False
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test

redis_password='pass'
export redis_password

ray start --head --redis-password="$redis_password"
sleep 20
python ray_lightning_DefaultTrainer.py --exp-name "clusterSmoke"

ray stop

# python bohb-slurm-launch.py --exp-name test-ray --command "python examples/mnist_pytorch_trainable.py" --num-nodes 2 --num-gpus 4 --partition mlhiwidlc_gpu-rtx2080-advanced
