#!/bin/bash -l
#SBATCH --job-name=RayCluster
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced

### Modify this according to your Ray workload.
# number of nodes, Ray will find and manage all resources, where each node runes one worker instance or the main.
#SBATCH --nodes=2

# So each ray instance gets the gpus needed, here 2 gpus per node?
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4 # I only want 2 gpus per node right now var will be SLURM_GPUS_ON_NODE
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/ray_clusterduck.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/ray_clusterduck.error

# when running on server, do ray start --head

python slurm-launch.py --exp-name test-ray --command "python examples/mnist_pytorch_trainable.py" --num-nodes 2 --num-gpus 4 --partition mlhiwidlc_gpu-rtx2080-advanced
