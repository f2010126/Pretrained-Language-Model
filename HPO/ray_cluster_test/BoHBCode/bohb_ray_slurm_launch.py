# bohb-slurm-launch.py
# Usage:
# python bohb-slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
from re import A
import subprocess
import sys
import time
import os
from pathlib import Path
import time
import signal


template_file = Path(os.path.abspath(os.path.dirname(__file__))) / "bohb_ray_slurm_template.sh"
JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
GIVEN_NODE = "${GIVEN_NODE}"
RUNTIME = "RUN_FORREST_RUN"

TASK = "DATASET_TO_OPTIMSE"
max_budget= "MAX_BUDGET"
WORKERS = "NUM_WORKER"
N_ITR= "NUM_ITER"
E = "bohb_eta"
W_GPU = "GPU_WORKERS"
RUN_ID = "RUN_ID"

def isTimeFormat(input):
    try:
        time.strptime(input, '%H:%M:%S')
        return True
    except ValueError:
        raise ValueError("Incorrect data format, should be HH:MM:SS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
             "return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)",
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,

    )
    parser.add_argument("--runtime", type=str, default='10:10:00', 
                        help="Run the experiment for a certain time")
    # Args related to the underlying python Script
    parser.add_argument("--task-name",
                        "-t", type=str, default="sentilex", help="Name of the dataset to use")
    parser.add_argument("--n_workers", type=int, default=2, help="Number of parallel workers to use")
    parser.add_argument("--eta", type=int, default=3, help="Eta of the BOHB")
    parser.add_argument("--max_budget", type=int, default=1, help="Max budget of the BOHB")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of iterations of the BOHB")

    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}_trials".format(args.exp_name,args.task_name)

    partition_option = (
        "#SBATCH --partition={}".format(args.partition) if args.partition else ""
    )

    run_id = "{}_{}".format(args.exp_name, args.task_name)
    task = "{}".format(args.task_name)
    num_workers = "{}".format(args.n_workers)
    budget = "{}".format(args.max_budget)
    eta= "{}".format(args.eta)
    n_iter= "{}".format(args.n_iter)

    if isTimeFormat(args.runtime):
        runtime = args.runtime


    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO " "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!",
    )
    text = text.replace(RUNTIME, runtime)
    # Python Script related
    text = text.replace(RUN_ID, run_id)
    text = text.replace(TASK, task)
    text = text.replace(max_budget, budget)
    text = text.replace(N_ITR, n_iter)
    text = text.replace(E, eta)
    text = text.replace(W_GPU, num_workers)
    text = text.replace(WORKERS, num_workers)

    # ===== Save the script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    submitted_process=subprocess.Popen(["sbatch", script_file], preexec_fn=os.setsid)
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "{}.log".format(job_name)
        )
    )
    print("You can check the status of your job by 'squeue -u <username>'.")
    # os.killpg(os.getpgid(submitted_process.pid), signal.SIGTERM)  # Send the signal to all the process groups
    sys.exit(0)
    sys.exit(0)
