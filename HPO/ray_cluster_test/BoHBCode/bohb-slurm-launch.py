# bohb-slurm-launch.py
# Usage:
# python bohb-slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import time

template_file = Path(os.path.abspath(os.path.dirname(__file__))) / "bohb_slurm_template.sh"
JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
GIVEN_NODE = "${GIVEN_NODE}"
RUNTIME = "RUN_FORREST_RUN"

TASK = "DATASET_TO_OPTIMSE"
SAMPLE = "NUMMER_TRIALS"
WORKERS = "NUM_WORKERS"

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
    # Args related to the underlying python Script
    parser.add_argument("--task-name",
                        "-t", type=str, default="sentilex", help="Name of the dataset to use")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of times BOHB should sample the space")
    parser.add_argument("--runtime", type=str, default='10:10:00', help="Run the experiment for a certain time")
    parser.add_argument("--n_workers", type=int, default=2, help="Number of parallel workers to use")

    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}_trials".format(args.exp_name, args.num_trials)

    partition_option = (
        "#SBATCH --partition={}".format(args.partition) if args.partition else ""
    )
    task = "{}".format(args.task_name)
    trial_count = "{}".format(args.num_trials)
    num_workers = "{}".format(args.n_workers)

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
    text = text.replace(TASK, task)
    text = text.replace(SAMPLE, trial_count)
    text = text.replace(WORKERS, num_workers)

    # ===== Save the script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "{}.log".format(job_name)
        )
    )
    print("You can check the status of your job by 'squeue -u <username>'.")
    sys.exit(0)
    sys.exit(0)
