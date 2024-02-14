# Add an introduction to the file as a multi-line comment
"""
Starts the BOHB optimization process. This is the main file that is run on the master node. Each worker is run as a process but will access
GPU resources via the Ray cluster. The master node will also access the Ray cluster to submit the workers and to get the results from the workers.
"""

import subprocess
import argparse
import os
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Executer for BoHB')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=5)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=2)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.',
                        default='UsingRay')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.',default='eth0')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.',default='ddp_debug')
    parser.add_argument('--task', type=str, help='Which task to run on.',default='cardiff_multi_sentiment')
    parser.add_argument('--eta', type=int, help='Eta value for BOHB',default=2)
    parser.add_argument('--num_gpu', type=int, help='Number of GPUs to use',default=8)
    parser.add_argument('--prev_run', type=str, help='Previous run id, if any.',default='None')

    args = parser.parse_args()

	# where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)

    # Entry point here.
    # open a file in the working directory to signal that we are ready

    m_op=open(os.path.join(working_dir,"Master_output.txt"), "w+")
    m_debugr=open(os.path.join(working_dir,"Master_debug.txt"), "w+")
    master_command= 'python3 bohb_ray_cluster.py --n_iterations {}'.format(args.n_iterations) + \
        ' --n_workers {}'.format(args.n_workers) + ' --min_budget {}'.format(args.min_budget) + \
        ' --max_budget {}'.format(args.max_budget) + ' --run_id {}'.format(args.run_id) + \
        ' --nic_name {}'.format(args.nic_name) + ' --shared_directory {}'.format(args.shared_directory) +\
         ' --task {}'.format(args.task) + ' --eta {}'.format(args.eta) + ' --num_gpu {}'.format(args.num_gpu) + \
        ' --previous_run {}'.format(args.prev_run)
    worker_command= master_command + ' --worker'

    master_proc = subprocess.Popen(master_command, shell=True,stdout = m_op, stderr = m_debugr)
    proc_list=[master_proc]
    print('Started master')
    time.sleep(45) 
    # run for n-1 workers and wait 5 seconds
    for i in range(args.n_workers-1):
        # i+1 because master is 0
        w_op=open(os.path.join(working_dir,f"Worker_{i+1}_output.txt"), "w+")
        w_debug=open(os.path.join(working_dir,f"Worker_{i+1}_debug.txt"), "w+")
        worker_proc = subprocess.Popen(worker_command, shell=True,stdout = w_op, stderr = w_debug)
        proc_list.append(worker_proc)
        print('Started worker {}'.format(i+1))
        time.sleep(5) 

    # shell=True executes the program in a new shell if only kept true. Need it
    # w_debug=open("Worker_debug.txt", "w+")
    # w = subprocess.Popen("python3 hp_cluster.py --worker", shell=True,stderr = w_debug)
        
    print("Waiting for processes to finish...")
    exit_codes = [p.wait() for p in proc_list]

	

