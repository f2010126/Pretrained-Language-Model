import logging
import argparse
import pickle
import time
import os
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker


logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize the metamodel')
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
    parser.add_argument('--n_workers',    type=int,   help='Number of workers to run in parallel.',            default=1)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')

    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.',
                        default='en0')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.', default='ddp_debug')
    parser.add_argument('--previous_run', type=str, default=None,
                        help='Path to the directory of the previous run. Prev run is assumed to be in the same '
                             'working dir as current')

    
    
    args = parser.parse_args()


    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)