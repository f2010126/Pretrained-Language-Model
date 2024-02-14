"""
Example 1 - Local and Sequential
================================

"""
import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB


import numpy
import time
import ConfigSpace as CS
from hpbandster.core.worker import Worker
import os

class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = numpy.clip(config['x'] + numpy.random.randn()/budget, config['x']/2, 1.5*config['x'])
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })
    
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x1', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x2', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x4', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x5', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x6', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x7', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x8', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x9', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x10', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x11', lower=0, upper=1))
        return(config_space)



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=3)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=49)
parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.',default='ddp_debug')
parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.',
                        default='LocalExample1')
parser.add_argument('--eta', type=int,default=2, help='Configuration of the Hyperband parameter eta')


args=parser.parse_args()

    # where all the run artifacts are kept
working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
os.makedirs(working_dir, exist_ok=True)

# ensure the file is empty, init the config and results json
result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)


# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1',run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'example1', nameserver='127.0.0.1',
              min_budget=args.min_budget, max_budget=args.max_budget,
              result_logger=result_logger, eta=args.eta,
           )
res = bohb.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))