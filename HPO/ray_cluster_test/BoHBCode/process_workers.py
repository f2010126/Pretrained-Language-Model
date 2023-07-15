import logging
logging.basicConfig(level=logging.INFO)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker

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
        return(config_space)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example 3 - Local and Parallel Execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=5)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=20)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=3)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

    args = parser.parse_args()

    if args.worker:
        w = MyWorker(sleep_interval=0.5, nameserver='127.0.0.1', run_id='example3')
        w.run(background=True)
        exit(0)

    # Start a nameserver (see example_1)
    NS = hpns.NameServer(run_id='example3', host='127.0.0.1', port=None)
    NS.start()

    # Run an optimizer (see example_2)
    bohb = BOHB(configspace=MyWorker.get_configspace(),
                run_id='example3',
                min_budget=args.min_budget, max_budget=args.max_budget
                )
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

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

    all_runs = res.get_all_runs()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (
                sum([r.budget for r in all_runs]) / args.max_budget))
    print('Total budget corresponds to %.1f full function evaluations.' % (
                sum([r.budget for r in all_runs]) / args.max_budget))
    print('The run took  %.1f seconds to complete.' % (
                all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))