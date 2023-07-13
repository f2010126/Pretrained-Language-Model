import os
import sys
import numpy
import argparse
import ConfigSpace as CS
import yaml
from copy import deepcopy
import torch

# go to parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.join(script_dir, os.pardir)
sys.path.append(par_dir)
os.chdir(par_dir)

import random
import numpy as np
import time
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import psutil
from hpbandster.core.worker import Worker
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB

class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 1
        params['max_budget'] = 1
        params['eta'] = 2
        params['random_fraction'] = 1
        params['iterations'] = 10000

        return params

    def get_configspace(self):
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return config_space

    def get_specific_config(self, cso, default_config, budget):
        return self.get_configspace().sample_configuration()

    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):

        if torch.cuda.is_available():
            print(f'Number of devices: {torch.cuda.device_count()}')
        else:
            print('No GPU available.')

        config = self.get_specific_config()

        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        res = numpy.clip(config['x'] + numpy.random.randn() / budget, config['x'] / 2, 1.5 * config['x'])
        time.sleep(0.5)

        return ({
            'loss': float(res),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })



class BohbWorker(Worker):
    def __init__(self, id, working_dir, experiment_wrapper, *args, **kwargs):
        super(BohbWorker, self).__init__(*args, **kwargs)
        print(kwargs)

        self.id = id
        self.working_dir = working_dir
        self.experiment_wrapper = experiment_wrapper

    def compute(self, config_id, config, budget, *args, **kwargs):
        return self.experiment_wrapper.compute(self.working_dir, self.id, config_id, config, budget, *args, **kwargs)



class BohbWrapper(Master):
    def __init__(self, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 3, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):
        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        cg = BOHB(configspace=configspace,
                  min_points_in_model=min_points_in_model,
                  top_n_percent=top_n_percent,
                  num_samples=num_samples,
                  random_fraction=random_fraction,
                  bandwidth_factor=bandwidth_factor,
                  min_bandwidth=min_bandwidth
                  )

        super().__init__(config_generator=cg, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0,
                                                               self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        # number of 'SH rungs'
        s = self.max_SH_iter - 1
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns,
                                  budgets=self.budgets[(-s - 1):],
                                  config_sampler=self.config_generator.get_config,
                                  **iteration_kwargs))


def get_bohb_interface():
    addrs = psutil.net_if_addrs()
    if 'eth0' in addrs.keys():
        print('FOUND eth0 INTERFACE')
        return 'eth0'
    elif 'eno1' in addrs.keys():
        print('FOUND eno1 INTERFACE')
        return 'eno1'
    elif 'ib0' in addrs.keys():
        print('FOUND ib0 INTERFACE')
        return 'ib0'
    elif 'lo0' in addrs.keys():
        print('FOUND lo0 INTERFACE. Local on Mac')
        return 'lo0'
    else:
        print('FOUND lo INTERFACE')
        return 'lo'


def get_working_dir(run_id):
    return str(os.path.join(os.getcwd(), "results", run_id))


def run_bohb_parallel(id, run_id, bohb_workers, experiment_wrapper):
    # get bohb params
    bohb_params = experiment_wrapper.get_bohb_parameters()

    # get suitable interface (eth0 or lo)
    bohb_interface = get_bohb_interface()

    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    # every process has to lookup the hostname
    host = hpns.nic_name_to_host(bohb_interface)

    os.makedirs(working_dir, exist_ok=True)
    print(f"Id of this one: {id}")
    if int(id) % 1000 != 0:
        print('START NEW WORKER')
        time.sleep(50)
        w = BohbWorker(host=host,
                       id=id,
                       run_id=run_id,
                       working_dir=working_dir,
                       experiment_wrapper=experiment_wrapper)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=True)
        exit(0)

    print('START NEW MASTER')
    ns = hpns.NameServer(run_id=run_id,
                         host=host,
                         port=0,
                         working_directory=working_dir)
    ns_host, ns_port = ns.start()

    w = BohbWorker(host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   id=id,
                   run_id=run_id,
                   working_dir=working_dir,
                   experiment_wrapper=experiment_wrapper)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=working_dir,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=experiment_wrapper.get_configspace(),
        run_id=run_id,
        eta=bohb_params['eta'],
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=bohb_params['min_budget'],
        max_budget=bohb_params['max_budget'],
        random_fraction=bohb_params['random_fraction'],
        result_logger=result_logger)

    # res = bohb.run(n_iterations=bohb_params['iterations'])
    res = bohb.run(n_iterations=bohb_params['iterations'],
                   min_n_workers=int(bohb_workers))

    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def run_bohb_serial(run_id, experiment_wrapper):
    # get bohb parameters
    bohb_params = experiment_wrapper.get_bohb_parameters()

    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BohbWorker(nameserver="127.0.0.1",
                   id=0,
                   run_id=run_id,
                   nameserver_port=port,
                   working_dir=working_dir,
                   experiment_wrapper=experiment_wrapper)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=working_dir,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=experiment_wrapper.get_configspace(),
        run_id=run_id,
        eta=bohb_params['eta'],
        min_budget=bohb_params['min_budget'],
        max_budget=bohb_params['max_budget'],
        random_fraction=bohb_params['random_fraction'],
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger)

    res = bohb.run(n_iterations=bohb_params['iterations'])
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--bohb_id', type=int, help='ID of the Script.', default=0)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)

    args = parser.parse_args()

    print('STARTING EXPERIMENT')
    run_id='little run'
    res = run_bohb_parallel(id=args.bohb_id,
                            bohb_workers=args.n_workers,
                            run_id=run_id,
                            experiment_wrapper=ExperimentWrapper())
    print('FINISHED EXPERIMENT')