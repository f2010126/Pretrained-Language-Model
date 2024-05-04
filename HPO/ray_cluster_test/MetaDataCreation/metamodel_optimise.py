import logging
import argparse
import pickle
import re
import time
import numpy
import os
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import traceback
import sys
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random

from xgboost import cv

# local
from metamodel_train import TrainModel

logging.basicConfig(level=logging.INFO)

class SurrogateWorker(Worker):

    def __init__(self, *args, seed=42, inputsize=27, output=1, batch=204,loss='regression', **kwargs):
        super().__init__(*args, **kwargs)
        # seed stuff
        numpy.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.input_size=inputsize   
        self.output_size=output
        self.batch_size=batch
        self.loss_func=loss


    def compute(self, config, budget, **kwargs):
        """
        Compute the objective function
        """
        # train the model
        try:
            obj=TrainModel(input_size=self.input_size, hidden_size=64, output_size=self.output_size, 
                              epochs=int(budget), lr=config['lr'], batch_size=self.batch_size, 
                              fold_no=config['cv_fold'], loss_func=self.loss_func, seed=self.seed, config=config)
            model, ndcg1_val=obj.train()
            res=obj.test()
            # res = random.random()
        except Exception as e:
            print(f'Error in training the model: {e}')
            return({
                    'loss': -float(-1000),  # this is the a mandatory field to run hyperband
                    'info': {'run':'Failed','test':-100}  # can be used for any user-defined information - also mandatory
                })
        
        return({
                    'loss': -float(res),  # this is the a mandatory field to run hyperband
                    'info': {'run':'Sucess','test':res}  # can be used for any user-defined information - also mandatory
                })
    
    def get_configspace(seed=42):
        cs = CS.ConfigurationSpace(seed=seed)
        
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-2, log=True)
        min_lr = CSH.UniformFloatHyperparameter('min_lr', lower=1e-8, upper=1e-6, log=True)
        optimizer_type = CSH.CategoricalHyperparameter('optimizer_type', ['Adam', 'SGD'])
        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-2, log=True)
        scheduler_type = CSH.CategoricalHyperparameter('scheduler_type', ['ReduceLROnPlateau', 'CosineAnnealingLR','CosineAnnealingWarmRestarts'])
        cs.add_hyperparameters([lr, min_lr, optimizer_type, weight_decay, scheduler_type])
        
        
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, log=False)
        cv_fold = CSH.UniformIntegerHyperparameter('cv_fold', lower=1, upper=5)
        cs.add_hyperparameters([sgd_momentum, cv_fold])
        momentum_cond = CS.EqualsCondition(sgd_momentum, optimizer_type, 'SGD')
        cs.add_conditions([momentum_cond])

        num_hidden_layers =  CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=2, upper=10)
        num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=32, upper=512)
        cs.add_hyperparameters([num_hidden_layers, num_hidden_units])
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, log=False)
        cs.add_hyperparameters([dropout_rate])
        
        return cs 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize the metamodel')
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.', default=10)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.', default=20)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=2)
    parser.add_argument('--n_workers',    type=int,   help='Number of workers to run in parallel.', default=1)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.',
                        default='optimiseModel')

    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.',
                        default='en0')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.', default='ddp_debug')
    parser.add_argument('--previous_run', type=str, default=None,
                        help='Path to the directory of the previous run. Prev run is assumed to be in the same '
                             'working dir as current')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--input_size', type=int, default=27, help='Input size of the model')
    parser.add_argument('--output_size', type=int, default=1, help='Output size of the model')
    parser.add_argument('--loss_func', type=str, default='regression', help='loss function can be regression|bpr|hingeloss')
    parser.add_argument('--batch_size', type=int, default=204, help='Batch size of the model')
    
    # sample command from terminal:
    # python metamodel_optimise.py --min_budget 10 --max_budget 20 --n_iterations 2 --n_workers 1 --run_id optimiseModel --nic_name en0 --shared_directory ddp_debug --previous_run None --seed 42 --input_size 27 --output_size 1 --loss_func regression --batch_size 204
    
    
    args = parser.parse_args()
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)

    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        time.sleep(5)   # short artificial delay to make sure the nameserver is already running
        w = SurrogateWorker(seed=args.seed, inputsize=args.input_size, output=args.output_size,
                            loss=args.loss_func, batch=args.batch_size,
                            run_id=args.run_id, host=host,timeout=300,)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)
    
    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)
    
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir)
    ns_host, ns_port = NS.start()

    w = SurrogateWorker(seed=args.seed, inputsize=args.input_size, output=args.output_size,
                        loss=args.loss_func, batch=args.batch_size,
                        run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=300)
    w.run(background=True)

    try:
        prev_dir = os.path.join(os.getcwd(), args.shared_directory, args.previous_run)
        previous_run = hpres.logged_results_to_HBS_result(prev_dir)
    except Exception:
        print('No prev run')
        previous_run = None
    
    bohb = BOHB(configspace=SurrogateWorker.get_configspace(args.seed),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                previous_result=previous_run,
                result_logger=result_logger,
                # eta=args.eta,  # Determines how many configurations advance to the next round.
                )
    
    try:

        res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        print('Best found configuration:', id2config[incumbent]['config'])

    except Exception as e:
        print(e)
        traceback.print_exc()
        print('Exiting BOHB run due to exception--------------->')
        sys.exit('My error message. BohB Run Failed')
    
    with open(os.path.join(working_dir, f'{args.run_id}_results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('logs are located in ------> {}'.format(working_dir))

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (
            sum([r.budget for r in res.get_all_runs()]) / args.max_budget))



