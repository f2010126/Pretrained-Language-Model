import logging
import argparse
import pickle
import time
import sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random
from hpbandster.core.worker import Worker
import traceback
import logging
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import os
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.core.master import Master
from hpbandster.core.dispatcher import Job

from pytorch_lightning import seed_everything, Trainer
from data_modules import get_datamodule
from train_module import GLUETransformer
from pytorch_lightning.loggers import WandbLogger
import argparse
import wandb
import torchmetrics.functional as F
import time
import torch

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment

# test
from pytorch_min import SimpleDataset, MyModel

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    raise ImportError("For this example you need to install pytorch-vision.")

logging.basicConfig(level=logging.DEBUG)


class PyTorchWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
        super().__init__(**kwargs)

        batch_size = 64
        print("Data setup? Inside the worker")
        if torch.cuda.is_available():
            logging.debug("Inside init of Worker CUDA available, using GPU no. of device: {}".format(torch.cuda.device_count()))
            n_devices = torch.cuda.device_count()
            print("n_devices: ", n_devices)
        else:
            logging.debug("Inside init of Worker CUDA not available, using CPU")

    def new_compute(self, config, budget, working_directory,*args, **kwargs):
        seed_everything(0)
        model = MyModel()
        dm = SimpleDataset()
        trainer = Trainer(max_epochs=3, devices=torch.cuda.device_count())
        trainer.fit(model, dm)
        # generate a random number between 0 and 1



        X = torch.Tensor([[1.0], [51.0], [89.0]])
        _, y = model(X)
        print(f"-----> Returning")
        return ({
            'loss': 1 - random.random(),  # remember: HpBandSter always minimizes!
            'info': {'all_metrics': trainer.logged_metrics,
                     }

        })
    def compute(self, config, budget, working_directory, *args, **kwargs):
        print("budget aka epochs------> {}".format(budget))
        if torch.cuda.is_available():
            logging.debug("CUDA available, using GPU no. of device: {}".format(torch.cuda.device_count()))
            n_devices = torch.cuda.device_count()
            accelerator = 'cpu' if n_devices == 0 else 'cuda'
        else:
            logging.debug("CUDA not available, using CPU")
            accelerator = 'cpu'
            n_devices = 1

        seed_everything(0)

        # set up data loaders
        dm = get_datamodule(task_name="tyqiangz", model_name_or_path=config['model_name_or_path'],
                            max_seq_length=config['max_seq_length'], train_batch_size=config['train_batch_size_gpu'],
                            eval_batch_size=config['eval_batch_size_gpu'])
        dm.setup("fit")
        # set up model and experiment
        logging.debug("Inside compute, before model init after data setup")
        model = GLUETransformer(
            model_name_or_path=config['model_name_or_path'],
            num_labels=dm.task_metadata["num_labels"],
            eval_splits=dm.eval_splits,
            task_name=dm.task_name,
            learning_rate=config['learning_rate'],
            adam_epsilon=1e-8,
            warmup_steps=0,
            weight_decay=0.0,
            train_batch_size=config['train_batch_size_gpu'],
            eval_batch_size=config['eval_batch_size_gpu'],
            hyperparameters=config,
        )

        # set up logger

        # set up trainer
        # set up trainer
        logging.debug("Inside compute, before trainer init")
        n_devices = torch.cuda.device_count()
        accelerator = 'cpu' if n_devices == 0 else 'auto'
        ddp = DDPStrategy(process_group_backend="nccl")
        trainer = Trainer(
            max_epochs=int(budget),
            accelerator=accelerator,
            num_nodes=1, # you. IDIOT. you forgot to set this to 1
            devices=n_devices,
            # strategy=ddp,
            strategy='auto',  # Use whatver device is available
            # max_steps=5, limit_val_batches=5, limit_test_batches=5, num_sanity_val_steps=1,  # and no sanity check
           #val_check_interval=1,
         check_val_every_n_epoch=1,  # check_val_every_n_epoch=1 and every 5 batches
        )
        # train model
        print(f"Training model From PID {os.getgid()}")
        try:
            trainer.fit(model, datamodule=dm)
        except Exception as e:
            print("Exception in training: ")
            print(e)
            traceback.print_exc()

        val_acc = trainer.logged_metrics['val_acc_epoch'].item()
        print(f"From PID {os.getgid()}  Best checkpoint path: { trainer.checkpoint_callback.best_model_path}")
        print(f" From PID {os.getgid()} Training complete Metrics Epoch Val Acc: {val_acc}")

        return ({
            'loss': 1 - val_acc,  # remember: HpBandSter always minimizes!
            'info': {'all_metrics': trainer.logged_metrics,
                     }

        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-5, upper=7e-5, default_value='3e-5', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer_name', ['Adam', 'AdamW','SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)
        model_name_or_path = CSH.CategoricalHyperparameter('model_name_or_path', ['bert-base-uncased', 'bert-large-uncased'])
        max_seq_length = CSH.UniformIntegerHyperparameter('max_seq_length', lower=32, upper=512, default_value=128, log=True)
        cs.add_hyperparameters([model_name_or_path, max_seq_length])

        train_batch_size_gpu = CSH.UniformIntegerHyperparameter('train_batch_size_gpu', lower=2, upper=3, default_value=2, log=True)
        eval_batch_size_gpu = CSH.UniformIntegerHyperparameter('eval_batch_size_gpu', lower=2, upper=3, default_value=2, log=True)
        cs.add_hyperparameters([train_batch_size_gpu, eval_batch_size_gpu])

        scheduler_name = CSH.CategoricalHyperparameter('scheduler_name', ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
        cs.add_hyperparameters([scheduler_name])
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                      log=False)
        num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

        cs.add_hyperparameters([dropout_rate, num_fc_units])

        return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoHB MultiNode Example')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=3)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4) # no of times to sample??
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    # master also counts as a worker. so if n_workers is 1, only the master is used
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = PyTorchWorker(run_id=args.run_id, host=host, timeout=6000)
        w.load_nameserver_credentials(working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    w = PyTorchWorker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=6000)
    w.run(background=True)

    # Run an optimizer
    # We now have to specify the host, and the nameserver information
    bohb = BOHB(configspace=PyTorchWorker.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget,
                max_budget=args.max_budget
                )
    try:
        res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)
    except Exception as e:
        print(e)
        traceback.print_exc()
        print('Exiting run due to exception--------------->')
        sys.exit('My error message. BohB Run Failed')

    # In a cluster environment, you usually want to store the results for later analysis.
    # One option is to simply pickle the Result object
    savepath= os.path.join(args.shared_directory, args.run_id)
    with open(os.path.join(savepath, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (
                sum([r.budget for r in res.get_all_runs()]) / args.max_budget))
