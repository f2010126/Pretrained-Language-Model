import argparse
import logging
import multiprocessing
import os
import pickle
import sys
import time
import traceback
import uuid
import datetime

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.core.dispatcher import Job
from hpbandster.optimizers import BOHB as BOHB
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from ConfigSpace.read_and_write import json as cs_json
from pytorch_lightning.strategies import DDPStrategy

logger = multiprocessing.log_to_stderr()

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("For this example you need to install pytorch.")

# local imports
sys.path.append("/work/dlclarge1/dsengupt-zap_hpo_og/TinyBert/HPO/ray_cluster_test/BoHBCode")

try:
    from data_modules import get_datamodule
    from train_module import PLMTransformer, GLUETransformer
    from asha_ray_transformers import trial_dir_name
except ImportError:
    from data_modules import get_datamodule
    from train_module import PLMTransformer, GLUETransformer


class PyTorchWorker(Worker):
    def __init__(self, *args, data_dir="./", log_dir="./", task_name="sentilex", seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.task = task_name
        self.seed = seed


    def compute(self,config_id, config, budget, working_directory, *args, **kwargs):
        print("budget aka epochs------> {}".format(budget))
        if torch.cuda.is_available():
            logging.debug("CUDA available, using GPU no. of device: {}".format(torch.cuda.device_count()))
        else:
            logging.debug("CUDA not available, using CPU")

        seed_everything(self.seed)

        # set up data and model
        dm = get_datamodule(task_name=self.task, model_name_or_path=config['model_name_or_path'],
                            max_seq_length=config['max_seq_length'],
                            train_batch_size=config['per_device_train_batch_size'],
                            eval_batch_size=config['per_device_eval_batch_size'], data_dir=self.data_dir)
        dm.setup("fit")
        model = PLMTransformer(config=config, num_labels=dm.task_metadata['num_labels'])
        n_devices = torch.cuda.device_count()
        accelerator = 'cpu' if n_devices == 0 else 'auto'
        strategy = DDPStrategy(timeout=datetime.timedelta(seconds=3600)),

        # set up logger
        trial_id = str(uuid.uuid4().hex)[:5]
        folder_name = config['model_name_or_path'].split("/")[-1]  # last part is usually the model name
        log_dir = os.path.join(self.log_dir, f"{self.run_id}_logs/{folder_name}/run_{trial_id}")
        os.makedirs(log_dir, exist_ok=True)

        trainer = Trainer(
            max_epochs=int(budget),
            accelerator="auto",
            num_nodes=1,
            devices="auto",
            strategy="ddp_spawn",  # change to ddp_spawn when in interactive mode
            logger=[TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version="."),
                    CSVLogger(save_dir=log_dir, name="csv_logs", version=".")],
            max_time="00:1:00:00",  # give each run a time limit

            log_every_n_steps=5,
            val_check_interval=0.5,
            enable_checkpointing=False,

            limit_train_batches=10,
            limit_val_batches=5,
            limit_test_batches=5,

            accumulate_grad_batches=config['gradient_accumulation_steps'],
            # gradient_clip_val=config['max_grad_norm'],
            # gradient_clip_algorithm=config['gradient_clip_algorithm'],
        )
        # train model
        try:
            trainer.fit(model, datamodule=dm)
        except Exception as e:
            print(f"Exception in training job {config_id} : with config {config} and budget {budget}")
            print(e)
            traceback.print_exc()

        trainer.strategy.barrier()
        val_acc = 1 - trainer.callback_metrics['val_acc_epoch'].item()


        return ({
            'loss': 1 - val_acc,  # remember: HpBandSter always minimizes!
            'info': {'all_metrics': trainer.callback_metrics,
                     'budget': budget,
                     'trial_id': trial_id,
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
        """
        hpo_config = {
    'model_name_or_path': tune.choice(["bert-base-uncased", "bert-base-multilingual-cased",
                                       "deepset/bert-base-german-cased-oldvocab", "uklfr/gottbert-base",
                                       "dvm1983/TinyBERT_General_4L_312D_de",
                                       "linhd-postdata/alberti-bert-base-multilingual-cased",
                                       "dbmdz/distilbert-base-german-europeana-cased", ]),

    'optimizer_name': tune.choice(["AdamW", "Adam"]),
    'scheduler_name': tune.choice(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]),
    'learning_rate': tune.loguniform(1e-5, 6e-5),
    'weight_decay': tune.loguniform(1e-5, 1e-3),
    'adam_epsilon': tune.loguniform(1e-8, 1e-6),
    'warmup_steps': tune.choice([0, 100, 1000]),
    'per_device_train_batch_size': tune.choice([2]),
    'per_device_eval_batch_size': tune.choice([2, ]),
    'gradient_accumulation_steps': tune.choice([1, 2, 4]),
    'num_train_epochs': tune.choice([2, 3, 4]),--> Budget
    'max_steps': tune.choice([-1, 100, 1000]),
    'max_grad_norm': tune.choice([0.0, 1.0, 2.0]),
    'seed': tune.choice([42, 1234, 2021]),
    'max_seq_length': tune.choice([128, 256, 512]),
    "num_epochs": tune.choice([2, 3, 4]),
    "gradient_clip_algorithm": tune.choice(["norm", "value"]),
}
        """
        # Dataset related
        max_seq_length = CSH.CategoricalHyperparameter('max_seq_length', choices=[128, 256, 512])
        train_batch_size_gpu = CSH.CategoricalHyperparameter('per_device_train_batch_size', choices=[4, 8, 16])
        eval_batch_size_gpu = CSH.CategoricalHyperparameter('per_device_eval_batch_size', choices=[4, 8, 16])
        cs.add_hyperparameters([train_batch_size_gpu, eval_batch_size_gpu, max_seq_length])
        model_name_or_path = CSH.CategoricalHyperparameter('model_name_or_path',
                                                           ["bert-base-uncased", "bert-base-multilingual-cased",
                                                            "deepset/bert-base-german-cased-oldvocab",
                                                            "uklfr/gottbert-base",
                                                            "dvm1983/TinyBERT_General_4L_312D_de",
                                                            "linhd-postdata/alberti-bert-base-multilingual-cased",
                                                            "dbmdz/distilbert-base-german-europeana-cased"])

        # Model related
        optimizer = CSH.CategoricalHyperparameter('optimizer_name', ['Adam', 'AdamW', 'SGD', 'RAdam'])
        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=2e-5, upper=7e-5, log=True)
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, log=False)
        cs.add_hyperparameters([model_name_or_path, lr, optimizer, sgd_momentum])
        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        scheduler_name = CSH.CategoricalHyperparameter('scheduler_name',
                                                       ['linear_with_warmup', 'cosine_with_warmup',
                                                        'inverse_sqrt', 'cosine_with_hard_restarts_with_warmup',
                                                        'polynomial_decay_with_warmup', 'constant_with_warmup'])

        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-5, log=True)
        warmup_steps = CSH.CategoricalHyperparameter('warmup_steps', choices=[10, 100, 500])
        cs.add_hyperparameters([scheduler_name, weight_decay, warmup_steps])

        adam_epsilon = CSH.UniformFloatHyperparameter('adam_epsilon', lower=1e-8, upper=1e-7, log=True)
        gradient_accumulation_steps = CSH.CategoricalHyperparameter('gradient_accumulation_steps',
                                                                    choices=[1, 4, 8, 16])
        # max_grad_norm = CSH.UniformFloatHyperparameter('max_grad_norm', lower=0.0, upper=2.0, log=False)
        # gradient_clip_algorithm = CSH.CategoricalHyperparameter('gradient_clip_algorithm', ['norm', 'value'])
        cs.add_hyperparameters([adam_epsilon, gradient_accumulation_steps])
        return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoHB MultiNode Example')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=2)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=10)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer',
                        default=4)  # no of times to sample??
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    # master also counts as a worker. so if n_workers is 1, only the master is used
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the '
                             'clusters scheduler.')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')
    parser.add_argument("--task-name", type=str, default="sentilex")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()


    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    # where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)
    # central location for the datasets
    data_path = os.path.join(os.getcwd(), "tokenized_data")
    os.makedirs(data_path, exist_ok=True)

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = PyTorchWorker(data_dir=data_path, log_dir=working_dir, task_name=args.task_name,
                          run_id=args.run_id, host=host, timeout=300, seed=args.seed)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    # ensure the file is empty, init the config and results json
    # result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)
    # Save the configspace
    cs_string = cs_json.write(PyTorchWorker.get_configspace())
    with open(os.path.join(working_dir, f'configspace.json'), 'w') as fh:
        fh.write(cs_string)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    # comment out for now.

    w = PyTorchWorker(data_dir=data_path, log_dir=working_dir, task_name=args.task_name,
                      run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port,
                      timeout=600, seed=args.seed)
    w.run(background=True)

    try:
        previous_run = hpres.logged_results_to_HBS_result(working_dir)
    except Exception:
        print('No prev run')
        previous_run = None

    # Run an optimizer
    # We now have to specify the host, and the nameserver information
    bohb = BOHB(configspace=PyTorchWorker.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                previous_result=None,
                # result_logger=result_logger,
                )
    try:
        res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        print('Best found configuration:', id2config[incumbent]['config'])

    except Exception as e:
        print(e)
        traceback.print_exc()
        print('Exiting run due to exception--------------->')
        sys.exit('My error message. BohB Run Failed')

    # In a cluster environment, you usually want to store the results for later analysis.
    # One option is to simply pickle the Result object
    with open(os.path.join(working_dir, f'{args.run_id}_results.pkl'), 'wb') as fh:
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
