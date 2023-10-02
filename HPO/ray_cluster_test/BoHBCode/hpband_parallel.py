import logging
import multiprocessing
import os
import pickle
import traceback
import uuid
import sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import argparse
import time

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        print("budget aka epochs------> {}".format(budget))
        if torch.cuda.is_available():
            logging.debug("CUDA available, using GPU no. of device: {}".format(torch.cuda.device_count()))
        else:
            logging.debug("CUDA not available, using CPU")

        seed_everything(0)
        data_dir = os.environ.get('DATADIR', os.path.join(os.getcwd(), "tokenised_data"))
        hpo_working_directory = os.environ.get('HPO_WORKING_DIR', os.getcwd())

        # set up data and model
        dm = get_datamodule(task_name="sentilex", model_name_or_path=config['model_name_or_path'],
                            max_seq_length=config['max_seq_length'],
                            train_batch_size=config['per_device_train_batch_size'],
                            eval_batch_size=config['per_device_eval_batch_size'], data_dir=data_dir)
        dm.setup("fit")
        model = PLMTransformer(config=config, num_labels=dm.task_metadata['num_labels'])
        n_devices = torch.cuda.device_count()
        accelerator = 'cpu' if n_devices == 0 else 'auto'
        trial_id = str(uuid.uuid4().hex)[:5]
        log_dir = os.path.join(hpo_working_directory, f"{self.run_id}_logs/run_{trial_id}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"trial run logged at ------> {log_dir}")
        # make the shared directory
        trainer = Trainer(
            max_epochs=int(budget),
            accelerator=accelerator,
            num_nodes=1,
            devices=n_devices,
            strategy="ddp_spawn",
            logger=[CSVLogger(save_dir=log_dir, name="csv_logs", version="."),
                    TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version=".")],
            max_time="00:1:00:00",  # give each run a time limit
            num_sanity_val_steps=1,
            log_every_n_steps=10,
            val_check_interval=50,
        )
        # train model
        try:
            trainer.fit(model, datamodule=dm)
        except Exception as e:
            print("Exception in training: ")
            print(e)
            traceback.print_exc()

        # print metrics
        val_acc = trainer.callback_metrics['val_acc_epoch'].item()
        print(f"From PID {os.getgid()}  Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")
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

        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-5, upper=7e-5, log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer_name', ['Adam', 'AdamW', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)
        model_name_or_path = CSH.CategoricalHyperparameter('model_name_or_path',
                                                           ["bert-base-uncased", "bert-base-multilingual-cased",
                                                            "deepset/bert-base-german-cased-oldvocab",
                                                            "uklfr/gottbert-base",
                                                            "dvm1983/TinyBERT_General_4L_312D_de",
                                                            "linhd-postdata/alberti-bert-base-multilingual-cased",
                                                            "dbmdz/distilbert-base-german-europeana-cased"])
        max_seq_length = CSH.UniformIntegerHyperparameter('max_seq_length', lower=32, upper=512, log=True)

        train_batch_size_gpu = CSH.UniformIntegerHyperparameter('per_device_train_batch_size', lower=2, upper=3,
                                                                log=True)
        eval_batch_size_gpu = CSH.UniformIntegerHyperparameter('per_device_eval_batch_size', lower=2, upper=3,
                                                               log=True)
        cs.add_hyperparameters([train_batch_size_gpu, eval_batch_size_gpu])

        scheduler_name = CSH.CategoricalHyperparameter('scheduler_name',
                                                       ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',
                                                        'constant', 'constant_with_warmup'])
        cs.add_hyperparameters([model_name_or_path, max_seq_length, scheduler_name])

        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-3, log=True)
        warmup_steps = CSH.UniformIntegerHyperparameter('warmup_steps', lower=10, upper=1000, log=True)
        cs.add_hyperparameters([weight_decay, warmup_steps])

        adam_epsilon = CSH.UniformFloatHyperparameter('adam_epsilon', lower=1e-8, upper=1e-6, log=True)
        gradient_accumulation_steps = CSH.UniformIntegerHyperparameter('gradient_accumulation_steps', lower=1, upper=4,
                                                                       log=True)
        max_grad_norm = CSH.UniformFloatHyperparameter('max_grad_norm', lower=0.0, upper=2.0, log=False)
        gradient_clip_algorithm = CSH.CategoricalHyperparameter('gradient_clip_algorithm', ['norm', 'value'])

        cs.add_hyperparameters([adam_epsilon, gradient_accumulation_steps, max_grad_norm, gradient_clip_algorithm])
        return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoHB MultiNode Example')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=3)
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

    args = parser.parse_args()

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    # make the shared directory
    os.makedirs(working_dir, exist_ok=True)
    # store working dir as env variable
    os.environ['HPO_WORKING_DIR'] = working_dir

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = PyTorchWorker(run_id=args.run_id, host=host, timeout=6000, )
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    # comment out for now.
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
