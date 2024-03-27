"""
Full implementation of multi cluster BOHB with Ray Train
"""
import argparse

import os
import pickle
import sys
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import time
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB

import ray
import uuid
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import traceback

try:
    from bohb_ray import transformer_train_function
    from experiment_utilities import remove_checkpoint_files

except ImportError:
    from bohb_ray import transformer_train_function
    from experiment_utilities import remove_checkpoint_files


class RayWorker(Worker):
    def __init__(self, *args, data_dir="./", log_dir="./", task_name="sentilex", seed=42, num_gpu=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir  # mostly HPO/ray_cluster_test/BoHBCode/tokenized_data
        self.log_dir = log_dir  # mostly HPO/ray_cluster_test/BoHBCode/datasetruns/RUN_NAME
        self.task = task_name
        self.seed = seed
        self.gpus = num_gpu
        if ray.is_initialized():
            print('Ray is initialized')
        else:
            print('Ray is not initialized')

    def compute(self, config, budget, working_directory, *args, **kwargs):
        self.logger.debug(f"Working directory: {working_directory} with devices: ")
        scaling_config = ScalingConfig(num_workers=self.gpus, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1})

        run_config = ray.train.RunConfig(
            log_to_file=False,
            storage_path=os.path.join(self.log_dir, "ray_results"),
            # checkpoint_config=ray.train.CheckpointConfig(
            #     num_to_keep=1,
            #     checkpoint_score_attribute="ptl/val_accuracy",
            #     checkpoint_score_order="max",
            # )
        )
        # [5] Launch distributed training job.
        config['epochs'] = int(budget)
        config['task'] = self.task
        config['log'] = self.log_dir
        config['data_dir'] = self.data_dir
        config['run_id'] = self.run_id
        config['trial_id'] = str(uuid.uuid4().hex)[:5]
        config['seed'] = self.seed

        trainer = TorchTrainer(transformer_train_function,
                               scaling_config=scaling_config,
                               train_loop_config=config,
                               run_config=run_config,

                               )
        try:
            result = trainer.fit()
            end_acc = result.metrics['ptl/val_accuracy']
            # pick the required info
            info_dict = {k: result.metrics[k] for k in
                     result.metrics.keys() & {"train_acc", "train_loss", "train_f1", "val_acc", "val_acc_epoch",
                                              "val_loss", "val_loss_epoch", "val_f1", "val_f1_epoch", "ptl/val_loss",
                                              "ptl/val_accuracy", "ptl/val_f1"}}
            # delete the checkpoint file .pt
            remove_checkpoint_files(result.path)
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('Trail failed--------------->')
            end_acc =  -10000 # it failed. so worst possible accuracy
            info_dict = {'error': 'Trail failed'}

        return ({
            'loss': -end_acc,  # remember: HpBandSter always minimizes! So we need to negate the accuracy
            'info': info_dict
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
        # Dataset related
        max_seq_length = CSH.CategoricalHyperparameter('max_seq_length', choices=[128, 256, 512])
        train_batch_size_gpu = CSH.CategoricalHyperparameter('per_device_train_batch_size', choices=[4, 8, 16])
        # eval_batch_size_gpu = CSH.CategoricalHyperparameter('per_device_eval_batch_size', choices=[4, 8, 16]) # Use
        # train
        cs.add_hyperparameters([train_batch_size_gpu, max_seq_length])  # eval_batch_size_gpu,
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
                                                        'cosine_with_hard_restarts_with_warmup',
                                                        'polynomial_decay_with_warmup', 'constant_with_warmup'])

        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-3, log=True)
        warmup_steps = CSH.CategoricalHyperparameter('warmup_steps', choices=[10, 100, 500])
        cs.add_hyperparameters([scheduler_name, weight_decay, warmup_steps])

        adam_epsilon = CSH.UniformFloatHyperparameter('adam_epsilon', lower=1e-8, upper=1e-6, log=True)
        gradient_accumulation_steps = CSH.CategoricalHyperparameter('gradient_accumulation_steps',
                                                                    choices=[1, 4, 8, 16])
        cs.add_hyperparameters([adam_epsilon, gradient_accumulation_steps])
        return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=2)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=2)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=1)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the '
                             'clusters scheduler.',
                        default='UsingRay')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.',
                        default='eth0')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.', default='ddp_debug')
    parser.add_argument('--task_name', type=str, help='Which task to run.', default='tagesschau')
    parser.add_argument('--eta', type=int, help='Eta value for BOHB', default=2)
    parser.add_argument('--num_gpu', type=int, help='Number of GPUs to use per worker', default=2)
    parser.add_argument('--previous_run', type=str, default=None,
                        help='Path to the directory of the previous run. Prev run is assumed to be in the same '
                             'working dir as current')

    args = parser.parse_args()
    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    # where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)

    data_dir = os.path.join(os.getcwd(), 'tokenized_data')

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = RayWorker(run_id=args.run_id, host=host, timeout=3000, task_name=args.task_name,
                      log_dir=working_dir, num_gpu=args.num_gpu, data_dir=data_dir)
        # increase timeout to 1 hour
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    # ensure the file is empty, init the config and results json
    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir, )
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    # comment out for now.

    w = RayWorker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=3000,
                  data_dir=data_dir, task_name=args.task_name, log_dir=working_dir, num_gpu=args.num_gpu)
    # increase timeout to 1 hour
    w.run(background=True)

    try:
        prev_dir = os.path.join(os.getcwd(), args.shared_directory, args.previous_run)
        previous_run = hpres.logged_results_to_HBS_result(prev_dir)
    except Exception:
        print('No prev run')
        previous_run = None

    # Run an optimizer
    # We now have to specify the host, and the nameserver information
    bohb = BOHB(configspace=RayWorker.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                previous_result=previous_run,
                result_logger=result_logger,
                eta=args.eta,  # Determines how many configurations advance to the next round.
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

    print('logs are located in ------> {}'.format(working_dir))

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (
            sum([r.budget for r in res.get_all_runs()]) / args.max_budget))
