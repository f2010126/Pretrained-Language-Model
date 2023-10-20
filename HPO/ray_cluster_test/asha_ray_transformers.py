import argparse
import logging
import multiprocessing
import os
import sys
import traceback
# for load dict
from typing import List

import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.train.lightning import LightningConfigBuilder
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
from ray.train.torch import TorchTrainer
from ray.tune import Callback
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler

# local changes
logger = multiprocessing.log_to_stderr()
# local imports
try:
    from HPO.ray_cluster_test.BoHBCode.data_modules import Omp, get_datamodule
    from HPO.ray_cluster_test.BoHBCode.train_module import PLMTransformer
except ImportError:
    from BoHBCode.data_modules import Omp, get_datamodule
    from BoHBCode.train_module import PLMTransformer
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

hpo_config = {
    'model_name_or_path': tune.choice(["bert-base-uncased", "bert-base-multilingual-cased",
                                       "deepset/bert-base-german-cased-oldvocab", "uklfr/gottbert-base",
                                       "dvm1983/TinyBERT_General_4L_312D_de",
                                       "linhd-postdata/alberti-bert-base-multilingual-cased",
                                       "dbmdz/distilbert-base-german-europeana-cased", ]),

    'optimizer_name': tune.choice(['Adam', 'AdamW', 'SGD', 'RAdam']),
    'scheduler_name': tune.choice(['linear_with_warmup', 'cosine_with_warmup',
                                   'inverse_sqrt', 'cosine_with_hard_restarts_with_warmup',
                                   'polynomial_decay_with_warmup',
                                   'constant_with_warmup']),
    'learning_rate': tune.loguniform(2e-5, 6e-5),
    'weight_decay': tune.loguniform(1e-5, 1e-3),
    'adam_epsilon': tune.loguniform(1e-8, 1e-6),
    'sgd_momentum': tune.loguniform(0.8, 0.99),
    'warmup_steps': tune.choice([100, 500, 1000]),
    'per_device_train_batch_size': tune.choice([2, 4, 8]),
    'per_device_eval_batch_size': tune.choice([2, 4, 8]),
    'gradient_accumulation_steps': tune.choice([2, 4, 8]),
    'max_steps': tune.choice([-1, 100, 1000]),
    'max_grad_norm': tune.choice([1.0, 2.0]),
    'seed': tune.choice([42, 1234, 2021]),
    'max_seq_length': tune.choice([128, 256, 512]),
    "num_epochs": tune.choice([2, 3, 4]),
    "gradient_clip_algorithm": tune.choice(["norm", "value"]),
}


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['metrics/val_accuracy']} for {trial.trainable_name} with config {trial.config}")

    def on_trial_error(
            self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        print(f"Got error for {trial.trainable_name} with config {trial.config}")


def objective_torch_trainer(config):
    data_dir = config['data_dir']
    logging.debug(f"dir {data_dir} and cwd {os.getcwd()}")
    dm = get_datamodule(task_name=config['task_name'], model_name_or_path=config['model_name_or_path'],
                        max_seq_length=config['max_seq_length'],
                        train_batch_size=config['per_device_train_batch_size'],
                        eval_batch_size=config['per_device_eval_batch_size'], data_dir=data_dir)
    dm.setup("fit")
    model = PLMTransformer(config=config, num_labels=dm.task_metadata['num_labels'])
    ckpt_report_callback = RayTrainReportCallback()
    log_dir = os.path.join(os.getcwd(), "ray_results_log/torch_trainer_logs")
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        logger=[CSVLogger(save_dir=log_dir, name="csv_logs", version="."),
                TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version=".")],

        # If fractional GPUs passed in, convert to int.
        devices='auto',
        enable_progress_bar=True,
        max_time="00:1:00:00",  # give each run a time limit
        val_check_interval=50,  # check validation after 100 train batches
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[ckpt_report_callback],
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        gradient_clip_val=config['max_grad_norm'],
        gradient_clip_algorithm=config['gradient_clip_algorithm'],
        accelerator='auto',
        log_every_n_steps=50,
    )

    # Validate your Lightning trainer configuration
    try:
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)
    except Exception as e:
        print(f"config ------> {config}")
        print(traceback.format_exc())
        print("Other error", e)
        print(traceback.format_exc())
    print("Finished obj")


def torch_trainer_asha(gpus_per_trial=0, num_trials=10, exp_name='bohb_sample', task_name="sentilex"):
    if torch.cuda.is_available():
        use_gpu = True
        # each Trainer gets that.
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_trial, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
        )
    else:
        use_gpu = False
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=1, use_gpu=use_gpu, resources_per_worker={"CPU": 1, }
        )

    train_fn_with_parameters = tune.with_parameters(objective_torch_trainer,
                                                    data_dir=os.path.join(os.getcwd(), "tokenized_data"))
    ray_trainer = TorchTrainer(
        objective_torch_trainer,
        scaling_config=scaling_config,
    )
    hpo_config['data_dir'] = os.path.join(os.getcwd(), "tokenised_data")
    hpo_config['task_name'] = task_name

    result_dir = os.path.join(os.getcwd(), "ray_results_result")
    # Fault Tolerance Code
    restore_path = os.path.join(result_dir, exp_name)
    print(f"Restore path is {restore_path}, result_dir is {result_dir}")

    max_iterations = 3
    # scheduler
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric="metrics/val_accuracy",
        mode='max',
        max_t=5,
        grace_period=3,
        reduction_factor=3,
        brackets=1,
    )

    if tune.Tuner.can_restore(restore_path):
        print("Restoring from checkpoint---->")
        tuner = tune.Tuner.restore(restore_path,
                                   trainable=ray_trainer,
                                   resume_unfinished=True,
                                   resume_errored=True,
                                   restart_errored=False,
                                   param_space={"train_loop_config": hpo_config},
                                   )
    else:
        tuner = tune.Tuner(ray_trainer,
                           tune_config=tune.TuneConfig(
                               scheduler=asha_scheduler,
                               reuse_actors=False,
                               num_samples=num_trials,
                           ),
                           run_config=ray.train.RunConfig(
                               name=exp_name,
                               verbose=2,
                               storage_path=result_dir,
                               log_to_file=True,
                               checkpoint_config=ray.train.CheckpointConfig(
                                   num_to_keep=3,
                                   checkpoint_score_attribute="metrics/val_accuracy",
                                   checkpoint_score_order="max",
                               ),
                           ),
                           param_space={"train_loop_config": hpo_config},
                           )

    logger.debug(f"Tuner setup. Starting tune.run")

    try:
        results = tuner.fit()
        best_result = results.get_best_result(metric="metrics/val_accuracy", mode="max")
        print("Best hyperparameters found were: ", best_result.config)
    except ray.exceptions.RayTaskError:
        print("User function raised an exception!")
    except Exception as e:
        print("Other error", e)
        print(traceback.format_exc())
    print("END")


def parse_args():
    parser = argparse.ArgumentParser(description="Bohb on Slurm")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")  # store_false will default to True
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
             "Ray Client.")
    parser.add_argument("--exp-name", type=str, default="tune_bohb")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-gpu", type=int, default=4, help="Number of distributed workers per trial")
    parser.add_argument("--task-name", type=str, default="sentilex")
    return parser.parse_known_args()


if __name__ == "__main__":
    # args, _ = parse_args()
    #
    # if args.server_address:
    #     context = ray.init(f"ray://{args.server_address}")
    # elif args.ray_address:
    #     context = ray.init(address=args.ray_address)
    # elif args.smoke_test:
    #     context = ray.init()
    # else:
    #     context = ray.init(address='auto', _redis_password=os.environ['redis_password'])
    #
    # print("Dashboard URL: http://{}".format(context.dashboard_url))

    parser = argparse.ArgumentParser(description="Tune on local")
    parser.add_argument(
        "--smoke-test", action="store_false", help="Finish quickly for testing")  # store_false will default to True
    parser.add_argument("--exp-name", type=str, default="local_tune_asha10")
    parser.add_argument("--task-name", type=str, default='omp')
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--num-gpu", type=int, default=0, help="Number of distributed workers per trial")
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        args.exp_name = args.exp_name + "_cpu"
        args.num_gpu = 0
        args.smoke_test = True

    if args.smoke_test:
        print("Running smoke test")
        args.exp_name = args.exp_name + "_smoke"
        torch_trainer_asha(gpus_per_trial=args.num_gpu, num_trials=3,
                           exp_name=args.exp_name, task_name=args.task_name)
    else:
        torch_trainer_asha(gpus_per_trial=args.num_gpu, num_trials=args.num_trials,
                           exp_name=args.exp_name, task_name=args.task_name)

    print("END OF MAIN SCRIPT")
