import ray.air as air
import ray.tune as tune
from typing import List
import argparse
from ray.tune import Callback

import torch
import os
import ray
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from ray import air, tune

import pytorch_lightning as pl
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import time
from ray.train.lightning import LightningConfigBuilder, LightningTrainer
import traceback
import math
import torch
from ray.tune.experiment import Trial

from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)

# local imports
try:
    from BoHBCode.data_modules import OmpData, get_datamodule
    from BoHBCode.train_module import AshaTransformer
except ImportError:
    from .BoHBCode.data_modules import OmpData, get_datamodule
    from .BoHBCode.train_module import AshaTransformer


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['ptl/val_accuracy']} for {trial.trainable_name} with config {trial.config}")

    def on_trial_error(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(f"Got error for {trial.trainable_name} with config {trial.config}")


hpo_config = {
    'model_name_or_path': tune.choice(["bert-base-uncased", "bert-large-uncased"]),
    'optimizer_name': tune.choice(["AdamW", "Adam"]),
    'scheduler_name': tune.choice(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]),
    'learning_rate': tune.loguniform(1e-5, 6e-5),
    'weight_decay': tune.loguniform(1e-5, 1e-3),
    'adam_epsilon': tune.loguniform(1e-8, 1e-6),
    'warmup_steps': tune.choice([0, 100, 1000]),
    'per_device_train_batch_size': tune.choice([2]),
    'per_device_eval_batch_size': tune.choice([2, ]),
    'gradient_accumulation_steps': tune.choice([1, 2, 4]),
    'num_train_epochs': tune.choice([2, 3, 4]),
    'max_steps': tune.choice([-1, 100, 1000]),
    'max_grad_norm': tune.choice([0.0, 1.0, 2.0]),
    'seed': tune.choice([42, 1234, 2021]),
    'max_seq_length': tune.choice([128, 256, 512]),
    "num_epochs": tune.choice([2, 3, 4]),
}


# https://github.com/ray-project/ray/issues/37394
# RAY TUNE LIGHTNING TRAINER
def tune_lightning(num_samples=10, num_epochs=10, exp_name="tune_transform"):
    config = {
        'model_name_or_path': tune.choice(["bert-base-uncased", "bert-large-uncased"]),
        'optimizer_name': tune.choice(["AdamW", "Adam"]),
        'scheduler_name': tune.choice(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]),
        'learning_rate': tune.loguniform(1e-5, 6e-5),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'adam_epsilon': tune.loguniform(1e-8, 1e-6),
        'warmup_steps': tune.choice([0, 100, 1000]),
        'per_device_train_batch_size': tune.choice([8, 16, 32]),
        'per_device_eval_batch_size': tune.choice([8, 16, 32]),
        'gradient_accumulation_steps': tune.choice([1, 2, 4]),
        'num_train_epochs': tune.choice([2, 3, 4]),
        'max_steps': tune.choice([-1, 100, 1000]),
        'max_grad_norm': tune.choice([0.0, 1.0, 2.0]),
        'seed': tune.choice([42, 1234, 2021]),
        'max_seq_length': tune.choice([128, 256, 512]),
        "batch_size": tune.choice([32, 64, 128]),
    }

    data_hpo = {
        'max_seq_length': tune.choice([128, 256, 512]),
        "batch_size": tune.choice([32, 64, 128]),
        'per_device_train_batch_size': tune.choice([8, 16, 32]),
        'per_device_eval_batch_size': tune.choice([8, 16, 32]),
    }

    # Static configs that does not change across trials
    # doesn't call prepare data
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-ptl-example", version=".")
    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_trial = 2
        scaling_config = ray.air.config.ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_trial, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
        )
    else:
        accelerator = 'cpu'
        use_gpu = False
        scaling_config = ray.air.config.ScalingConfig(
            # no of other nodes?
            num_workers=1, use_gpu=use_gpu, resources_per_worker={"CPU": 1, }
        )
    print(f" No of GPUs available : {torch.cuda.device_count()} and accelerator is {accelerator}")

    static_lightning_config = (
        LightningConfigBuilder()
        .module(cls=AshaTransformer, num_labels=2, dataset_name="omp")
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger, )
        .fit_params(datamodule=OmpData, )
        # .strategy(name='ddp')
        .checkpointing(monitor="ptl/val_accuracy", save_top_k=2, mode="max")
        .build()
    )

    # Searchable configs across different trials
    searchable_lightning_config = (
        LightningConfigBuilder()
        .module(config=config)
        .build()
    )

    # Make sure to also define an AIR CheckpointConfig here
    # to properly save checkpoints in AIR format.
    run_config = ray.air.config.RunConfig(
        checkpoint_config=ray.air.config.CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
        callbacks=[MyCallback()]
    )

    scheduler = ASHAScheduler(max_t=num_epochs,  # max no of epochs a trial can run
                              grace_period=1, reduction_factor=2,
                              time_attr="training_iteration")

    # as per  the post, this controls 1 Trainer.So I want each trainer to have 2 workers, each with 1 GPU, 2 CPU
    # Hence, for num_samples trials there's num_samples Trainers, each with 2 workers, each with 1 GPU, 2 CPU.
    # I have 2 nodes, 2 gpus each.
    # Result, only 1 trainer runs, uses 2 GPUs.

    lightning_trainer = LightningTrainer(
        lightning_config=static_lightning_config,
        scaling_config=scaling_config,
    )

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": searchable_lightning_config},
        tune_config=tune.TuneConfig(  # for Tuner
            time_budget_s=3000,
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,  # Number of times to sample from the hyperparameter space
            scheduler=scheduler,
            reuse_actors=False,
        ),
        run_config=air.RunConfig(  # for Tuner.run
            name=exp_name,
            verbose=2,
            storage_path="./ray_results",
            log_to_file=True,
            # configs given to Tuner are used.
            # progress_reporter=reporter,
            checkpoint_config=ray.air.config.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
            callbacks=[MyCallback()]
        ),

    )

    try:
        start = time.time()
        results = tuner.fit()
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
        print("Best hyperparameters found were: ", best_result)
    except ray.exceptions.RayTaskError:
        print("User function raised an exception!")
    except Exception as e:
        print("Other error", e)
        print(traceback.format_exc())

    print("END")


# RAY TUNE 3.0 WITH TORCH TRAINER

def objective_torch_trainer(config):
    print(f"config-------> {config}")
    dm = get_datamodule(task_name="sentilex", model_name_or_path=config['model_name_or_path'],
                        max_seq_length=config['max_seq_length'],
                        train_batch_size=config['per_device_train_batch_size'],
                        eval_batch_size=config['per_device_eval_batch_size'])
    dm.setup("fit")
    model = AshaTransformer(config=config, num_labels=dm.task_metadata['num_labels'], dataset_name="omp")
    ckpt_report_callback = RayTrainReportCallback()
    log_dir = os.path.join(os.getcwd(), "ray_results/torch_trainer_logs")
    print(f"log_dir-----> {log_dir}")
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        limit_train_batches=10,
        log_every_n_steps=5,
        logger=[CSVLogger(save_dir=log_dir, name="csv_torch_trainer_logs", version="."),
                TensorBoardLogger(save_dir=log_dir, name="tensorboard_torch_trainer_logs", version=".")],

        # If fractional GPUs passed in, convert to int.
        devices='auto',
        accelerator='auto',
        num_nodes=1,
        enable_progress_bar=True,
        max_time="00:12:00:00", # give each run a time limit

        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[ckpt_report_callback])

    # Validate your Lightning trainer configuration
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)
    print(f"trainer metrics: {trainer.callback_metrics}")
    print("Finished obj")


def tune_func_torch_trainer(num_samples=10, num_epochs=10, exp_name="torch_transform"):
    # hpo_config
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=1,
                              reduction_factor=2)

    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_trial = 4
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=2, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 4}
        )
    else:
        accelerator = 'cpu'
        use_gpu = False
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=5, use_gpu=use_gpu, resources_per_worker={"CPU": 1, }
        )
    # A `RunConfig` was passed to both the `Tuner` and the `TorchTrainer`.
    # The run config passed to the `Tuner` is the one that will be used
    ray_trainer = TorchTrainer(
        objective_torch_trainer,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        )
    )

    result_dir = os.path.join(os.getcwd(), "ray_results")
    tuner = tune.Tuner(ray_trainer,
                       tune_config=tune.TuneConfig(
                           metric="ptl/val_accuracy",
                           mode="max",
                           scheduler=scheduler,
                           num_samples=num_samples,

                       ),
                       run_config=ray.train.RunConfig(
                           name=exp_name,
                           verbose=2,
                           storage_path=result_dir,
                           log_to_file=True,

                           checkpoint_config=ray.train.CheckpointConfig(
                               num_to_keep=3,
                               checkpoint_score_attribute="ptl/val_accuracy",
                               checkpoint_score_order="max",
                           ),
                       ),
                       param_space={"train_loop_config": hpo_config},
                       )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


# VANILLA PYTORCH WITH TUNE. MULTI CPU, 1 GPU

def tune_function(num_samples=10, num_epochs=10, exp_name="tune_transform"):
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    train_fn_with_parameters = tune.with_parameters(objective_func, data_dir=os.path.join(os.getcwd(), "testing_data"))
    gpus_per_trial = 2 if torch.cuda.is_available() else 0
    resources_per_trial = {"cpu": 2,}

    result_dir = os.path.join(os.getcwd(), "ray_results")
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,

        ),
        run_config=air.RunConfig(
            name=exp_name,
            verbose=2,
            storage_path=result_dir,
            log_to_file=True,
        ),
        param_space=hpo_config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


def objective_func(config,data_dir='data' ):
    print(f"data_dir-------> {data_dir}")
    dm = get_datamodule(task_name="sentilex", model_name_or_path=config['model_name_or_path'],
                        max_seq_length=config['max_seq_length'],
                        train_batch_size=config['per_device_train_batch_size'],
                        eval_batch_size=config['per_device_eval_batch_size'],
                        data_dir=data_dir)
    dm.setup("fit")
    model = AshaTransformer(config=config, num_labels=dm.task_metadata['num_labels'], dataset_name="omp")
    tune_callback = TuneReportCallback(
        {
            "loss": "ptl/val_loss",
            "mean_accuracy": "ptl/val_accuracy"
        },
        on="validation_end")
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        limit_train_batches=10,

        # If fractional GPUs passed in, convert to int.
        devices='auto',
        accelerator='auto',
        num_nodes=1,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="localTransforn", version="."),
        enable_progress_bar=True,
        callbacks=[tune_callback])
    trainer.fit(model, dm)
    print(f"trainer metrics: {trainer.callback_metrics}")
    print("Finished obj")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune on MultiNode")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", action="store_false", help="Finish quickly for testing")
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
    parser.add_argument("--exp-name", type=str, default="tune_transform")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    #
    # if args.server_address:
    #     context = ray.init(f"ray://{args.server_address}")
    # elif args.ray_address:
    #     context = ray.init(address=args.ray_address)
    # elif args.smoke_test:
    #     context = ray.init()
    # else:
    #     context = ray.init(address='auto', _redis_password=os.environ['redis_password'])
    # print("Dashboard URL: http://{}".format(context.dashboard_url))

    parser = argparse.ArgumentParser(description="Tune on local")
    parser.add_argument(
        "--smoke-test", action="store_false", help="Finish quickly for testing")  # store_false will default to True
    parser.add_argument("--exp-name", type=str, default="local_tune_transform")
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        print("No GPU available")
        args.exp_name = "local_tune_transform_cpu"

    sample_config = {"model_name_or_path": "bert-base-uncased",
                     "optimizer_name": "AdamW",
                     "scheduler_name": "linear",
                     "learning_rate": 1e-5,
                     "weight_decay": 1e-5,
                     "adam_epsilon": 1e-8,
                     "warmup_steps": 0,
                     "per_device_train_batch_size": 16,
                     "per_device_eval_batch_size": 16,
                     "gradient_accumulation_steps": 1,
                     "num_train_epochs": 1,
                     "max_steps": -1,
                     "max_grad_norm": 0.0,
                     "seed": 42,
                     "max_seq_length": 128,
                     "num_epochs": 2, }

    # objective_func(sample_config)
    # objective_torch_trainer(sample_config)

    # Start training
    if args.smoke_test:
        print("Smoketesting...")
        tune_function(num_samples=2, num_epochs=4, exp_name='torch_trainer')
    else:
        tune_function(num_samples=15, num_epochs=4, exp_name='torch_trainer')
