import argparse
import os
import random
import string
from typing import List

import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from ray import air, tune
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
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import logging
import sys

# local imports
try:
    from BoHBCode.data_modules import OmpData, get_datamodule
    from BoHBCode.train_module import AshaTransformer
except ImportError:
    from .BoHBCode.data_modules import OmpData, get_datamodule
    from .BoHBCode.train_module import AshaTransformer
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['ptl/val_accuracy']} for {trial.trainable_name} with config {trial.config}")

    def on_trial_error(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(f"Got error for {trial.trainable_name} with config {trial.config}")


# class RayTrainReportCallback(Callback):
#     """A simple callback that reports checkpoints to Ray on train epoch end."""
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.trial_name = train.get_context().get_trial_name()
#         self.local_rank = train.get_context().get_local_rank()
#         self.tmpdir_prefix = os.path.join(tempfile.gettempdir(), self.trial_name)
#         if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
#             shutil.rmtree(self.tmpdir_prefix)
#
#         record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")
#
#     def on_train_epoch_end(self, trainer, pl_module) -> None:
#         # Creates a checkpoint dir with fixed name
#         tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
#         os.makedirs(tmpdir, exist_ok=True)
#
#         # Fetch metrics
#         metrics = trainer.callback_metrics
#         metrics = {k: v.item() for k, v in metrics.items()}
#
#         # (Optional) Add customized metrics
#         metrics["epoch"] = trainer.current_epoch
#         metrics["step"] = trainer.global_step
#
#         # Save checkpoint to local
#         ckpt_path = os.path.join(tmpdir, "checkpoint.ckpt")
#         trainer.save_checkpoint(ckpt_path, weights_only=False)
#
#         # Report to train session
#         checkpoint = Checkpoint.from_directory(tmpdir)
#         train.report(metrics=metrics, checkpoint=checkpoint)
#
#         if self.local_rank == 0:
#             shutil.rmtree(tmpdir)

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
    'num_train_epochs': tune.choice([2, 3, 4]),
    'max_steps': tune.choice([-1, 100, 1000]),
    'max_grad_norm': tune.choice([0.0, 1.0, 2.0]),
    'seed': tune.choice([42, 1234, 2021]),
    'max_seq_length': tune.choice([128, 256, 512]),
    "num_epochs": tune.choice([2, 3, 4]),
}


# https://github.com/ray-project/ray/issues/37394
# RAY TUNE 3.0 WITH TORCH TRAINER

def objective_torch_trainer(config, data_dir=os.path.join(os.getcwd(), "testing_data")):
    logging.debug(f"config-------> {config} in dir {data_dir} and cwd {os.getcwd()}")
    dm = get_datamodule(task_name="sentilex", model_name_or_path=config['model_name_or_path'],
                        max_seq_length=config['max_seq_length'],
                        train_batch_size=config['per_device_train_batch_size'],
                        eval_batch_size=config['per_device_eval_batch_size'], data_dir=data_dir)
    dm.setup("fit")
    model = AshaTransformer(config=config, num_labels=dm.task_metadata['num_labels'])
    ckpt_report_callback = RayTrainReportCallback()
    log_dir = os.path.join(os.getcwd(), "ray_results_log/torch_trainer_logs")
    print(f"log_dir-----> {log_dir}")
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        logger=[CSVLogger(save_dir=log_dir, name="csv_torch_trainer_logs", version="."),
                TensorBoardLogger(save_dir=log_dir, name="tensorboard_torch_trainer_logs", version=".")],

        # If fractional GPUs passed in, convert to int.
        devices='auto',
        accelerator='auto',
        enable_progress_bar=True,
        max_time="00:12:00:00",  # give each run a time limit
        val_check_interval=0.5,  # check validation set 4 times during a training epoch
        limit_train_batches=5,
        limit_val_batches=5,
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[ckpt_report_callback])

    # Validate your Lightning trainer configuration
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)
    print(f"trainer metrics: {trainer.callback_metrics}")
    print("Finished obj")


def trial_dir_name(trial):
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    return "{}_{}_{}".format(trial.trainable_name, trial.trial_id, x)


def tune_func_torch_trainer(num_samples=10, num_epochs=10, exp_name="torch_transform"):
    # hpo_config
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=1,
                              reduction_factor=2)

    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_worker = 8
        # each Trainer gets that.
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_worker, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
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
    train_fn_with_parameters = tune.with_parameters(objective_torch_trainer,
                                                    data_dir=os.path.join(os.getcwd(), "testing_data"))
    ray_trainer = TorchTrainer(
        train_fn_with_parameters,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        )
    )

    result_dir = os.path.join(os.getcwd(), "ray_results_result")
    logging.debug(f"result_dir-----> {result_dir}")
    restore_path = os.path.join(result_dir, exp_name)
    tuner = tune.Tuner(ray_trainer,
                       tune_config=tune.TuneConfig(
                           metric="ptl/val_accuracy",
                           mode="max",
                           scheduler=scheduler,
                           num_samples=num_samples,
                           trial_name_creator=trial_dir_name,
                           trial_dirname_creator=trial_dir_name,
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


def debug_torch(num_samples=10, num_epochs=10, exp_name="debug_torch"):
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=1,
                              reduction_factor=2)

    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_worker = 4
        # each Trainer gets that.
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_worker, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
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
    train_fn_with_parameters = tune.with_parameters(objective_torch_trainer,
                                                    data_dir=os.path.join(os.getcwd(), "debug_testing_data"))
    ray_trainer = TorchTrainer(
        train_fn_with_parameters,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        )
    )

    result_dir = os.path.join(os.getcwd(), "ray_results_result")
    print(f"result_dir-----> {result_dir}")
    # Fault Tolerance Code
    storage_path = os.path.expanduser("~/ray_results")
    exp_name = "tune_fault_tolerance_guide"
    restore_path = os.path.join(result_dir, exp_name)

    if tune.Tuner.can_restore(restore_path):
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
                               metric="ptl/val_accuracy",
                               mode="max",
                               scheduler=scheduler,
                               num_samples=num_samples,
                               trial_name_creator=trial_dir_name,
                               trial_dirname_creator=trial_dir_name,
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


def debug_bohb(num_samples=10, num_epochs=10, exp_name="debug_bohb"):
    max_iterations = 3
    # scheduler
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations,  # max time per trial? max length of time a trial
        reduction_factor=2,  # cut down trials by factor of?
        stop_last_trials=True,
        # Whether to terminate the trials after reaching max_t. Will this clash wth num_epochs of trainer
    )

    # search Algo
    bohb_search = TuneBOHB(
        # space=config_space,  # If you want to set the space manually
    )
    # Number of parallel runs allowed
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_worker = 4
        # each Trainer gets that.
        scaling_config = ray.train.ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_worker, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
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
    train_fn_with_parameters = tune.with_parameters(objective_torch_trainer,
                                                    data_dir=os.path.join(os.getcwd(), "debug_testing_data"))
    ray_trainer = TorchTrainer(
        train_fn_with_parameters,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        )
    )

    result_dir = os.path.join(os.getcwd(), "ray_results_result")
    # Fault Tolerance Code
    exp_name = "bohb_tune_fault_tolerance_guide"
    restore_path = os.path.join(result_dir, exp_name)
    print(f"result_dir-----> {result_dir}")
    if tune.Tuner.can_restore(restore_path):
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
                               metric="ptl/val_accuracy",
                               mode="max",
                               scheduler=bohb_hyperband,
                               search_alg=bohb_search,
                               reuse_actors=False,
                               num_samples=num_samples,
                               # trial_name_creator=trial_dir_name,
                               # trial_dirname_creator=trial_dir_name,
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
    resources_per_trial = {"cpu": 2, }
    # saved correctly
    result_dir = os.path.join(os.getcwd(), "ray_results_vanilla")
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


def objective_func(config, data_dir='data'):
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
    log_dir = os.path.join(os.getcwd(), "ray_results_log/torch_trainer_logs")
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        # If fractional GPUs passed in, convert to int.
        devices='auto',
        accelerator='auto',
        num_nodes=1,
        logger=TensorBoardLogger(
            save_dir=log_dir),
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
        "--smoke-test", action="store_true", help="Finish quickly for testing")
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
    args = parse_args()

    if args.server_address:
        context = ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        context = ray.init(address=args.ray_address)
    elif args.smoke_test:
        context = ray.init()
    else:
        context = ray.init(address='auto', _redis_password=os.environ['redis_password'])
    print("Dashboard URL: http://{}".format(context.dashboard_url))

    # parser = argparse.ArgumentParser(description="Tune on local")
    # parser.add_argument(
    #     "--smoke-test", action="store_false", help="Finish quickly for testing")  # store_false will default to True
    # parser.add_argument("--exp-name", type=str, default="local_tune_transform")
    # args, _ = parser.parse_known_args()

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

    # debug_torch(num_samples=10, num_epochs=4, exp_name="debug_torch_transform")

    # Start training Main
    if args.smoke_test:
        print("Smoketesting...")
        tune_func_torch_trainer(num_samples=2, num_epochs=4, exp_name=args.exp_name)
    else:
        tune_func_torch_trainer(num_samples=150, num_epochs=4, exp_name=args.exp_name)


    # if args.smoke_test:
    #     print("Smoketesting...")
    #     tune_function(num_samples=2, num_epochs=4, exp_name=args.exp_name)
    # else:
    #     tune_function(num_samples=15, num_epochs=4, exp_name=args.exp_name)
