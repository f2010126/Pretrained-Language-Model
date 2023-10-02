import argparse
import os
import time
import traceback
from typing import List

import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import LightningConfigBuilder, LightningTrainer
from ray.tune import CLIReporter
from ray.tune import Callback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from ray.tune.experiment import Trial


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['ptl/val_accuracy']} for {trial.trainable_name} with config {trial.config}")

    def on_trial_error(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(f"Got error for {trial.trainable_name} with config {trial.config}")


class DataModuleMNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = './data'
        self.batch_size = 32
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataworkers = int(os.cpu_count() / 2)
        self.prepare_data_per_node = True

    def prepare_data(self):
        datasets.MNIST(self.download_dir,
                       train=True, download=True)

        datasets.MNIST(self.download_dir, train=False,
                       download=True)

    def setup(self, stage=None):
        data = datasets.MNIST(self.download_dir,
                              train=True, transform=self.transform, download=True)

        self.train_data, self.valid_data = random_split(data, [55000, 5000])

        self.test_data = datasets.MNIST(self.download_dir,
                                        train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.dataworkers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.dataworkers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.dataworkers)


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(LightningMNISTClassifier, self).__init__()

        self.lr = config["lr"]
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_output_list = []

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_fit_start(self):
        print("---------- Starting training")
        if torch.cuda.is_available():
            print(f" No of GPUs available : {torch.cuda.device_count()}")
        else:
            print("No GPU available")

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        result = {"val_loss": loss, "val_accuracy": acc}
        self.val_output_list.append(result)
        return {"val_loss": loss, "val_accuracy": acc}

    def on_validation_epoch_end(self):
        outputs = self.val_output_list
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        self.log("ptl/val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_accuracy", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": avg_loss, "acc": avg_acc}

    def on_validation_end(self):
        # last hook that's used by Trainer.
        print("---------- Finished validation?")


def tune_bohb(num_samples=10, num_epochs=10, exp_name="tune_bohb_mnist"):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    # Static configs that does not change across trials
    dm = DataModuleMNIST()
    # doesn't call prepare data
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-bohb-ptl-example", version=".")
    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_trial = 2
        scaling_config = ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_trial, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
        )
    else:
        accelerator = 'cpu'
        use_gpu = False
        scaling_config = ScalingConfig(
            # no of other nodes?
            num_workers=1, use_gpu=use_gpu, resources_per_worker={"CPU": 1, }
        )
    print(f" No of GPUs available : {torch.cuda.device_count()} and accelerator is {accelerator}")

    static_lightning_config = (
        LightningConfigBuilder()
        .module(cls=LightningMNISTClassifier)
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger, )
        .fit_params(datamodule=dm)
        # .strategy(name='ddp')
        .checkpointing(monitor="ptl/val_accuracy", save_top_k=2, mode="max")
        .build()
    )

    # Searchable configs across different trials
    searchable_lightning_config = (
        LightningConfigBuilder()
        .module(config={
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
        })
        .build()
    )

    # Make sure to also define an AIR CheckpointConfig here
    # to properly save checkpoints in AIR format.
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
        callbacks=[MyCallback()]
    )

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
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=6)

    lightning_trainer = LightningTrainer(
        lightning_config=static_lightning_config,
        scaling_config=scaling_config,
    )
    reporter = CLIReporter(
        parameter_columns=["layer_1", "layer_2", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])
    # Fault Tolerance
    result_dir = os.path.join(os.getcwd(), "ray_results")
    restore_path = os.path.join(result_dir, exp_name)

    if os.path.exists(restore_path):
        restore_config = {
            "lightning_config/_module_init_config/config/layer_1": tune.choice([32, 64, 128]),
            "lightning_config/_module_init_config/config/layer_2": tune.choice([64, 128, 256]),
            "lightning_config/_module_init_config/config/lr": tune.loguniform(1e-4, 1e-1),
            "lightning_config/_module_init_config/config/batch_size": tune.choice([32, 64, 128]),
        }

        print("Restoring from checkpoint---->")
        tuner = tune.Tuner.restore(restore_path,
                                   trainable=lightning_trainer,
                                   resume_unfinished=True,
                                   resume_errored=True,
                                   restart_errored=False,
                                   # param_space=restore_config,
                                   )
    else:
        print("Creating new tuner---->")
        tuner = tune.Tuner(
            lightning_trainer,
            param_space={"lightning_config": searchable_lightning_config},
            tune_config=tune.TuneConfig(  # for Tuner
                time_budget_s=3000,
                metric="ptl/val_accuracy",
                mode="max",
                num_samples=num_samples,  # Number of times to sample from the hyperparameter space
                scheduler=bohb_hyperband,
                search_alg=bohb_search,
                reuse_actors=False,
            ),
            run_config=air.RunConfig(  # for Tuner.run
                name=exp_name,
                verbose=2,
                storage_path=result_dir,
                log_to_file=True,
                # configs given to Tuner are used.
                # progress_reporter=reporter,
                checkpoint_config=CheckpointConfig(
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


def tune_mnist(num_samples=10, num_epochs=10, exp_name="tune_mnist"):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    # Static configs that does not change across trials
    dm = DataModuleMNIST()
    # doesn't call prepare data
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-ptl-example", version=".")
    if torch.cuda.is_available():
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_trial = 2
        scaling_config = ScalingConfig(
            # no of other nodes?
            num_workers=gpus_per_trial, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": 1}
        )
    else:
        accelerator = 'cpu'
        use_gpu = False
        scaling_config = ScalingConfig(
            # no of other nodes?
            num_workers=3, use_gpu=use_gpu, resources_per_worker={"CPU": 1}
        )
    print(f" No of GPUs available : {torch.cuda.device_count()} and accelerator is {accelerator}")

    static_lightning_config = (
        LightningConfigBuilder()
        .module(cls=LightningMNISTClassifier)
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger, )
        .fit_params(datamodule=dm)
        # .strategy(name='ddp')
        .checkpointing(monitor="ptl/val_accuracy", save_top_k=2, mode="max")
        .build()
    )

    # Searchable configs across different trials
    searchable_lightning_config = (
        LightningConfigBuilder()
        .module(config={
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
        })
        .build()
    )

    # Make sure to also define an AIR CheckpointConfig here
    # to properly save checkpoints in AIR format.
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
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
    reporter = CLIReporter(
        parameter_columns=["layer_1", "layer_2", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])
    # Fault Tolerance
    result_dir = os.path.join(os.getcwd(), "ray_results")
    restore_path = os.path.join(result_dir, exp_name)

    if os.path.exists(restore_path):
        restore_config = {
            "lightning_config/_module_init_config/config/layer_1": tune.choice([32, 64, 128]),
            "lightning_config/_module_init_config/config/layer_2": tune.choice([64, 128, 256]),
            "lightning_config/_module_init_config/config/lr": tune.loguniform(1e-4, 1e-1),
            "lightning_config/_module_init_config/config/batch_size": tune.choice([32, 64, 128]),
        }

        print("Restoring from checkpoint---->")
        tuner = tune.Tuner.restore(restore_path,
                                   trainable=lightning_trainer,
                                   resume_unfinished=True,
                                   resume_errored=True,
                                   restart_errored=False,
                                   param_space=restore_config,
                                   )
    else:
        print("Creating new tuner---->")
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
                storage_path=result_dir,
                log_to_file=True,
                # configs given to Tuner are used.
                # progress_reporter=reporter,
                checkpoint_config=CheckpointConfig(
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
    parser.add_argument("--exp-name", type=str, default="tune_mnist")
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
    parser.add_argument("--exp-name", type=str, default="tune_mnist")
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        args.exp_name = "bohb13" + "_cpu"
        args.num_gpu = 0

    # Start training
    if args.smoke_test:
        print("Smoketesting...")
        # train_mnist_tune(config={"layer_1": 32, "layer_2": 64, "lr": 1e-3, "batch_size": 64},
        #                  num_epochs=1, num_gpus=2)
        tune_bohb(num_samples=13, num_epochs=2, exp_name=args.exp_name)
    else:
        tune_bohb(num_samples=15, num_epochs=3, exp_name=args.exp_name)
