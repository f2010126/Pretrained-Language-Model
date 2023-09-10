import os
import tempfile
from ray import air, tune
import pytorch_lightning as pl
import torch
from ray import tune
from torch.nn import functional as F
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
import traceback

from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

import ray
import argparse
from ray.tune import CLIReporter
from ray.train.lightning import LightningConfigBuilder, LightningTrainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune import Callback
from typing import List
# local changes
import socket


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['ptl/val_accuracy']} for {trial.trainable_name} with config {trial.config}")

    def on_trial_error(
            self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        print(f"Got error for {trial.trainable_name} with config {trial.config}")


class DataModuleMNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = ''
        self.batch_size = 32
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataworkers = os.cpu_count() / 2

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
                                        train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2)


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()
        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

        self.val_output_list = []

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

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

    def on_fit_start(self):
        print("Running in IP ---> ", socket.gethostbyname(socket.gethostname()))
        if torch.cuda.is_available():
            print(f"GPU is available {torch.cuda.device_count()}")
        else:
            print("GPU is not available")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        result = {"val_loss": loss, "val_accuracy": accuracy}
        self.val_output_list.append(result)
        self.log("ptl/val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/val_accuracy", accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self, ):
        outputs = self.val_output_list
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_validation_end(self):
        # last hook that's used by Trainer.
        print(f"---------- Finished validation?---->{self.trainer.logged_metrics}")


# BOHB LOOP
def air_bohb(smoke_test=False, gpus_per_trial=0, exp_name='bohb_mnist'):
    # Static configs that does not change across trials
    # Details:
    print(f'Running this on {ray._private.services.get_node_ip_address()}')
    dm = DataModuleMNIST()
    # doesn't call prepare data
    logger = TensorBoardLogger(save_dir=f'./ray_results/{exp_name}', name="tensorboard_files", version=".")
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        accelerator = 'cuda'
        use_gpu = True
        gpus_per_trial = 8
    else:
        n_devices = 0
        accelerator = 'cpu'
        use_gpu = False
    print(f" No of GPUs available : {torch.cuda.device_count()} and accelerator is {accelerator}")

    num_epochs = 10 if smoke_test else 9
    static_lightning_config = (
        LightningConfigBuilder()
        .module(cls=LightningMNISTClassifier)
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger)
        .fit_params(datamodule=dm)
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
    )

    max_iterations = 5 if smoke_test else 20  # each trial will stop after 5 iterations max
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

    scaling_config = ScalingConfig(
        # no of other nodes?
        num_workers=1, use_gpu=use_gpu, resources_per_worker={"CPU": 2, "GPU": gpus_per_trial}
    )

    lightning_trainer = LightningTrainer(
        lightning_config=static_lightning_config,
        scaling_config=scaling_config,
    )
    reporter = CLIReporter(
        parameter_columns=["layer_1", "layer_2", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": searchable_lightning_config},
        tune_config=tune.TuneConfig(
            time_budget_s=7500,  # Max time for the whole search
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=10 if smoke_test else 15,  # Number of times to sample from the config space
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            reuse_actors=True,

        ),
        run_config=air.RunConfig(
            name=exp_name,
            verbose=2,
            storage_path="./ray_results",
            log_to_file=True,
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
            callbacks=[MyCallback()],
            # progress_reporter=reporter,
        ),

    )

    try:
        results = tuner.fit()
        best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
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
    # parser.add_argument("--exp-name", type=str, default="tune_bohb")
    # args, _ = parser.parse_known_args()

    if args.smoke_test:
        print("Running smoke test")
        air_bohb(smoke_test=args.smoke_test, gpus_per_trial=0, exp_name=args.exp_name)

    else:
        air_bohb(smoke_test=args.smoke_test, gpus_per_trial=0, exp_name=args.exp_name)

    print("END")