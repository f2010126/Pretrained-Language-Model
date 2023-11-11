import os
from distributed import progress
from regex import P
import torch
import tempfile
import lightning.pytorch as pl
from ray.tune import Callback
from ray.tune.experiment import Trial
from typing import List
import torch.nn.functional as F
from filelock import FileLock
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from ray.train.lightning import (
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
RayDDPStrategy
)
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bohb import TuneBOHB
from pytorch_lightning.strategies import DDPStrategy
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import ray

from ray.tune import CLIReporter


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)
        self.eval_loss = []
        self.eval_accuracy = []

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.data_dir = tempfile.mkdtemp()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


default_config = {
    "layer_1_size": 128,
    "layer_2_size": 256,
    "lr": 1e-3,
}

def train_func(config):
    dm = MNISTDataModule(batch_size=config["batch_size"])
    model = MNISTClassifier(config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


def tune_mnist_asha(num_samples=10):
    search_space = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64]),
    }
    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = 10

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    if torch.cuda.is_available():
            scaling_config = ScalingConfig(
        num_workers=2, use_gpu=True, resources_per_worker={"CPU": 1,"GPU": 1},
        )
    else:
        scaling_config = ScalingConfig(
        num_workers=2, use_gpu=False, resources_per_worker={"CPU": 1},
        )
    print(f'Using scaling config: {scaling_config}')
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    reportercli = CLIReporter(
        parameter_columns={"layer_1_size":"L1", "batch_size":"BS", "lr":"LR"},
        metric_columns=["ptl/val_accuracy", "training_iteration"],
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        run_config=ray.train.RunConfig(
            progress_reporter=reportercli,
        ),
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            
        ),
    )
    return tuner.fit()


def run_sample(num_samples=10):
    results = tune_mnist_asha(num_samples=num_samples)


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['ptl/val_accuracy']} for {trial.trainable_name} with config {trial.config}")

    def on_trial_error(
            self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        print(f"Got error for {trial.trainable_name} with config {trial.config}")

def tune_mnist_bohb(num_samples=10):
    search_space = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64]),
    }
    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = num_samples

    if torch.cuda.is_available():
            scaling_config = ScalingConfig(
        num_workers=2, use_gpu=True, resources_per_worker={"CPU": 1,"GPU": 1},
        )
    else:
        scaling_config = ScalingConfig(
        num_workers=2, use_gpu=False, resources_per_worker={"CPU": 1},
        )
    print(f'Using scaling config: {scaling_config}')
    
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )
    result_dir = os.path.join(os.getcwd(), "ray_results_result")
    max_iterations = 7
    # scheduler
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations,  # max time per trial? max length of time a trial
        reduction_factor=2,  # cut down trials by factor of? eta here
        stop_last_trials=True,
        # Whether to terminate the trials after reaching max_t. Will this clash wth num_epochs of trainer
    )
    # options for bohb optimiser
    config_bohb = {'top_n_percent': 15, 'num_samples': 20,
                   'random_fraction': 1 / 3, 'bandwidth_factor': 3, 'min_bandwidth': 1e-3, }
    # search Algo
    bohb_search = TuneBOHB(bohb_config=config_bohb,)
    # Number of parallel runs allowed
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=3)

    reportercli = CLIReporter(
        parameter_columns={"layer_1_size":"L1", "batch_size":"BS", "lr":"LR"},
        metric_columns=["ptl/val_accuracy", "training_iteration"],
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        run_config=ray.train.RunConfig(
            verbose=None,
            name='mnist_bohb',
            storage_path=result_dir,
            log_to_file=True,
            progress_reporter=reportercli,
            callbacks=[MyCallback()],
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        ),
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
    print(f'Best result: {best_result}')
    

if __name__ == "__main__":
    tune_mnist_bohb(3)
    #run_sample(5)
