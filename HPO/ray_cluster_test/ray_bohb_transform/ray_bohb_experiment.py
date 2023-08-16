import os
import tempfile
from ray import air, tune
import pytorch_lightning as pl
import torch
from ray import tune
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.nn import functional as F
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
import traceback
import torchmetrics
import evaluate
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
from typing import List, Optional
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


class GLUETransformer(pl.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            task_name: str,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            optimizer_name: str = "AdamW",
            scheduler_name: str = "linear",
            hyperparameters: Optional[dict] = None,
            **kwargs,
    ):
        super().__init__()
        # access validation outputs, save them in-memory as instance attributes
        self.validation_step_outputs = []

        self.task = 'binary' if num_labels == 2 else 'multiclass'
        self.hyperparams = hyperparameters

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)

        self.config = AutoConfig.from_pretrained(hyperparameters['model_name_or_path'], num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(hyperparameters['model_name_or_path'], config=self.config)
        # self.metric = evaluate.load(
        #     "glue", self.hparams.task_name, experiment_id=datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # )
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)
        self.optimizer_name = hyperparameters['optimizer_name']
        self.scheduler_name = hyperparameters['scheduler_name']
        self.train_acc = evaluate.load( 'accuracy')
        self.train_f1 = evaluate.load( 'f1')
        self.train_bal_acc=evaluate.load( 'hyperml/balanced_accuracy')

        self.prepare_data_per_node = True

    def forward(self, **inputs):
        return self.model(**inputs)

    def evaluate_step(self, batch, batch_idx, stage='val'):
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        pass

        # calculate pred
        labels = batch["labels"]

        acc = self.accuracy(preds, labels)
        print(f'-----> {stage}_acc_step', acc)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True, on_step=True)
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0, print_str="val"):
        return self.evaluate_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluate_step(batch, batch_idx, stage='test')

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer_name == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparams['learning_rate'],
                              eps=self.hparams.adam_epsilon)
        elif self.optimizer_name == "Adam":
            optimizer = Adam(optimizer_grouped_parameters, lr=self.hyperparams['learning_rate'], eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        print(f'Load the optimizer {self.optimizer_name} and scheduler {self.scheduler_name}')
        return [optimizer], [scheduler]


# BOHB LOOP
def air_bohb(smoke_test=False, gpus_per_trial=0, exp_name='bohb_mnist'):

    # Static configs that does not change across trials
    dm = DataModuleMNIST()
    # doesn't call prepare data
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-ptl-example", version=".")
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_trial = 8
    else:
        n_devices = 0
        accelerator = 'cpu'
        use_gpu = False
    print(f" No of GPUs available : {torch.cuda.device_count()} and accelerator is {accelerator}")

    num_epochs = 10 if smoke_test else 100
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

    max_iterations = 5 if smoke_test else 100  # each trial will stop after 5 iterations max
    # scheduler
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations, # max time per trial? max length of time a trial
        reduction_factor=2, # cut down trials by factor of?
        stop_last_trials=True, #Whether to terminate the trials after reaching max_t. Will this clash wth num_epochs of trainer
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
            time_budget_s=750, # Max time for the whole search
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=3 if smoke_test else 100, # Number of times to sample
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
        "--smoke-test", action="store_false", help="Finish quickly for testing") # store_false will default to True
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
