
import argparse
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
import os
from torch.utils.data import random_split, DataLoader
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torchvision import datasets, transforms
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import ray
from ray.tune import CLIReporter
from ray.train.lightning import LightningConfigBuilder, LightningTrainer


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

    def teardown(self, stage: str):
        wandb.finish()


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

        wandb_obj = wandb.init(project="datasetname",
                               name=f"{config['batch_size']}_modelname",
                               entity="insane_gupta", group='modelname', )

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
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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

    def on_fit_end(self):
        print("---------- Finished training")

    def teardown(self, stage: str):
        wandb.finish()


def train_mnist_tune(config, num_epochs=10, num_gpus=0):

    pl.seed_everything(0)
    if torch.cuda.is_available():
        print(f" No of GPUs available : {torch.cuda.device_count()}")
    else:
        print("No GPU available")
    data_dir = os.path.abspath("./data")
    model = LightningMNISTClassifier(config, data_dir)
    dm = DataModuleMNIST()
    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
    n_devices = torch.cuda.device_count()
    accelerator = 'cpu' if n_devices == 0 else 'auto'
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        devices=num_gpus,
        accelerator="auto",
        strategy="ddp",
        enable_progress_bar=True,
        callbacks=[TuneReportCallback(['ptl/val_loss','ptl/val_accuracy'], on="validation_end")],
    )
    print("----->Starting training")
    trainer.fit(model, dm)
    print("Finished training")



def tune_mnist(num_samples=10, num_epochs=10, gpus_per_trial=0,exp_name="tune_mnist"):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    metric = "ptl/val_accuracy",
    mode = "max",

    # Static configs that does not change across trials
    dm = DataModuleMNIST()
    # doesn't call prepare data
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-ptl-example", version=".")
    n_devices = torch.cuda.device_count() or 0
    accelerator = 'cpu' if n_devices == 0 else 'gpu'
    print(f" No of GPUs available : {torch.cuda.device_count()} and accelerator is {accelerator}")

    wandb_logger = WandbLogger(project="datasetname",
                               log_model=True,
                               name=f"{config['batch_size']}_modelname",
                               entity="insane_gupta", group='modelname', )

    static_lightning_config = (
        LightningConfigBuilder()
        .module(cls=LightningMNISTClassifier)
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=wandb_logger)
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

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        # no of other nodes?
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": gpus_per_trial}
    )

    lightning_trainer = LightningTrainer(
        lightning_config=static_lightning_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    reporter = CLIReporter(
        parameter_columns=["layer_1", "layer_2", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": searchable_lightning_config},
        tune_config=tune.TuneConfig(
            time_budget_s=300,
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=air.RunConfig(
            name="tune_mnist_asha",
            verbose=2,
            progress_reporter=reporter,
        ),

    )


    try:
        results = tuner.fit()
        best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
        print("Best hyperparameters found were: ", best_result)
    except ray.exceptions.RayTaskError:
        print("User function raised an exception!")
    except Exception as e:
        print("Other error", e)

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
        "--smoke-test", action="store_true", help="Finish quickly for testing") # store_false will default to True
    parser.add_argument("--exp-name", type=str, default="tune_mnist")
    args, _ = parser.parse_known_args()

    # Start training
    if args.smoke_test:
        print("Smoketesting...")
        # train_mnist_tune(config={"layer_1": 32, "layer_2": 64, "lr": 1e-3, "batch_size": 64},
        #                  num_epochs=1, num_gpus=2)
        tune_mnist(num_samples=1, num_epochs=1, gpus_per_trial=0, exp_name=args.exp_name)
    else:
        tune_mnist(num_samples=10, num_epochs=10, gpus_per_trial=8,exp_name=args.exp_name)
