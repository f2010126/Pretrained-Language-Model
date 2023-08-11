import os
import tempfile
from ray import air, tune
import pytorch_lightning as pl
import torch
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import wandb
from pytorch_lightning.loggers import WandbLogger
import ray
import argparse

# local changes
import uuid
import socket

class DataModuleMNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = ''
        self.batch_size = 32
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataworkers = os.cpu_count()/2

    def prepare_data(self):
        datasets.MNIST(self.download_dir,
                       train=True, download=True)

        datasets.MNIST(self.download_dir, train=False,
                       download=True)

    def setup(self, stage=None):
        data = datasets.MNIST(self.download_dir,
                              train=True, transform=self.transform)

        self.train_data, self.valid_data = random_split(data, [55000, 5000])

        self.test_data = datasets.MNIST(self.download_dir,
                                        train=False, transform=self.transform)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        result = {"val_loss": loss, "val_accuracy": accuracy}
        self.val_output_list.append(result)
        self.log("ptl/val_loss", loss)
        self.log("ptl/val_accuracy", accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack(
    #         [x["val_loss"] for x in outputs]).mean()
    #     avg_acc = torch.stack(
    #         [x["val_accuracy"] for x in outputs]).mean()
    #     self.log("ptl/val_loss", avg_loss)
    #     self.log("ptl/val_accuracy", avg_acc)

    def on_validation_epoch_end(self):
        outputs = self.val_output_list
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def on_fit_end(self):
        print("fit end")


def train_mnist(config, data_dir=None, num_epochs=2):
    print("Running in IP ---> ", socket.gethostbyname(socket.gethostname()))
    if torch.cuda.is_available():
        print(f"GPU is available { torch.cuda.device_count()}")
    else:
        print("GPU is not available")
    wandb_logger = WandbLogger(project="datasetname",
                         log_model=True,
                         name=f"{config['batch_size']}_modelname",
                         entity="insane_gupta",group='modelname',)
    model = LightningMNISTClassifier(config, data_dir)
    dm = DataModuleMNIST()
    # Create the Tune Reporting Callback
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    n_devices = torch.cuda.device_count()
    accelerator = 'cpu' if n_devices == 0 else 'auto'

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices='auto', strategy='auto',  # Use whatver device is available
        enable_progress_bar=True,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)
    wandb.finish()

# BOHB LOOP

def parse_args():
    parser = argparse.ArgumentParser(description="Bohb on MultiNode")
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
    args, _ = parser.parse_known_args()
    return args

def mnist_bohb(smoke_test=False):
    if torch.cuda.is_available():
        print(f"GPU is available { torch.cuda.device_count()}")
    else:
        print("GPU is not available")
    ## BOHB CONFIGURATION
    num_epochs = 10
    gpus_per_trial = 2  # set this to higher if using GPU
    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    # Download data

    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    resources_per_trial = {"cpu": 2, "gpu": gpus_per_trial}

    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir, # params for the train_mnist function
        num_epochs=num_epochs,) # change this?

    # set all the resources to be used by BOHB here.
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    resource_trainable = tune.with_resources(trainable, resources=resources_per_trial)

    max_iterations = 3 #bohb will stop after this many evaluations in the given bracket/round
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations,
        reduction_factor=2,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB(
        # space=config_space,  # If you want to set the space manually
    )
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)
    tuner = tune.Tuner(
        resource_trainable,
        run_config=air.RunConfig(
            name="bohb_test", stop={"training_iteration": 2 if smoke_test else max_iterations,"acc": 0.99}
        ),
        tune_config=tune.TuneConfig(
            metric="acc",
            mode="max",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=2, # No of trials to be run/ No of brackets.
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == "__main__":
    args = parse_args()

    # os.environ['RAY_ADDRESS'] = '127.0.0.1:6379'

    if args.server_address:
        context = ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        context = ray.init(address=args.ray_address)
    elif args.smoke_test:
        context = ray.init(num_cpus=2)
    else:
        context = ray.init(address='auto', _redis_password=os.environ['redis_password'])
    print("Dashboard URL: http://{}".format(context.dashboard_url))
    mnist_bohb(smoke_test=args.smoke_test)

