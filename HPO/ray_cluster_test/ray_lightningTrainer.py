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


class DataModuleMNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = ''
        self.batch_size = 32
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

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
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


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


def train_mnist(config, data_dir=None, num_epochs=10, num_gpus=0):
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

# BOHB LOOP
def mnist_bohb():
    num_epochs = 10
    gpus_per_trial = 0  # set this to higher if using GPU
    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    # Download data

    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    # set all the resources to be used by BOHB here.

    max_iterations = 3 #bohb will stop after this many iterations
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
        trainable,
        run_config=air.RunConfig(
            name="bohb_test", stop={"training_iteration": max_iterations,"acc": 0.99}
        ),
        tune_config=tune.TuneConfig(
            metric="acc",
            mode="max",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=5, # No of trials to be run
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == "__main__":
    mnist_bohb()
