from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
import uuid
import numpy as np
from smac import Scenario
from smac.facade import HyperparameterOptimizationFacade
from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import torch

# Lightning
import pytorch_lightning as pl
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger

iris = datasets.load_iris()


class LightningMNISTClassifier(pl.LightningModule):
    """
    This has been adapted from PL's MNIST example
    """

    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

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

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss, sync_dist=True)
        self.log("ptl/train_accuracy", accuracy, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        result = {"val_loss": loss, "val_accuracy": accuracy}
        self.val_output_list.append(result)
        return {"val_loss": loss, "val_accuracy": accuracy}

    # https://github.com/Lightning-AI/lightning/pull/16520
    def on_validation_epoch_end(self):
        outputs = self.val_output_list
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        with FileLock(os.path.expanduser("~/.data.lock")):
            return MNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)
        # no attribute assignments here.

    def setup(self, stage=None):
        mnist_train = self.download_data(self.data_dir)

        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size))

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_mnist(config: Configuration, seed: int = 0):
    data_dir = "~/data"
    if torch.cuda.is_available():
        print(f'Number of devices: {torch.cuda.device_count()}')
    else:
        print('No GPU available.')

    data_dir = os.path.expanduser(data_dir)
    model = LightningMNISTClassifier(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        # If fractional GPUs passed in, convert to int. API changed.
        devices='auto',
        accelerator="auto",
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=False)
    trainer.fit(model)
    acc = trainer.callback_metrics["ptl/val_accuracy"].item()
    print(f'Accuracy returned----> {acc}')
    return 1 - acc


def train(config: Configuration, seed: int = 0) -> float:
    print(f'Config---> {config}')
    if torch.cuda.is_available():
        print(f'Number of devices: {torch.cuda.device_count()}')
    else:
        print('No GPU available.')
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    print(f'Cross validation scores: {scores}')
    return 1 - np.mean(scores)


if __name__ == "__main__":
    og_configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

    configspace = ConfigurationSpace(name='model_space',
                                     seed=42,
                                     space={"layer_1_size": Categorical("layer_1_size", [32, 64, 128]),
                                            "layer_2_size": Categorical("layer_2_size", [64, 128, 256]),
                                            "lr": Float("lr", bounds=(1e-4, 1e-1), log=True),
                                            "batch_size": Categorical("batch_size", [32, 64, 128]),
                                            "epochs": Integer("epochs", bounds=(2, 5)),
                                            }
                                     )

    # Scenario object specifying the optimization environment
    # random identifier
    id = uuid.uuid4()
    scenario = Scenario(configspace,
                        deterministic=True,
                        # trial_walltime_limit=300.0,
                        walltime_limit=3000,
                        n_trials=8,
                        n_workers=1,
                        output_directory=f'smac_output_{id}', )

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train_mnist)
    incumbent = smac.optimize()
    print(f'Incumbent----> {incumbent}')
