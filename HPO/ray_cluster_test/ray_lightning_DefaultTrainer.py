
import argparse
import torch
from filelock import FileLock
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
import os
from torch.utils.data import random_split, DataLoader
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torchvision import datasets, transforms
from ray import air, tune
import ray

class DataModuleMNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = ''
        self.batch_size = 32
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataworkers = os.cpu_count() / 2
        self.prepare_data_per_node = True

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
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
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
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
        return {"val_loss": avg_loss, "val_accuracy": avg_acc}

    def on_fit_end(self):
        print("---------- Finished training")


def train_mnist_tune(config, num_epochs=10, num_gpus=0):

    pl.seed_everything(0)
    if torch.cuda.is_available():
        print(f" No of GPUs available : {torch.cuda.device_count()}")
    else:
        print("No GPU available")
    data_dir = os.path.abspath("./data")
    model = LightningMNISTClassifier(config, data_dir)
    with FileLock(os.path.expanduser("~/.data.lock")):
        dm = DataModuleMNIST()
    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
    n_devices = torch.cuda.device_count()
    accelerator = 'cpu' if n_devices == 0 else 'auto'
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        devices="auto",
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=[TuneReportCallback(["val_loss", "val_accuracy" ], on="validation_end")],
    )
    trainer.fit(model, dm)



def tune_mnist(num_samples=10, num_epochs=10, gpus_per_trial=0,exp_name="tune_mnist"):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    trainable = tune.with_parameters(
        train_mnist_tune, num_epochs=num_epochs, num_gpus=gpus_per_trial
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="acc",
            mode="max",
            num_samples=num_samples,
            time_budget_s=200,
            reuse_actors=True,

        ),
        run_config=air.RunConfig(
            name=exp_name,
            verbose=2,
            stop={"training_iteration": 1},
        ),
        param_space=config,
    )
    try:
        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)
    except ray.exceptions.RayTaskError:
        print("User function raised an exception!")
    except Exception as e:
        print("Other error", e)

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

    # Start training
    if args.smoke_test:
        print("Smoketesting...")
        train_mnist_tune(config={"layer_1": 32, "layer_2": 64, "lr": 1e-3, "batch_size": 64},
                         num_epochs=1, num_gpus=0)
        tune_mnist(num_samples=2, num_epochs=1, gpus_per_trial=2, exp_name=args.exp_name)
    else:
        tune_mnist(num_samples=10, num_epochs=10, gpus_per_trial=0,exp_name=args.exp_name)
