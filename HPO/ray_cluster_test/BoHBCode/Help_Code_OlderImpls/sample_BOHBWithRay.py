"""
Implementation of using Ray Train with BOHB for hyperparameter optimization.
"""
from calendar import c
from logging import config
import tempfile
import torch
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

import time
import os
import pickle
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

import argparse
from hpbandster.optimizers import BOHB as BOHB
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from filelock import FileLock
from torchmetrics import Accuracy
import tempfile
import traceback
from torchvision import transforms

import ray
try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    raise ImportError("For this example you need to install pytorch-vision.")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import pytorch_lightning as pl
import logging
import torchmetrics

from bohb_ray import fake_ray_train_function

logging.basicConfig(level=logging.DEBUG)

class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.layer_1_size = 128
        self.layer_2_size = 256
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
    
    def on_fit_start(self) -> None:
        print(f'Number of GPUs before fit -------> {torch.cuda.device_count()} World is {self.trainer.world_size} Strategy is {self.trainer._accelerator_connector.strategy}')

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

# each distributed worker uses this
def train_func():
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # [1] Prepare model.
    model = ray.train.torch.prepare_model(model)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    mape = torchmetrics.MeanAbsolutePercentageError()
    # for averaging loss
    mean_valid_loss = torchmetrics.MeanMetric()

        # Data
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    # [2] Prepare dataloader.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)


    # Training
    for epoch in range(config["num_epochs"]):
        for images, labels in train_loader:
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_valid_loss(loss)
            mape(outputs, labels)

        # [3] Report metrics
        mape_collected = mape.compute().item()
        mean_valid_loss_collected = mean_valid_loss.compute().item()

        ray.train.report({"loss": loss.item(),      
                        "mape_collected": mape_collected,
                        "mean_valid_loss_collected": mean_valid_loss_collected,
                          },)
        # reset for next epoch
        mape.reset()
        mean_valid_loss.reset()


class LightningWorker(Worker):
    def __init__(self, sleep_interval=0.5, N_train = 8192, N_valid = 1024, **kwargs):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # context = ray.init(ignore_reinit_error=True)
        # print("Dashboard URL: http://{}".format(context.dashboard_url))
        
        # [4] Configure scaling and resource requirements.
        scaling_config = ScalingConfig(num_workers=2, use_gpu=False, resources_per_worker={"CPU": 2,})
        config["num_epochs"]=int(budget)

        # [5] Launch distributed training job.
        trainer = TorchTrainer(fake_ray_train_function, 
                               train_loop_config=config,
                               scaling_config=scaling_config)
        result = trainer.fit()
        print(result.metrics["loss"])

        val_acc=result.metrics["loss"]
        all_metric = result.metrics
        ray.shutdown()
        return ({
            'loss': 1 - val_acc,  # remember: HpBandSter always minimizes!
            'info': {'all_metrics': all_metric,
                     'worker':self.worker_id
                     }  })
    
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)
        cs.add_hyperparameters([lr, optimizer, sgd_momentum])
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=32, upper=256, default_value=128, log=True)
        cs.add_hyperparameter(batch_size)
        return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=9)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=243)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=1)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the '
                             'clusters scheduler.', default='RayBOHB')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='lo0')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.',default='ddp_debug')

    args = parser.parse_args()

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)
    # where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)
    print(f'Working dir is ----->{working_dir}')

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = LightningWorker(sleep_interval=0.5, run_id=args.run_id, host=host)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)
    

    # ensure the file is empty, init the config and results json
    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=True)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    w = LightningWorker(sleep_interval=0.5, run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    try:
        previous_run = hpres.logged_results_to_HBS_result(working_dir)
    except Exception:
        print('No prev run')
        previous_run = None

    # Run an optimizer
    # We now have to specify the host, and the nameserver information
    bohb = BOHB(configspace=LightningWorker.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget, max_budget=args.max_budget,
                previous_result=previous_run,
                result_logger=result_logger,
                )
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

    # In a cluster environment, you usually want to store the results for later analysis.
    # One option is to simply pickle the Result object
    with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
