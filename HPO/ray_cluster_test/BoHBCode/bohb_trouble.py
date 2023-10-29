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

from torchvision import transforms
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
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

logging.basicConfig(level=logging.DEBUG)

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


class PyTorchWorker(Worker):
    def __init__(self, sleep_interval=0.5, N_train=8192, N_valid=1024, **kwargs):
        super().__init__(**kwargs)

        batch_size = 64

        # Load the MNIST Data here
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train))
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train, N_train + N_valid))

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                        sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024,
                                                             sampler=validation_sampler)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        # device = torch.device('cpu')
        model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                             num_filters_1=config['num_filters_1'],
                             num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                             num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                             dropout_rate=config['dropout_rate'],
                             num_fc_units=config['num_fc_units'],
                             kernel_size=3
                             )

        criterion = torch.nn.CrossEntropyLoss()
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        for epoch in range(int(budget)):
            loss = 0
            model.train()
            for i, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()

        train_accuracy = self.evaluate_accuracy(model, self.train_loader)
        validation_accuracy = self.evaluate_accuracy(model, self.validation_loader)
        test_accuracy = self.evaluate_accuracy(model, self.test_loader)

        return ({
            'loss': 1 - validation_accuracy,  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': test_accuracy,
                     'train accuracy': train_accuracy,
                     'validation accuracy': validation_accuracy,
                     'number of parameters': model.number_of_parameters(),
                     }

        })

    def compute2(self, config, budget, working_directory, *args, **kwargs):


    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99,
                                                      log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, )

        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, log=True)

        cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])

        # You can also use inequality conditions:
        cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        cs.add_condition(cond)

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9,
                                                      log=False)
        num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, log=True)

        cs.add_hyperparameters([dropout_rate, num_fc_units])

        return cs


class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units,
                 kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=kernel_size)
        self.conv2 = None
        self.conv3 = None

        output_size = (28 - kernel_size + 1) // 2
        num_output_filters = num_filters_1

        if num_conv_layers > 1:
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size)
            num_output_filters = num_filters_2
            output_size = (output_size - kernel_size + 1) // 2

        if num_conv_layers > 2:
            self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size)
            num_output_filters = num_filters_3
            output_size = (output_size - kernel_size + 1) // 2

        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv_output_size = num_output_filters * output_size * output_size

        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 10)

    def forward(self, x):

        # switched order of pooling and relu compared to the original example
        # to make it identical to the keras worker
        # seems to also give better accuracies
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        if not self.conv2 is None:
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        if not self.conv3 is None:
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = self.dropout(x)

        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=9)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=243)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the '
                             'clusters scheduler.')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()
    args.run_id = 'TroubleMaker'

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    # where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), args.shared_directory, args.run_id)
    os.makedirs(working_dir, exist_ok=True)

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = PyTorchWorker(sleep_interval=0.5, run_id=args.run_id, host=host)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    w = PyTorchWorker(sleep_interval=0.5, run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    # Run an optimizer
    # We now have to specify the host, and the nameserver information
    bohb = BOHB(configspace=PyTorchWorker.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget, max_budget=args.max_budget
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
