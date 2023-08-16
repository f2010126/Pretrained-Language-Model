from pytorch_lightning import seed_everything, Trainer
from datamodule_glue import GLUEDataModule, GlueModule, getDataModule
from GlueLightningModule import GLUETransformer
from pytorch_lightning.loggers import WandbLogger
import argparse
import wandb
import torchmetrics.functional as F
import time
import torch
import yaml
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# should only get hpo config, and data dir.
def train_model(args, config=None):
    hyperparameters = config['hyperparameters']

    seed_everything(args.seed)

    # set up data loaders
    dm = getDataModule(task_name=config['task_name'], model_name_or_path=hyperparameters['model_name_or_path'],
                       max_seq_length=hyperparameters['max_seq_length'],
                       train_batch_size=hyperparameters['train_batch_size_gpu'],
                       eval_batch_size=hyperparameters['eval_batch_size_gpu'])
    dm.setup("fit")
    # set up model and experiment
    model = GLUETransformer(
        model_name_or_path=hyperparameters['model_name_or_path'],
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=hyperparameters['learning_rate'],
        adam_epsilon=1e-8,
        warmup_steps=0,
        weight_decay=0.0,
        train_batch_size=hyperparameters['train_batch_size_gpu'],
        eval_batch_size=hyperparameters['eval_batch_size_gpu'],
        hyperparameters=hyperparameters,
    )

    # set up logger
    wandb.init()
    wandb_logger = WandbLogger(entity="insane_gupta",
                               project=config['project_name'],  # group runs in "MNIST" project
                               name=config['run_name'],  # individual runs within project
                               tags=["bert", "pytorch-lightning"],
                               group=config['group_name'],
                               log_model='all')  # log all new checkpoints during training

    # set up trainer
    n_devices = torch.cuda.device_count()
    accelerator = 'cpu' if n_devices == 0 else 'auto'
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=hyperparameters['num_train_epochs'],
        accelerator=accelerator,
        devices='auto', strategy='auto',  # Use whatver device is available
        max_steps=10, limit_val_batches=5, limit_test_batches=5, num_sanity_val_steps=0,
        # max_steps=20 and no sanity check
        val_check_interval=5, check_val_every_n_epoch=1,  # check_val_every_n_epoch=1 and every 5 batches
    )
    # train model
    print("Training model")
    trainer.fit(model, datamodule=dm)
    print("Best checkpoint path: ", trainer.checkpoint_callback.best_model_path)
    # evaluate best model
    trainer.test(model, dataloaders=dm.test_dataloader())
    test_acc = trainer.logged_metrics
    print(f"Test accuracy: {test_acc}")
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Adapter Training")

    parser.add_argument("--model",
                        default='bert-base-uncased',
                        type=str,
                        help="Model Name")

    parser.add_argument("--task_name",
                        default='sst2',
                        type=str,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir",
                        default='./adapter_training_output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Tokenizer
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size_gpu",
                        default=32,
                        type=int,
                        help="Per GPU batch size for training.")

    parser.add_argument("--eval_batch_size_gpu",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=3e-3,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--optimizer_name",
                        default='AdamW',
                        type=str,
                        help="Optimiser name default AdamW.")

    parser.add_argument("--scheduler_name",
                        default='linear',
                        type=str,
                        help="Scheduler name default linear.")

    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=int,
                        help="Total number of training epochs to perform.")
    # Logging
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Seed used for training and evaluation.")
    parser.add_argument("--project_name", type=str, help="Name of WANDDB project.", default="HPO")
    parser.add_argument("--group_name", type=str, help="Name of WANDDB group.", default="Amazon")  # Dataset name
    parser.add_argument("--run_name", type=str, help="Name of WANDDB run.",
                        default="Bert-Adapter-Training")  # shoudld be model name

    parser.add_argument("--yaml_config", type=str, help="Path to yaml config file.",
                        default="ray_cluster_test/BoHBCode/yaml_config/default.yaml")

    return parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    try:
        with open(args.yaml_config, 'r') as stream:
            config = load(stream, Loader=Loader)
    except IOError:
        print("Could not read YAML file. Using default configs.")
        config = vars(args)

    train_model(args=args, config=config)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
