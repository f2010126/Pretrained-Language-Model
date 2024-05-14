import argparse
import logging
import os
import threading
import time
import traceback
import yaml

import torch
from lightning.pytorch import Trainer, seed_everything

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import trange
import sys
sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))

from BoHBCode.data_modules import get_datamodule
from BoHBCode.train_module import PLMTransformer

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16


def train_single_config(config, task_name='gnad10', budget=1, data_dir='./cleaned_datasets', train=True):
    eval_dir = os.path.join(os.getcwd(), 'Evaluations')
    # os.makedirs(data_dir, exist_ok=True)
    log_dir = os.path.join(eval_dir, 'Logs')
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(eval_dir, 'Models')
    os.makedirs(model_dir, exist_ok=True)

    print("budget aka epochs------> {}".format(budget))
    if torch.cuda.is_available():
        logging.debug("CUDA available, using GPU no. of device: {}".format(
            torch.cuda.device_count()))
    else:
        logging.debug("CUDA not available, using CPU")

    seed_everything(9)

    # set up data and model
    # data_dir should be the location of the tokenised dataset
    dm = get_datamodule(task_name=task_name, model_name_or_path=config['model_config']['model'],
                        max_seq_length=config['model_config']['dataset']['seq_length'],
                        train_batch_size=config['model_config']['dataset']['batch'],
                        eval_batch_size=config['model_config']['dataset']['batch'], data_dir=data_dir)
    dm.setup("fit")
    model_config = {'model_name_or_path': config['model_config']['model'],
                    'optimizer_name':config['model_config']['optimizer']['type'],
                    'learning_rate': config['model_config']['optimizer']['lr'],
                    'scheduler_name': config['model_config']['optimizer']['scheduler'],
                    'weight_decay': config['model_config']['optimizer']['weight_decay'],
                    'sgd_momentum': config['model_config']['optimizer']['momentum'],
                    'warmup_steps': config['model_config']['training']['warmup'],
                    }
    model = PLMTransformer(
        config=model_config, 
        num_labels=dm.task_metadata['num_labels'],)
    # wandb_logger = WandbLogger(name=f"{task_name}_single_config", project="BoHB", log_model=False, offline=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, save_top_k=1, monitor="val_acc_epoch")
    # set up trainer
    trainer = Trainer(
        max_epochs=int(budget),
        accelerator='cpu', devices=1,
       # devices="auto",
        #strategy="ddp_spawn",  # change to ddp_spawn when in interactive mode
        logger=False,
        # logger=[TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version="."),
        #         CSVLogger(save_dir=log_dir, name="csv_logs", version="."),
                # wandb_logger
                # ],
        # max_time="00:1:00:00",  # give each run a time limit

        log_every_n_steps=10,
        limit_train_batches=0.1,
        # limit_test_batches=0.1,
        # limit_val_batches=0.2,
         val_check_interval=10,
        callbacks=[checkpoint_callback],

        accumulate_grad_batches=config['model_config']['training']['gradient_accumulation'],
    )
    if train:
        try:
            start = time.time()
            trainer.fit(model, datamodule=dm)
            print(f"Training completed for {task_name} epochs {trainer.default_root_dir}")
        except Exception as e:
            print(f"Exception in training: with config {config['incumbent_for']} and task {task_name} ")
            print(e)
            traceback.print_exc()
        
        end = time.time() - start
        print(f"Time taken for {task_name} epochs: {end}")
        print(f"Return values available: {trainer.callback_metrics}")
        return {"end_time": end, "metrics": trainer.callback_metrics, "budget": budget, }

    try:
        # return {"end_time": 0, "metrics": {'test_f1_epoch': 0.40}, "budget": budget, }
        start = time.time()
        result = trainer.test(ckpt_path='best',datamodule=dm) if train else trainer.test(model=model, datamodule=dm)
        # convert result to dictionary
        print(result)
        end = time.time() - start
        print(f"Time taken for {task_name} epochs: {end}")
        return {"end_time": end, "metrics": result[0], "budget": budget, }
    except Exception as e:
        print(f"Exception in training: with config {config['incumbent_for']} and task {task_name} ")
        # print(e)
        # traceback.print_exc()
        return {"end_time": 0, "metrics": {'test_f1_epoch': 0.0}, "budget": budget, }

def evaluate(args):
    sample_config = {'adam_epsilon': 7.648065011196061e-08,
                     'gradient_accumulation_steps': 8,
                     'gradient_clip_algorithm': 'norm',
                     'learning_rate': 2.8307701958512803e-05,
                     'max_grad_norm': 1.9125507303302376, 'max_seq_length': 128,
                     'model_name_or_path': 'deepset/bert-base-german-cased-oldvocab',
                     'optimizer_name': 'SGD', 'per_device_eval_batch_size': 8,
                     'per_device_train_batch_size': 4, 'scheduler_name': 'cosine_with_warmup',
                     'warmup_steps': 500, 'weight_decay': 8.372735952480551e-05, 'sgd_momentum': 0.12143549900084782}
    
    sample_config = {"adam_epsilon": 1.2372243448105274e-07,
                     "gradient_accumulation_steps": 16,
                     "learning_rate": 3.277577722487855e-05,
                     "max_seq_length": 128,
                     "model_name_or_path": "dbmdz/distilbert-base-german-europeana-cased",
                     "optimizer_name": "Adam",
                     "per_device_train_batch_size": 4,
                     "scheduler_name": "cosine_with_warmup",
                     "warmup_steps": 10,
                     "weight_decay": 0.00011557671486497145,
                     
                     'gradient_clip_algorithm': 'norm', 'max_grad_norm': 1.9125507303302376, 
                     'per_device_eval_batch_size': 8,
                     }
    output = []
    for name in ['mtop_domain', 'tyqiangz', 'omp',"cardiff_multi_sentiment","swiss_judgment_prediction","hatecheck-german","german_argument_mining","tagesschau", 'gnad10']:
        config_file=os.path.join(f'/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/IncumbentConfigs/{name}_incumbent.yaml')
        try:
            with open(config_file) as in_stream:
                metadata = yaml.safe_load(in_stream)
                sample_config = metadata
        except Exception as e:
            print(f"problem loading config file {config_file}")
            print(e)
            traceback.print_exc()

        print(f"Training for {name}")
        output.append(train_single_config(
        config=sample_config, task_name=name, budget=1, train=False))
    print(output)

    # write to file
    lock = threading.Lock()
    os.makedirs(os.path.join(os.getcwd(), 'SingleConfig',), exist_ok=True)
    output_file = os.path.join(
        os.getcwd(), 'SingleConfig', f'{args.model_name}_meta_dataset_time.txt')
    # create output file

    # open file for appendin
    with lock:
        with open(output_file, 'w') as file:
            # write text to data
            file.write(str(output))

def run_aug():
    config={
    'seed': 42,
    'incumbent_for': 'miam_1X_10Labels',
    'model_config': {
        'model': 'bert-base-uncased',
        'optimizer': {
            'type': 'RAdam',
            'lr': 6.146670783169018e-05,
            'momentum': 0.9,
            'scheduler': 'cosine_with_warmup',
            'weight_decay': 6.265835646508776e-05,
            'adam_epsilon': 8.739737941142407e-08
        },
        'training': {
            'warmup': 100,
            'gradient_accumulation': 4
        },
        'dataset': {
            'name': 'miam_1X_10Labels',
            'seq_length': 512,
            'batch': 8,
            'num_training_samples': 9935,
            'average_text_length': 6.091092098641168,
            'num_labels': 10
        }
    },
    'aug': True,
    'run_info': [],}

    data_dir="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/tokenized_data/Augmented/miam_1X_10Labels"

    return_val=train_single_config(config, task_name='augmented', budget=1, data_dir=data_dir, train=True)
    print(return_val)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Single Config')
    parser.add_argument('--model-name', type=str,
                        default="bert-base-german-cased", help='Task name')
    parser.add_argument('--budget', type=int, default=1, help='Budget')

    args = parser.parse_args()
    # evaluate(args)
    run_aug()

# GNAD10 batch size 8 takes 115 s per epoch
# MTOP batch size 8 takes 264 s per epoch
# Cardiff batch size 8 takes 18 s per epoch
# Sentilex batch size 8 takes 34 s per epoch
# OMP batch size 8 takes 785 s per epoch
# Tyqiangz batch size 8 takes 19 s per epoch
# Amazon batch size 8 takes 3600 s per epoch
