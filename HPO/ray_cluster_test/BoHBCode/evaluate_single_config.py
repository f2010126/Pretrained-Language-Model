import argparse
import logging
import os
import threading
import time
import traceback

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
    # reading from the incumbent yaml file
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
        config=model_config, num_labels=config['model_config']['dataset']['num_labels'])
    # wandb_logger = WandbLogger(name=f"{task_name}_single_config", project="BoHB", log_model=False, offline=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, save_top_k=1, monitor="val_acc_epoch")
    # set up trainer
    trainer = Trainer(
        max_epochs=int(budget),
        accelerator='cpu', devices=1,
       # devices="auto",
        #strategy="ddp_spawn",  # change to ddp_spawn when in interactive mode
        logger=[TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version="."),
                CSVLogger(save_dir=log_dir, name="csv_logs", version="."),
                # wandb_logger
                ],
        # max_time="00:1:00:00",  # give each run a time limit

        log_every_n_steps=10,
        # limit_train_batches=0.1,
        limit_test_batches=0.1,
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
            print(
                f"Exception in training: with config {config} and budget {budget}")
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
        print(result)
        end = time.time() - start
        print(f"Time taken for {task_name} epochs: {end}")
        return {"end_time": end, "metrics": result, "budget": budget, }
    except Exception as e:
        print(f"Exception in testing: with config {config} and budget {budget}")
        print(e)
        traceback.print_exc()
        return {"end_time": 0, "metrics": {'test_f1_epoch': 0.0}, "budget": budget, }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Single Config')
    parser.add_argument('--model-name', type=str,
                        default="bert-base-german-cased", help='Task name')
    parser.add_argument('--budget', type=int, default=1, help='Budget')

    args = parser.parse_args()
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

    # weght decay no major effect needs to be smaller order e-08

    # output=[]
    # for task in ['gnad10',"mtop_domain","cardiff_multi_sentiment","sentilex","omp","tyqiangz","amazon_reviews_multi"]:
    #     print(f"Running task {task} time")
    #     output.append(train_single_config(config=sample_config, task_name=task,budget=10))
    # print(output)
    output = []
    output.append(train_single_config(
        config=sample_config, task_name='cardiff_multi_sentiment', budget=5))
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

# GNAD10 batch size 8 takes 115 s per epoch
# MTOP batch size 8 takes 264 s per epoch
# Cardiff batch size 8 takes 18 s per epoch
# Sentilex batch size 8 takes 34 s per epoch
# OMP batch size 8 takes 785 s per epoch
# Tyqiangz batch size 8 takes 19 s per epoch
# Amazon batch size 8 takes 3600 s per epoch
