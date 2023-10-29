import argparse
import logging
import os
import threading
import time
import traceback

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from BoHBCode.data_modules import get_datamodule
from BoHBCode.train_module import PLMTransformer

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16


def train_single_config(config, task_name='gnad10', budget=1):
    data_dir = os.path.join(os.getcwd(), 'SingleConfig')
    os.makedirs(data_dir, exist_ok=True)
    log_dir = os.path.join(data_dir, 'Logs')
    model_dir = os.path.join(data_dir, 'Models')


    print("budget aka epochs------> {}".format(budget))
    if torch.cuda.is_available():
        logging.debug("CUDA available, using GPU no. of device: {}".format(torch.cuda.device_count()))
    else:
        logging.debug("CUDA not available, using CPU")

    seed_everything(9)

    # set up data and model
    dm = get_datamodule(task_name=task_name, model_name_or_path=config['model_name_or_path'],
                        max_seq_length=config['max_seq_length'],
                        train_batch_size=config['per_device_train_batch_size'],
                        eval_batch_size=config['per_device_eval_batch_size'], data_dir=data_dir)
    dm.setup("fit")
    model = PLMTransformer(config=config, num_labels=dm.task_metadata['num_labels'])
    wandb_logger = WandbLogger(name=f"{task_name}_single_config", project="BoHB", log_model=False, offline=False)
    checkpoint_callback = ModelCheckpoint(dirpath=model_dir, save_top_k=1, monitor="val_acc_epoch")

    trainer = Trainer(
        max_epochs=int(budget),
        accelerator="cpu",
        num_nodes=1,
        devices="auto",
        strategy="ddp_spawn",  # change to ddp_spawn when in interactive mode
        logger=[TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version="."),
                CSVLogger(save_dir=log_dir, name="csv_logs", version="."),wandb_logger],
        #max_time="00:1:00:00",  # give each run a time limit

        log_every_n_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback],

        accumulate_grad_batches=config['gradient_accumulation_steps'],
    )
    # train model
    try:
        start = time.time()
        trainer.fit(model, datamodule=dm)
    except Exception as e:
        print(f"Exception in training: with config {config} and budget {budget}")
        print(e)
        traceback.print_exc()

    end = time.time() - start
    print(f"Time taken for {task_name} epochs: {end}")
    print(f"Return values available: {trainer.callback_metrics}")

    try:
        result = trainer.test(ckpt_path="best", datamodule=dm)
        print(result)
    except Exception as e:
        print(f"Exception in testing: with config {config} and budget {budget}")
        print(e)
        traceback.print_exc()

    return {"end_time": end, "metrics": trainer.callback_metrics, "budget": budget,}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Single Config')
    parser.add_argument('--model-name', type=str, default="bert-base-german-cased", help='Task name')
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

    # weght decay no major effect needs to be smaller order e-08

    # output=[]
    # for task in ['gnad10',"mtop_domain","cardiff_multi_sentiment","sentilex","omp","tyqiangz","amazon_reviews_multi"]:
    #     print(f"Running task {task} time")
    #     output.append(train_single_config(config=sample_config, task_name=task,budget=10))
    # print(output)
    output = []
    output.append(train_single_config(config=sample_config, task_name='gnad10', budget=5))
    print(output)

    # write to file
    lock = threading.Lock()
    output_file = os.path.join(os.getcwd(), 'SingleConfig', f'{args.model_name}_Amaz_dataset_time.txt')
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
