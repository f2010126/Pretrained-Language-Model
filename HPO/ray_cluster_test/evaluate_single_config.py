import argparse
import logging
import os
import threading
import time
import traceback

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from BoHBCode.data_modules import get_datamodule
from BoHBCode.train_module import PLMTransformer

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16


def train_single_config(config, task_name='gnad10', budget=1):
    data_dir = os.path.join(os.getcwd(), 'SingleConfig')
    os.makedirs(data_dir, exist_ok=True)
    log_dir = os.path.join(data_dir, 'Logs')

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

    trainer = Trainer(
        max_epochs=int(budget),
        accelerator="auto",
        num_nodes=1,
        devices="auto",
        strategy="ddp",  # change to ddp_spawn when in interactive mode
        logger=[TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs", version="."),
                CSVLogger(save_dir=log_dir, name="csv_logs", version=".")],
        max_time="00:1:00:00",  # give each run a time limit

        log_every_n_steps=10,
        val_check_interval=10,
        enable_checkpointing=False,

        accumulate_grad_batches=config['gradient_accumulation_steps'],
        gradient_clip_val=config['max_grad_norm'],
        gradient_clip_algorithm=config['gradient_clip_algorithm'],
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
    return_val = 1 - trainer.callback_metrics['metrics/val_accuracy'].item()
    print(f"1-val to return: {return_val} and other values available: {trainer.callback_metrics}")
    return {"end_time": end}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Single Config')
    parser.add_argument('--model-name', type=str, default="bert-base-german-cased", help='Task name')
    parser.add_argument('--budget', type=int, default=1, help='Budget')

    args = parser.parse_args()
    sample_config = {'adam_epsilon': 8.739737941142407e-08, 'gradient_accumulation_steps': 16,
                     'gradient_clip_algorithm': 'norm', 'learning_rate': 6.146670783169018e-05,
                     'max_grad_norm': 1.263152537528855, 'max_seq_length': 512,
                     'model_name_or_path': args.model_name,
                     'optimizer_name': 'RAdam', 'per_device_eval_batch_size': 8, 'per_device_train_batch_size': 8,
                     'scheduler_name': 'cosine_with_warmup', 'warmup_steps': 100,
                     'weight_decay': 6.265835646508776e-05}

    # output=[]
    # for task in ['gnad10',"mtop_domain","cardiff_multi_sentiment","sentilex","omp","tyqiangz","amazon_reviews_multi"]:
    #     print(f"Running task {task} time")
    #     output.append(train_single_config(config=sample_config, task_name=task,budget=10))
    # print(output)
    output = []
    output.append(train_single_config(config=sample_config, task_name="mtop_domain", budget=1000))
    print(output)

    # write to file
    lock = threading.Lock()
    output_file = os.path.join(os.getcwd(), 'SingleConfig', 'dataset_time.txt')
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
