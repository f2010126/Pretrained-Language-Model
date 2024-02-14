# Add an introduction to the file as a multi-line comment
"""
Contains training functions to be used with Ray Train. BOHB would also use these functions to train models and evaluate them.
"""
import tempfile
import torch
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint
import ray
import traceback
import sys

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from lightning.pytorch import seed_everything
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
import lightning.pytorch as pl
import uuid
import os   

try:
    from data_modules import get_datamodule
    from train_module import PLMTransformer
except ImportError:
    from .data_modules import get_datamodule
    from .train_module import PLMTransformer



def transformer_train_function(config):
    print("budget aka epochs------> {}".format(config['epochs']))
    seed_everything(config['seed'])

     # [4] Build your datasets on each worker
    dm = get_datamodule(task_name=config['task'], model_name_or_path=config['model_name_or_path'],
                            max_seq_length=config['max_seq_length'],
                            train_batch_size=config['per_device_train_batch_size'],
                            eval_batch_size=config['per_device_train_batch_size'], data_dir=config['data_dir'])
    # per_device_eval_batch_size

    model =PLMTransformer(config=config, num_labels=dm.task_metadata['num_labels'])
    ckpt_report_callback = RayTrainReportCallback()
    # set up logger
    folder_name = config['model_name_or_path'].split("/")[-1]  # last part is usually the model name
    log_dir = os.path.join(config["log"], f"{config['run_id']}_logs/{folder_name}/run_{config['trial_id']}")
    # create dir for current trial
    os.makedirs(log_dir, exist_ok=True)

    dm.setup("fit")

    trainer = pl.Trainer(
        max_epochs=int(config['epochs']),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=20,
        val_check_interval=0.25,
        logger=pl.loggers.TensorBoardLogger(log_dir, name="", version=""),

        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[ckpt_report_callback],
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        
        )
    
    # Validate your Lightning trainer configuration
    trainer = prepare_trainer(trainer)
    try:
        trainer.fit(model, datamodule=dm)
    except Exception:
        print(traceback.format_exc())
        

def train_func(config):
    print(f'config---->: {config}')
    # Model, Loss, Optimizer
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # [1] Prepare model.
    model = ray.train.torch.prepare_model(model)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Data
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    # [2] Prepare dataloader.
    print("Data loading----->")
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    print("Data loaded----->")
    # Training
    for epoch in range(config['epochs']):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        checkpoint_dir = tempfile.gettempdir()
        checkpoint_path = checkpoint_dir + "/model.checkpoint"
        torch.save(model.state_dict(), checkpoint_path)
        # [3] Report metrics and checkpoint.
        ray.train.report({"loss": loss.item()}, checkpoint=Checkpoint.from_directory(checkpoint_dir))
   

def trainf_hf_func(config):
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    small_train_dataset = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
    small_eval_dataset = dataset["test"].select(range(1000)).map(tokenize_function, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    # Evaluation Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch", report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # [2] Report Metrics and Checkpoints to Ray Train
    # ===============================================
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

    # [3] Prepare Transformers Trainer
    # ================================
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    # [4] Configure scaling and resource requirements.
    test_config={'epochs': 1, 'batch_size': 64, 'lr': 0.001,}
    test_config={'epochs': 1, 'batch_size': 64, 'learning_rate': 0.001, 
                 'max_seq_length': 128, 'per_device_train_batch_size': 8,
                # 'per_device_eval_batch_size': 8, 
                'data_dir': '.', 
                'task': "mtop_domain", 
                'model_name_or_path': 'bert-base-uncased', 
                'log': '.',
                'weight_decay': 0.01, 'warmup_steps': 500,
                 'adam_epsilon': 1e-08,
                 'optimizer_name': 'AdamW', 'scheduler_name': 'cosine_with_warmup',
                 'gradient_accumulation_steps':2,
                 'run_id': 'test_run',
                 'seed': 42,
                'trial_id': str(uuid.uuid4().hex)[:5] }

    # where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), "ddp_debug", test_config['run_id'])
    os.makedirs(working_dir, exist_ok=True)
    test_config['log'] = working_dir
    # central location for the datasets
    test_config['data_dir'] = os.path.join(os.getcwd(), "tokenized_data")
    os.makedirs(test_config['data_dir'], exist_ok=True)

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True,resources_per_worker={"CPU": 2, "GPU": 1})

    # [5] Launch distributed training job.
    trainer = TorchTrainer(transformer_train_function, scaling_config=scaling_config, train_loop_config=test_config)
    result = trainer.fit()
    print(result.metrics)


    # Time per epoch 2 gpu. DO not run two at a time
    # gnad10 3 min
    # tyqiangz 2 min
    # "omp" 11 min
    # "sentilex"  2 min
    # "cardiff_multi_sentiment" 2
    # "mtop_domain" 5 min