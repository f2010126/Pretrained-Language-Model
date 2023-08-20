"""
This example is uses the official
huggingface transformers `hyperparameter_search` API.
"""
import os
from ray import tune
from ray.air.config import CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)
import torch
import time
import traceback
import ray

os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU_NUM'

class CustomTrainer(Trainer):
    pass

def tune_transformer(num_samples=8, gpus_per_trial=0, exp_name='', smoke_test=False):
    data_dir_name = "./data" if not smoke_test else "./test_data"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, 0o755)

    # Change these as needed.
    model_name = (
        "bert-base-uncased" if not smoke_test else "sshleifer/tiny-distilroberta-base"
    )
    # Dataset related
    task_name = "rte"

    task_data_dir = os.path.join(data_dir, task_name.upper())

    num_labels = glue_tasks_num_labels[task_name]

    config = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task=task_name
    )

    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Triggers tokenizer download to cache
    print("Downloading and caching pre-trained model")
    AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

    # Download data.
    download_data(task_name, data_dir)

    data_args = GlueDataTrainingArguments(task_name=task_name, data_dir=task_data_dir)

    train_dataset = GlueDataset(
        data_args, tokenizer=tokenizer, mode="train", cache_dir=task_data_dir
    )
    eval_dataset = GlueDataset(
        data_args, tokenizer=tokenizer, mode="dev", cache_dir=task_data_dir
    )
    # Starting args
    training_args = TrainingArguments(
        output_dir=".",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(task_name),
    )

    # Vary This
    tune_config = {
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 10 if smoke_test else -1,  # Used for smoke test.
    }
    # Change this
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [16, 32, 64],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
    )
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        accelerator = 'gpu'
        use_gpu = True
        gpus_per_trial = n_devices
    else:
        n_devices = 0
        accelerator = 'cpu'
        use_gpu = False
        gpus_per_trial = 0
    print(f"Number of devices per trial-----> {gpus_per_trial}")
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="training_iteration",
    )
    try:
        start = time.time()
        best_trial = trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            backend="ray",
            n_trials=num_samples,
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            scheduler=scheduler,
            keep_checkpoints_num=1,
            stop={"eval_acc": .76, } if smoke_test else {"eval_acc": .86, },
            progress_reporter=reporter,
            storage_path="./ray_results/",
            name="tune_transformer_pbt",
            log_to_file=True,
            fail_fast="raise",
        )
        print(f"Objective: {best_trial.objective} Best trial config: {best_trial.hyperparameters}")
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    except ray.exceptions.RayTaskError:
        print("User function raised an exception!")
    except Exception as e:
        print("Other error", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_false", help="Finish quickly for testing"
    )
    parser.add_argument("--exp-name", type=str, default="tune_hf")
    args, _ = parser.parse_known_args()

    if args.smoke_test:
        print("Smoke test mode")
        tune_transformer(num_samples=3, gpus_per_trial=0, smoke_test=False, exp_name=args.exp_name)
    else:
        # You can change the number of GPUs here:
        tune_transformer(num_samples=8, gpus_per_trial=2, exp_name=args.exp_name)
