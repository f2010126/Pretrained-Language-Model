from datasets import load_dataset
from transformers import RobertaTokenizer, BertTokenizer
from transformers import BertConfig, BertModelWithHeads
import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
import evaluate
import wandb
import os
import argparse



# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Adapter Training")

    parser.add_argument("--model",
                        default='bert-base-uncased',
                        type=str,
                        help="Model Name")

    parser.add_argument("--task_name",
                        default='SST-2',
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

    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--project_name", type=str, help="Name of WANDDB project.", default="Adapters")
    parser.add_argument("--group_name", type=str, help="Name of WANDDB group.", default="BERT")
    parser.add_argument("--run_name", type=str, help="Name of WANDDB run.", default="Bert-Adapter-Training")

    return parser.parse_args()

# Training Function
def train(args, run=None):
    dataset = load_dataset("glue", 'sst2')
    model_name = args.model
    print(f'Model Name: {model_name}')

    tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(batch["sentence"], max_length=args.max_seq_length, truncation=True, padding="max_length")

    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_train_test= dataset['train'].train_test_split(test_size=0.3)

    config = BertConfig.from_pretrained(
        model_name,
        num_labels=2,
    )
    model = BertModelWithHeads.from_pretrained(
        model_name,
        config=config,
    )

    # Add a new adapter
    model.add_adapter("glue")
    # Add a matching classification head
    model.add_classification_head(
        "glue",
        num_labels=2,
        id2label={0: "👎", 1: "👍"}
    )
    # Activate the adapter
    model.train_adapter("glue")

    # Training
    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size_gpu, # x # of GPUS for total batch size
        per_device_eval_batch_size=args.eval_batch_size_gpu, # x # of GPUS for total batch size
        logging_steps=200,
        evaluation_strategy="steps",
        output_dir="./adapter_training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        report_to = 'wandb',
        eval_steps=500, # Evaluate every 5000 steps
        save_steps=1000, # Save checkpoint every 10000 steps
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        label_names=["0", "1"],
        run_name=args.run_name # Name of W&B run
    )

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(trainer.evaluate())
    # Evaluation
    metric = evaluate.load("accuracy")
    predictions = trainer.predict(dataset["validation"])
    preds = np.argmax(predictions.predictions, axis=-1)
    print(f'Final-->{metric.compute(predictions=preds, references=predictions.label_ids)}')






if __name__ == "__main__":
    args = parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    run = wandb.init(entity="insane_gupta",
                     project="Adapters",
                     name="bert-base-adapters",
                     tags=["adapter", "6eps"],
                     group="bert")

    train(args,run)
