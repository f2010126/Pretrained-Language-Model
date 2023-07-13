from datasets import load_dataset
from transformers import RobertaTokenizer, BertTokenizer
from transformers import RobertaConfig, RobertaModelWithHeads, BertConfig, BertModelWithHeads,AutoModelForSequenceClassification
import numpy as np
import evaluate

from transformers import TrainingArguments, Trainer

def train():

    # Data and encoding
    dataset = load_dataset("glue", 'sst2')
    print(dataset.num_rows)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(batch["sentence"], max_length=256, truncation=True, padding="max_length")

    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Trainer
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir="./finetune_output",
                                      evaluation_strategy="steps",
                                      learning_rate=2e-05,
                                      num_train_epochs=3,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      logging_steps=200,
                                      overwrite_output_dir=True,
                                      # The next line is important to ensure the dataset labels are properly passed to the model
                                      remove_unused_columns=False, )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluation
    metric = evaluate.load("glue", "sst2")
    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    print(f'Final-->{metric.compute(predictions=preds, references=predictions.label_ids)}')




if __name__ == "__main__":
    train()