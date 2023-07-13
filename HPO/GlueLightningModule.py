from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from typing import Optional
import torch
import datetime
import torchmetrics
import evaluate
from torch.optim import Adam, AdamW
class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        optimizer_name: str = "AdamW",
        scheduler_name: str = "linear",
        **kwargs,
    ):
        super().__init__()
        # access validation outputs, save them in-memory as instance attributes
        self.validation_step_outputs = []

        self.task = 'binary' if num_labels == 2 else 'multiclass'

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.accuracy= torchmetrics.Accuracy(task=self.task, num_classes=num_labels)
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.train_metric = evaluate.load("glue", task_name)

        self.prepare_data_per_node = True


    def forward(self, **inputs):
        return self.model(**inputs)


    def evaluate_step(self, batch, batch_idx,stage='val'):
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        pass

        # calculate pred
        labels = batch["labels"]

        acc = self.accuracy(preds, labels)
        self.log(f'{stage}_acc_step', acc, prog_bar=True, sync_dist=True, on_step=True)
        self.log(f'{stage}_loss_step', loss, prog_bar=True, sync_dist=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx,stage='train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0,print_str="val"):
        return self.evaluate_step(batch, batch_idx,stage='val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluate_step(batch, batch_idx,stage='test')

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer_name== "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        elif self.optimizer_name == "Adam":
            optimizer = Adam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        print(f'Load the optimizer {self.optimizer_name} and scheduler {self.scheduler_name}')
        return [optimizer], [scheduler]