import logging
from typing import Optional

import evaluate
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim import Adam, AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_inverse_sqrt_schedule



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
            hyperparameters: Optional[dict] = None,
            **kwargs,
    ):
        super().__init__()
        # access validation outputs, save them in-memory as instance attributes
        self.validation_step_outputs = []

        self.task = 'binary' if num_labels == 2 else 'multiclass'
        self.hyperparams = hyperparameters

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)

        self.config = AutoConfig.from_pretrained(hyperparameters['model_name_or_path'], num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(hyperparameters['model_name_or_path'],
                                                                        config=self.config)
        # self.metric = evaluate.load(
        #     "glue", self.hparams.task_name, experiment_id=datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # )
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)
        self.optimizer_name = hyperparameters['optimizer_name']
        self.scheduler_name = hyperparameters['scheduler_name']
        self.train_acc = evaluate.load('accuracy')
        self.train_f1 = evaluate.load('f1')
        self.train_bal_acc = evaluate.load('hyperml/balanced_accuracy')

        self.prepare_data_per_node = True

    def forward(self, **inputs):
        return self.model(**inputs)

    def setup(self, stage: str):
        print(f"Setup stage: {stage} rank: {self.trainer.global_rank}, world_size: {self.trainer.world_size}")

    def evaluate_step(self, batch, batch_idx, stage='val'):
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
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True, on_step=True)
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0, print_str="val"):
        return self.evaluate_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluate_step(batch, batch_idx, stage='test')

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

        if self.optimizer_name == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparams['learning_rate'],
                              eps=self.hparams.adam_epsilon)
        elif self.optimizer_name == "Adam":
            optimizer = Adam(optimizer_grouped_parameters, lr=self.hyperparams['learning_rate'],
                             eps=self.hparams.adam_epsilon)
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.hyperparams['learning_rate'],
                                        momentum=0.9)
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer_name}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        print(f'Load the optimizer {self.optimizer_name} and scheduler {self.scheduler_name}')
        return [optimizer], [scheduler]


class PLMTransformer(LightningModule):
    def __init__(
            self,
            config,
            num_labels: int,
            **kwargs,
    ):
        super().__init__()

        # access validation outputs, save them in-memory as instance attributes
        self.validation_step_outputs = []

        self.task = 'binary' if num_labels == 2 else 'multiclass'
        self.config = config

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)

        self.model_config = AutoConfig.from_pretrained(config['model_name_or_path'], num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name_or_path'],
                                                                        config=self.model_config)
        # self.metric = evaluate.load(
        #     "glue", self.hparams.task_name, experiment_id=datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # )
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)
        self.optimizer_name = config['optimizer_name']
        self.scheduler_name = config['scheduler_name']
        self.train_acc = evaluate.load('accuracy')
        self.train_f1 = evaluate.load('f1')
        self.train_bal_acc = evaluate.load('hyperml/balanced_accuracy')

        self.prepare_data_per_node = True

    # Training
    def forward(self, **inputs):
        return self.model(**inputs)

    def on_fit_start(self) -> None:
        pass

    def evaluate_step(self, batch, batch_idx, stage='val'):
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        # calculate pred
        labels = batch["labels"]

        acc = self.accuracy(preds, labels)

        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True, on_step=True)
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True, on_step=True)
        return {f"loss": loss, f"accuracy": acc}

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0, print_str="val"):
        result = self.evaluate_step(batch, batch_idx, stage='val')
        self.validation_step_outputs.append({"val_loss": result["loss"], "val_accuracy": result["accuracy"]})
        return result

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluate_step(batch, batch_idx, stage='test')

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        self.log("ptl/val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        logging.debug("on_validation_epoch_end--->")
        return {"loss": avg_loss, "acc": avg_acc}

    def on_validation_end(self):
        # last hook that's used by Trainer in ray.
        logging.debug("on_validation_end")

    # Optimizers
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer_name == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                              eps=self.config['adam_epsilon'])
        elif self.optimizer_name == "Adam":
            optimizer = Adam(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                             eps=self.config['adam_epsilon'])
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                                        momentum=self.config['sgd_momentum'])
        elif self.optimizer_name == "RAdam":
            optimizer = torch.optim.RAdam(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                                          eps=self.config['adam_epsilon'])
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer_name}")

        if self.scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_name == "cosine_with_warmup":
            scheduler= get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_name == "inverse_sqrt":
            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_name == "constant_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
            )
        elif self.scheduler_name == "polynomial_decay_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_name == "cosine_with_hard_restarts_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # data
