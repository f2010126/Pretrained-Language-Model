from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import datasets
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
from pathlib import Path


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "mrpc",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)



    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs

        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='longest', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


class GlueModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "mrpc",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.prepare_data_per_node = True
        self.n_cpu = 2

    def prepare_data(self):
        # called only on 1 GPU per node. Do not do any self.x assignments here
        # download, split tokenise data and save to disk
        print(f'Download and Tokenise')
        dataset = datasets.load_dataset("glue", self.task_name)

        # split the train set into train and test
        spare_data = dataset['train'].train_test_split(test_size=0.025)
        dataset['train'] = spare_data['train']
        dataset['test'] = spare_data['test']

        for split in dataset.keys():
            dataset[split] = dataset[split].map(self.encode_batch, batched=True)
            dataset[split] = dataset[split].rename_column("label", "labels")
            # Transform to pytorch tensors and only output the required columns
            # self.dataset[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            columns = [c for c in dataset[split].column_names if c in self.loader_columns]
            dataset[split].set_format(type="torch", columns=columns)

        # save the tokenized data to disk
        torch.save(dataset, 'tokenized_data.pt')
        print(f'Dataset Tokenised')

    def setup(self, stage: str):
        # load data here
        self.dataset = torch.load('tokenized_data.pt')
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        print('dataset loaded')

    def encode_batch(self, batch):
        """Encodes a batch of input data using the model tokenizer."""
        return self.tokenizer(batch["sentence"], max_length=self.max_seq_length, truncation=True, padding="max_length")

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.n_cpu)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.n_cpu)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu) for x in
                    self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.n_cpu)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu) for x in
                    self.eval_splits]


class AmazonMultiReview(LightningDataModule):
    task_metadata = {
        "num_labels": 5,
        "label_col": "label",
        "tokenize_folder_name": "amazon_multi_review",
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "mrpc",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'label',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__()
        if encode_columns is None:
            encode_columns = []
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.task_metadata['num_labels']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.prepare_data_per_node = True
        self.n_cpu = 2
        self.label_column = label_column
        self.encode_columns = encode_columns

        #  Tokenize the dataset
        self.prepare_data_per_node = True

    def clean_data(self, example):
        cols = self.encode_columns
        example['sentence'] = ["{} {}".format(title, review) for title, review in
                               zip(example['review_title'], example['review_body'])]
        example['stars'] = [int(star) - 1 for star in example['stars']]
        return example

    def prepare_data(self):

        if not os.path.isfile(f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt'):
            print("File not exist")
            print(f'Download and Tokenise')
            dataset = datasets.load_dataset(self.task_name, 'german')
            for split in dataset.keys():
                dataset[split] = dataset[split].map(self.encode_batch, batched=True)
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                dataset[split] = dataset[split].rename_column(self.label_column, "labels")

                # Transform to pytorch tensors and only output the required columns
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                Path(f'{self.task_metadata["tokenize_folder_name"]}').mkdir(parents=True, exist_ok=True)
                torch.save(dataset, f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt')
            except:
                print("File already exist")

        else:
            print("File exist. Load Tokenized data in setup.")



    def setup(self, stage: str):
        # load data here
        try:
            self.dataset = torch.load(f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt')
        except:
            print("File not exist")
            self.prepare_data()
            self.dataset = torch.load(f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt')
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        print('dataset loaded')

    def encode_batch(self, batch):
        """Encodes a batch of input data using the model tokenizer."""
        return self.tokenizer(batch["sentence"], max_length=self.max_seq_length, truncation=True, padding="max_length")

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.n_cpu)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.n_cpu)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu) for x in
                    self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.n_cpu)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu) for x in
                    self.eval_splits]


class TyqiangzData(LightningDataModule):

    task_metadata = {
        "num_labels": 3,
        "label_col": "label",
        "tokenize_folder_name": "tyqiangzdata",
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "tyqiangz/multilingual-sentiments",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'label',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__()
        if encode_columns is None:
            encode_columns = ['text']
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.prepare_data_per_node = True
        self.n_cpu = 2
        self.label_column = label_column
        self.encode_columns = encode_columns

        #  Tokenize the dataset
        self.prepare_data()

    # no need to clean data for this task so no self.clean() method
    def prepare_data(self):
        # called only on 1 GPU per node. Do not do any self.x assignments here
        # download, split tokenise data and save to disk

        if not os.path.isfile(f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt'):
            print("File not exist")
            print(f'Download and Tokenise')
            dataset = datasets.load_dataset(self.task_name, 'german')
            for split in dataset.keys():
                dataset[split] = dataset[split].map(self.encode_batch, batched=True)
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                dataset[split] = dataset[split].rename_column(self.label_column, "labels")

                # Transform to pytorch tensors and only output the required columns
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                Path(f'{self.task_metadata["tokenize_folder_name"]}').mkdir(parents=True, exist_ok=True)
                torch.save(dataset, f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt')
            except:
                print("File already exist")

        else:
            print("File exist. Load Tokenized data in setup.")


    def setup(self, stage: str):
        # load data here
        try:
            self.dataset = torch.load(f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt')
        except:
            print("File not exist")
            self.prepare_data()
            self.dataset = torch.load(f'{self.task_metadata["tokenize_folder_name"]}/tokenized_data.pt')

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        print('dataset loaded')

    def encode_batch(self, batch):
        """Encodes a batch of input data using the model tokenizer."""
        return self.tokenizer(batch["text"], max_length=self.max_seq_length, truncation=True, padding="max_length")

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.n_cpu)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.n_cpu)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu) for x in
                    self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.n_cpu)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu) for x in
                    self.eval_splits]


def getDataModule(task_name="", model_name_or_path: str = "distilbert-base-uncased",
                  max_seq_length: int = 128, train_batch_size: int = 32,
                  eval_batch_size: int = 32):

    if task_name == "sst2":
        return GLUEDataModule(model_name_or_path=model_name_or_path, task_name="sst2",
                          max_seq_length=max_seq_length,
                          train_batch_size=train_batch_size,
                          eval_batch_size=eval_batch_size)

    elif task_name == "amazon_reviews_multi":
        return AmazonMultiReview(model_name_or_path=model_name_or_path,task_name="amazon_reviews_multi",
                             max_seq_length=max_seq_length, train_batch_size=train_batch_size,
                             eval_batch_size=eval_batch_size,
                             label_column='stars',
                             encode_columns=['review_body', 'review_title'])

    elif task_name == "tyqiangz":
        return TyqiangzData(model_name_or_path=model_name_or_path,
                             task_name="tyqiangz/multilingual-sentiments",
                             max_seq_length=max_seq_length,
                             train_batch_size=train_batch_size,
                             eval_batch_size=eval_batch_size,
                             label_column='label',
                             encode_columns=['text'])


if __name__ == "__main__":
    dm = getDataModule(task_name="tyqiangz",
                       model_name_or_path="distilbert-base-uncased",
                       max_seq_length=256, train_batch_size=32,eval_batch_size=32)
    # dm.prepare_data()
    dm.setup("fit")
    print(next(iter(dm.train_dataloader())))
