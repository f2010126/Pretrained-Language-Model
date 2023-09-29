from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import datasets
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
from pathlib import Path
from datasets import DatasetDict
from typing import List, Optional, Dict
from filelock import FileLock
import re
import logging

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
        self.n_cpu = 10

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


# Utils
def set_file_name(model_name_or_path, max_seq_length):
    model_name_or_path= model_name_or_path.replace(r'/','_')
    return f'{model_name_or_path}_{max_seq_length}_tokenized_data.pt'


# Base Class
class DataModule(LightningDataModule):
    task_metadata = {
        "num_labels": -1,
        "label_col": "label",
        "tokenize_folder_name": "store_folder",
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
            config=Optional[Dict],
            model_name_or_path: str ="bert-base-uncased",
            task_name: str = "mrpc",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'labels',
            data_dir='./data',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__()
        self.prepare_data_per_node = True
        self.n_cpu = 0

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.label_column = label_column

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.dir_path = os.path.join(data_dir,self.task_metadata['tokenize_folder_name'])
        self.tokenised_file=set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    def clean_data(self, example):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage: str):
        logging.debug(f'Setup data in directory: {self.dir_path}')
        # load data here
        try:
            self.dataset = torch.load(f'{self.dir_path}/{self.tokenised_file}')
        except:
            logging.debug("The tokenised data file not exist")
            self.prepare_data()
            self.dataset = torch.load(f'{self.dir_path}/{self.tokenised_file}')

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        logging.debug('dataset loaded')

    def encode_batch(self, batch):
        """Encodes a batch of input data using the model tokenizer."""
        return self.tokenizer(batch["sentence"], max_length=self.max_seq_length, truncation=True, padding="max_length")

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.n_cpu, pin_memory=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.n_cpu, pin_memory=True)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu, pin_memory=True) for x in
                    self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.n_cpu, pin_memory=True)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.n_cpu, pin_memory=True) for x in
                    self.eval_splits]
class AmazonMultiReview(DataModule):
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
            task_name: str = "amazon_reviews_multi",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'labels',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path,max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size,eval_batch_size=eval_batch_size,
                         task_name=task_name,data_dir=data_dir )
        if encode_columns is None:
            encode_columns = []
        # self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.task_metadata['num_labels']
        self.encode_columns = encode_columns
        self.label_column = label_column

    def clean_data(self, example):
        example['sentence'] = ["{} {}".format(title, review) for title, review in
                               zip(example['review_title'], example['review_body'])]
        example['stars'] = [int(star) - 1 for star in example['stars']]

        return example

    def prepare_data(self):

        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data File not exist")
            print(f'Download and Tokenise')
            dataset = datasets.load_dataset(self.task_name, 'de').shuffle(seed=42)
            dataset = dataset.map(self.clean_data, batched=True)
            dataset = dataset.map(self.encode_batch, batched=True)
            dataset = dataset.rename_column("stars", "labels")
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)

                # Transform to pytorch tensors and only output the required columns
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')

            except:
                print("File already exist")

        else:
            print("File exist. Load Tokenized data in setup.")


class TyqiangzData(DataModule):
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
            label_column: str = 'labels',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path,max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size,eval_batch_size=eval_batch_size,
                         task_name=task_name,data_dir=data_dir)
        if encode_columns is None:
            encode_columns = ['text']

        self.label_column = label_column
        self.encode_columns = encode_columns


    # no need to clean data for this task so no self.clean() method
    def prepare_data(self):
        # called only on 1 GPU per node. Do not do any self.x assignments here
        # download, split tokenise data and save to disk

        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data File not exist")
            print(f'Download and Tokenise')
            dataset = datasets.load_dataset(self.task_name, 'german')
            dataset = dataset.rename_column("label", "labels")
            dataset = dataset.rename_column("text", "sentence")
            dataset = dataset.map(self.encode_batch, batched=True)
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)

                # Transform to pytorch tensors and only output the required columns
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

        else:
            print("File exist. Load Tokenized data in setup.")


class OmpData(DataModule):
    task_metadata = {
        "num_labels": 9,
        "label_col": "labels",
        "tokenize_folder_name": "omp",
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
            max_seq_length: int,
            train_batch_size: int,
            eval_batch_size: int,
            task_name: str = "omp",
            label_column: str = 'labels',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path, )
        if encode_columns is None:
            encode_columns = ['text']
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.prepare_data_per_node = True
        self.label_column = label_column
        self.encode_columns = encode_columns

        #  Tokenize the dataset
        self.prepare_data()

    def clean_data(self, example):
        # combine the title and review for text field
        example['sentence'] = ["{} {}".format(title, review) for title, review in
                               zip(example['Headline'], example['Body'])]
        example['labels'] = example['Category']
        return example

    def prepare_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("File not exist Prepare data")
            print(f'Download and Tokenise')
            # load a shuffled version of the dataset
            dataset = datasets.load_dataset(self.task_name, 'posts_labeled', split='train').shuffle(seed=42)
            # 90% train, 10% test + validation
            train_testvalid = dataset.train_test_split(test_size=0.1)
            # Split the 10% test + valid in half test, half valid
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
            # gather everyone if you want to have a single DatasetDict
            dataset = DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'validation': test_valid['train']})
            dataset = dataset.map(self.clean_data, batched=True)
            dataset = dataset.map(self.encode_batch, batched=True)
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

        else:
            print("File exist. Load Tokenized data in setup.")


class SentiLexData(DataModule):
    task_metadata = {
        "num_labels": 2,
        "label_col": "labels",
        "tokenize_folder_name": "sentilex",
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
            task_name: str = "senti_lex",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'labels',
            encode_columns=None,
            data_dir='./data',
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path,max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size,eval_batch_size=eval_batch_size,
                         task_name=task_name,data_dir=data_dir)
        if encode_columns is None:
            encode_columns = ['text']

        self.label_column = label_column
        self.encode_columns = encode_columns
        self.num_labels = self.task_metadata['num_labels']


    def clean_data(self, example):
        # rename/ combine columns
        example['sentence'] = example['word']
        example['labels'] = example['sentiment']
        return example

    def prepare_data(self):

        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data File not exist")
            print(f'Download and Tokenise')
            # load a shuffled version of the dataset
            dataset = datasets.load_dataset(self.task_name, 'de', split='train').shuffle(seed=42)
            # 90% train, 10% test + validation
            dataset = dataset.map(self.clean_data, batched=True)
            dataset = dataset.map(self.encode_batch, batched=True)
            train_testvalid = dataset.train_test_split(test_size=0.1)
            # Split the 10% test + valid in half test, half valid
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
            # gather everyone if you want to have a single DatasetDict
            dataset = DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'validation': test_valid['train']})
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

        else:
            print("File exist in Prepare Data. Load Tokenized data in setup.")


class CardiffMultiSentiment(DataModule):
    task_metadata = {
        "num_labels": 3,
        "label_col": "labels",
        "tokenize_folder_name": "cardiff_multi_sentiment",
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
            task_name: str = "cardiffnlp/tweet_sentiment_multilingual",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'labels',
            encode_columns=None,
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path,max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size,eval_batch_size=eval_batch_size,
                         task_name=task_name,data_dir=data_dir )
        if encode_columns is None:
            encode_columns = ['text']

        self.label_column = label_column
        self.encode_columns = encode_columns


    def clean_data(self, example):
        # combine the title and review for text field
        example['sentence'] = example['text']
        return example

    def prepare_data(self):

        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data File not exist")
            print(f'Download and Tokenise')
            # load a shuffled version of the dataset
            dataset = datasets.load_dataset(self.task_name, 'german').shuffle(seed=42)
            dataset = dataset.map(self.clean_data, batched=True)
            dataset = dataset.map(self.encode_batch, batched=True)
            dataset = dataset.rename_column("label", "labels")
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

        else:
            print("File exist. Load Tokenized data in setup.")


class MtopDomain(DataModule):
    task_metadata = {
        "num_labels": 11,
        "label_col": "labels",
        "tokenize_folder_name": "mtop_domain",
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
            task_name: str = "mteb/mtop_domain",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'labels',
            encode_columns=None,
            data_dir='./data',
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path,max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size,eval_batch_size=eval_batch_size,
                         task_name=task_name,data_dir=data_dir)
        if encode_columns is None:
            encode_columns = ['text']

        self.label_column = label_column
        self.encode_columns = encode_columns
        self.num_labels = self.task_metadata['num_labels']


    def clean_data(self, example):
        # rename/ combine columns
        example['sentence'] = example['word']
        example['labels'] = example['sentiment']
        return example

    def prepare_data(self):

        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data File not exist")
            print(f'Download and Tokenise')
            # load a shuffled version of the dataset
            dataset = datasets.load_dataset(self.task_name, 'de')
            dataset = dataset.rename_column("label", "labels")
            dataset = dataset.rename_column("text", "sentence")
            dataset = dataset.map(self.encode_batch, batched=True)
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

        else:
            print("File exist in Prepare Data. Load Tokenized data in setup.")


class GermEval2018Coarse(DataModule):
    task_metadata = {
        "num_labels": 2,
        "label_col": "labels",
        "tokenize_folder_name": "germeval2018coarse",
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
            task_name: str = "gwlms/germeval2018",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            label_column: str = 'labels',
            encode_columns=None,
            data_dir='./data',
            **kwargs,
    ):

        super().__init__(model_name_or_path=model_name_or_path,max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size,eval_batch_size=eval_batch_size,
                         task_name=task_name,data_dir=data_dir)
        if encode_columns is None:
            encode_columns = ['text']

        self.label_column = label_column
        self.encode_columns = encode_columns
        self.num_labels = self.task_metadata['num_labels']

    def prepare_data(self):

        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data File not exist")
            print(f'Download and Tokenise')
            # load a shuffled version of the dataset
            dataset = datasets.load_dataset(self.task_name)
            dataset=dataset.class_encode_column("coarse-grained")
            dataset = dataset.rename_column("coarse-grained", "labels")
            dataset = dataset.rename_column("text", "sentence")
            train_testvalid = dataset['train'].train_test_split(test_size=0.3)
            dataset = DatasetDict({
                'train': train_testvalid['train'],
                'test': train_testvalid['test'],
                'validation': dataset['test']})
            dataset = dataset.map(self.encode_batch, batched=True)
            for split in dataset.keys():
                remove_features = set(dataset[split].features) ^ set(
                    [self.label_column] + ["input_ids", "attention_mask"])
                dataset[split] = dataset[split].remove_columns(remove_features)
                columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                dataset[split].set_format(type="torch", columns=columns)

            # save the tokenized data to disk
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

        else:
            print("File exist in Prepare Data. Load Tokenized data in setup.")


def get_datamodule(task_name="", model_name_or_path: str = "distilbert-base-uncased",
                   max_seq_length: int = 128, train_batch_size: int = 32,
                   eval_batch_size: int = 32, data_dir='./data'):
    if task_name == "amazon_reviews_multi":
        return AmazonMultiReview(model_name_or_path=model_name_or_path,
                                 max_seq_length=max_seq_length, train_batch_size=train_batch_size,
                                 eval_batch_size=eval_batch_size,
                                 data_dir=data_dir )

    elif task_name == "tyqiangz":
        return TyqiangzData(model_name_or_path=model_name_or_path,
                            max_seq_length=max_seq_length,
                            train_batch_size=train_batch_size,
                            eval_batch_size=eval_batch_size,
                            data_dir=data_dir)
    elif task_name == "omp":
        return OmpData(model_name_or_path=model_name_or_path,
                       max_seq_length=max_seq_length,
                       train_batch_size=train_batch_size,
                       eval_batch_size=eval_batch_size, data_dir=data_dir)

    elif task_name == "sentilex":
        return SentiLexData(model_name_or_path=model_name_or_path,
                            max_seq_length=max_seq_length,
                            train_batch_size=train_batch_size,
                            eval_batch_size=eval_batch_size,
                            data_dir=data_dir )

    elif task_name == "cardiff_multi_sentiment":
        return CardiffMultiSentiment(model_name_or_path=model_name_or_path,
                                     max_seq_length=max_seq_length,
                                     train_batch_size=train_batch_size,
                                     eval_batch_size=eval_batch_size,
                                     data_dir=data_dir )
    elif task_name == "mtop_domain":
        return MtopDomain(model_name_or_path=model_name_or_path,
                                 max_seq_length=max_seq_length, train_batch_size=train_batch_size,
                                 eval_batch_size=eval_batch_size,
                                 data_dir=data_dir)

    elif task_name == "germeval2018_coarse":
        return GermEval2018Coarse(model_name_or_path=model_name_or_path,
                                 max_seq_length=max_seq_length, train_batch_size=train_batch_size,
                                 eval_batch_size=eval_batch_size,
                                 data_dir=data_dir)

    elif task_name == "germeval2018_fine":
        return MtopDomain(model_name_or_path=model_name_or_path,
                                 max_seq_length=max_seq_length, train_batch_size=train_batch_size,
                                 eval_batch_size=eval_batch_size,
                                 data_dir=data_dir)
    else:
        print("Task not found")
        raise NotImplementedError


if __name__ == "__main__":
    print(f'current working directory: {os.getcwd()}')
    data_dir=os.path.join(os.getcwd(), "testing_data")
    dm = get_datamodule(task_name="germeval2018_coarse", model_name_or_path="distilbert-base-uncased", max_seq_length=156,
                        train_batch_size=32, eval_batch_size=32,data_dir=data_dir)
    dm.prepare_data()
    dm.setup("fit")
    print(next(iter(dm.val_dataloader())))
