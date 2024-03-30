"""
More data modules for the different datasets
"""
import os
from pathlib import Path
from typing import Optional, Dict

import datasets
import torch
from filelock import FileLock
from transformers import AutoTokenizer

from data_modules import DataModule, set_file_name, get_datamodule


# repeat
class Miam(DataModule):
    task_metadata = {
        "num_labels": 31,
        "label_col": "label",
        "tokenize_folder_name": "miam",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = "miam",
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column("Label", "labels")
                dataset[split] = dataset[split].rename_column('Utterance', "sentence")
                dataset[split] = dataset[split].remove_columns(['Speaker', 'Idx', 'Dialogue_ID', 'Dialogue_Act'])

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, data_folder))

    # For processed data that just needs to be tokenised
    def prepare_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Tokenised File not exist")
            print(f'Tokenise the cleaned data')
            clean_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            data_folder = self.task_name.split("/")[-1]
            try:
                dataset = datasets.load_from_disk(os.path.join(clean_data_path, data_folder))
                for split in dataset.keys():
                    dataset[split] = dataset[split].map(self.encode_batch, batched=True)
                    dataset[split] = dataset[split].remove_columns(['sentence'])

                    # Transform to pytorch tensors and only output the required columns
                    columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                    dataset[split].set_format(type="torch", columns=columns)

            except Exception as e:
                print("Error loading dataset: ", self.task_name)
                print(e)
                return

            try:
                with FileLock(f"Tokenised.lock"):
                    cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")


# same thing fot swiss judgement
class SwissJudgement(Miam):
    task_metadata = {
        "num_labels": 3,
        "label_col": "label",
        "tokenize_folder_name": "swiss_judgment_prediction",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'rcds/swiss_judgment_prediction',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column('label', "labels")
                dataset[split] = dataset[split].rename_column('text', "sentence")
                dataset[split] = dataset[split].remove_columns(
                    ["id", "year", "language", "region", "canton", "legal area"])

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


class XStance(Miam):
    task_metadata = {
        "num_labels": 10,
        "label_col": "label",
        "tokenize_folder_name": "x_stance",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'x_stance',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    def clean_data(self, example):
        # remove extra columns, combine columns, rename columns, etc.
        # return here th example only has text, label. Text is string, labels is a number
        example['sentence'] = ["{} {}".format(title, review) for title, review in
                               zip(example['question'], example['comment'])]
        return example

    # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].class_encode_column(
                    'topic')  # since we want to classify based on topic, encode it
                dataset[split] = dataset[split].rename_column('topic', "labels")
                dataset[split] = dataset[split].map(self.clean_data, batched=True)
                dataset[split] = dataset[split].remove_columns(
                    ['question', 'id', 'question_id', 'language', 'comment', 'label', 'numerical_label', 'author'])

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            print(f'clean path-> {cleaned_data_path}')
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


class FinancialPhrasebank(Miam):
    task_metadata = {
        "num_labels": 3,
        "label_col": "label",
        "tokenize_folder_name": "financial_phrasebank_75agree_german",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'scherrmann/financial_phrasebank_75agree_german',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        # onetime processing of the dataset

    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                # shuffle the dataset
                dataset[split] = dataset[split].shuffle()
                dataset[split] = dataset[split].rename_column('label', "labels")

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


class TargetHateCheck(Miam):
    task_metadata = {
        "num_labels": 6,
        "label_col": "label",
        "tokenize_folder_name": "hatecheck-german",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'Paul/hatecheck-german',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        # onetime processing of the dataset

    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].filter(lambda example: example['target_ident'] != 'null')
                dataset[split] = dataset[split].class_encode_column(
                    'target_ident')  # since we want to classify based on topic, encode it
                dataset[split] = dataset[split].rename_column('target_ident', "labels")
                dataset[split] = dataset[split].rename_column('test_case', "sentence")
                # remove other columns
                dataset[split] = dataset[split].remove_columns(
                    ['mhc_case_id', 'functionality', 'label_gold', 'ref_case_id', 'ref_templ_id',
                     'templ_id', 'case_templ', 'gender_male', 'gender_female', 'label_annotated',
                     'label_annotated_maj', 'disagreement_in_case', 'disagreement_in_template'])

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


class Mlsum(Miam):
    task_metadata = {
        "num_labels": 21,
        "label_col": "label",
        "tokenize_folder_name": "mlsum",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'mlsum',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

    def clean_data(self, example):
        # remove extra columns, combine columns, rename columns, etc.
        # return here th example only has text, label. Text is string, labels is a number
        example['sentence'] = ["{} {}".format(title, review) for title, review in
                               zip(example['title'], example['text'])]
        return example

        # onetime processing of the dataset

    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].class_encode_column(
                    'topic')  # since we want to classify based on topic, encode it
                dataset[split] = dataset[split].rename_column('topic', "labels")
                dataset[split] = dataset[split].map(self.clean_data, batched=True)
                # remove other columns
                dataset[split] = dataset[split].remove_columns(['title', 'summary', 'url', 'date'])

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


class ArgumentMining(Miam):
    task_metadata = {
        "num_labels": 4,
        "label_col": "label",
        "tokenize_folder_name": "german_argument_mining",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'joelniklaus/german_argument_mining',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].class_encode_column(
                    'label')  # since we want to classify based on topic, encode it
                dataset[split] = dataset[split].rename_column('label', "labels")
                dataset[split] = dataset[split].rename_column('input_sentence', "sentence")
                # remove other columns
                dataset[split] = dataset[split].remove_columns(['file_number', 'context_before', 'context_after'])

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


# who said what in the German Bundestag
class Bundestag(Miam):
    task_metadata = {
        "num_labels": 7,
        "label_col": "label",
        "tokenize_folder_name": "Bundestag-v2",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'threite/Bundestag-v2',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                # remove the rows that have no party
                dataset[split] = dataset[split].filter(lambda example: example['party'] != '')
                dataset[split] = dataset[split].class_encode_column(
                    'party')  # since we want to classify based on topic, encode it
                dataset[split] = dataset[split].rename_column('party', "labels")
                dataset[split] = dataset[split].rename_column('text', "sentence")

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


class Tagesschau(Miam):
    task_metadata = {
        "num_labels": 7,
        "label_col": "label",
        "tokenize_folder_name": "tagesschau",
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
            model_name_or_path: str = "bert-base-uncased",
            task_name: str = 'tillschwoerer/tagesschau',
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
        self.dir_path = os.path.join(data_dir, self.task_metadata['tokenize_folder_name'])
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column('label', "labels")
                dataset[split] = dataset[split].rename_column('text', "sentence")
                # remove other columns

            # Save this dataset to disk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))


def get_datamodule_extra(task_name="", model_name_or_path: str = "distilbert-base-uncased",
                   max_seq_length: int = 128, train_batch_size: int = 32,
                   eval_batch_size: int = 32, data_dir='./data'):
    # add documenation
    """
    Get the datamodule for the task
    :param task_name: the name of the task
    :param model_name_or_path: the name of the model to use
    :param max_seq_length: the maximum sequence length
    :param train_batch_size: the training batch size
    :param eval_batch_size: the evaluation batch size
    :param data_dir: the directory to save the data
    """
    if task_name == "miam":
        return Miam(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                    train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "swiss_judgment_prediction":
        return SwissJudgement(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                              train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "x_stance":
        return XStance(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                       train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "financial_phrasebank_75agree_german":
        return FinancialPhrasebank(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                                   train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
                                   data_dir=data_dir)
    elif task_name == "hatecheck-german":
        return TargetHateCheck(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                               train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "mlsum":
        return Mlsum(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                     train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "german_argument_mining":
        return ArgumentMining(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                              train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "Bundestag-v2":
        return Bundestag(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    elif task_name == "tagesschau":
        return Tagesschau(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length,
                          train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    else:
        raise ValueError(f"Task {task_name} not supported")

    # Clean the datasets, only 'sentence' and 'label' columns are kept. Run this only once


def clean_datasets():
    # add documenation
    """
    Clean the datasets by removing unwanted columns, only 'sentence' and 'label' columns are kept
    """
    dataset_list = ['Bundestag-v2', 'tagesschau', 'german_argument_mining',
                    'mlsum', 'hatecheck-german', 'financial_phrasebank_75agree_german',
                    'x_stance', 'swiss_judgment_prediction', 'miam']

    data_dir = os.path.join(os.getcwd(), "tokenized_data")
    os.makedirs(data_dir, exist_ok=True)
    for dataset_name in dataset_list:
        print("Cleaning dataset: ", dataset_name)
        dm = get_datamodule(task_name=dataset_name, data_dir=data_dir)
        dm.prepare_raw_data()
        print(f"Done cleaning dataset: {dataset_name}")

    # Tokenise the datasets

models_list = ["bert-base-uncased", "bert-base-multilingual-cased",
                   "deepset/bert-base-german-cased-oldvocab", "uklfr/gottbert-base",
                   "dvm1983/TinyBERT_General_4L_312D_de", "linhd-postdata/alberti-bert-base-multilingual-cased",
                   "dbmdz/distilbert-base-german-europeana-cased"]
max_seq_length = [128, 256, 512]

def tokenise_datasets():
    # add documenation
    """
    Tokenise the datasets using the models
    """
    dataset_list = ['Bundestag-v2', 'tagesschau', 'german_argument_mining',
                    'mlsum', 'hatecheck-german', 'financial_phrasebank_75agree_german',
                    'x_stance', 'swiss_judgment_prediction', 'miam']
    
    # Tokenise the datasets
    for dataset_name in dataset_list:
        print("Tokenising dataset: ", dataset_name)
        data_dir = os.path.join(os.getcwd(), "tokenized_data")
        for model in models_list:
            for seq_length in max_seq_length:
                # check if the file already exists
                tokenised_file = set_file_name(model, seq_length)
                if os.path.isfile(f'{data_dir}/{dataset_name}/{tokenised_file}'):
                    print(
                        f"Tokenised dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length} already exists")
                    continue

                print(f"Tokenising dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length}")
                dm = get_datamodule(task_name=dataset_name, model_name_or_path=model, max_seq_length=seq_length,
                                    data_dir=data_dir)
                dm.prepare_raw_data()
                dm.prepare_data()
                print(f"Done tokenising dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length}")



def tokenise_model_seq(model_name, seq_length):
    pass

def tokenise_augmented_datasets(tokneised_dir):
    data_dir = os.path.join(os.getcwd(), "cleaned_datasets", "Augmented")
    # path of cleaned datasets/Augmented
    for dataset_name in os.listdir(data_dir):
        if not dataset_name.startswith('.'):
            print("Tokenising dataset: ", dataset_name)
            token_dataset_path = os.path.join(tokneised_dir, dataset_name)
            print(f"Future path fpr dataset --->{token_dataset_path}")
            dm = get_datamodule(task_name="augmented", model_name_or_path="dbmdz/distilbert-base-german-europeana-cased",
                            max_seq_length=128,
                            train_batch_size=32, eval_batch_size=32, data_dir=token_dataset_path)
            dm.prepare_data()

            for model in models_list:
                for seq_length in max_seq_length:
                # check if the file already exists
                    tokenised_file = set_file_name(model, seq_length)
                    if os.path.isfile(f'{token_dataset_path}/{tokenised_file}'):
                        print(f"Tokenised dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length} already exists")
                        continue
                    print(f"Tokenising dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length}")
                    dm = get_datamodule(task_name="augmented", model_name_or_path=model, max_seq_length=seq_length,
                                    data_dir=token_dataset_path)
                    dm.prepare_data()
                    print(f"Done tokenising dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length}")



if __name__ == "__main__":
    # prepare raw data for all classes
    data_dir = os.path.join(os.getcwd(), "raw_datasets")
    os.makedirs(data_dir, exist_ok=True)
    data_dir = os.path.join(os.getcwd(), "cleaned_datasets")
    os.makedirs(data_dir, exist_ok=True)
    data_dir = os.path.join(os.getcwd(), "tokenized_data")
    os.makedirs(data_dir, exist_ok=True)

    # tokenise_datasets()
    data_dir = os.path.join(os.getcwd(), "tokenized_data", "Augmented")
    tokenise_augmented_datasets(data_dir)
