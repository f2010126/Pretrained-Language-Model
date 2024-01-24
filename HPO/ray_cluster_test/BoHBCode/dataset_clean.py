from data_modules import DataModule, set_file_name
from typing import List, Optional, Dict
import os
import datasets
import datasets
import torch
from pathlib import Path
from filelock import FileLock
from transformers import AutoTokenizer

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
            model_name_or_path: str ="bert-base-uncased",
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
        self.dir_path = os.path.join(data_dir,self.task_metadata['tokenize_folder_name'])
        self.tokenised_file=set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    def clean_data(self, example):
        # remove extra columns, combine columns, rename columns, etc.
        # return here th example only has text, label. Text is string, labels is a number
        raise NotImplementedError

    # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
            data_folder=self.task_name.split("/")[-1]
            dataset=datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column("Label", "labels")
                dataset[split] = dataset[split].rename_column('Utterance', "text")
                dataset[split] = dataset[split].remove_columns(['Speaker','Idx', 'Dialogue_ID','Dialogue_Act'])
            
            # Save this dataset to disk
            cleaned_data_path=os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, data_folder))
    
    # For processed data that just needs to be tokenised
    def prepare_data(self):
         if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Tokenised File not exist")
            print(f'Tokenise cleaned data')
            clean_data_path=os.path.join(os.getcwd(),  "cleaned_datasets")
            data_folder=self.task_name.split("/")[-1]
            try:
                dataset=datasets.load_from_disk(os.path.join(clean_data_path, data_folder))
                for split in dataset.keys():
                    dataset[split] = dataset[split].map(self.encode_batch, batched=True)
                    dataset[split] = dataset[split].remove_columns(['text'])

                    # Transform to pytorch tensors and only output the required columns
                    columns = [c for c in dataset[split].column_names if c in self.loader_columns]
                    dataset[split].set_format(type="torch", columns=columns)
                
            except Exception as e:
                print("Error loading dataset: ", self.task_name)
                print(e)
                return
            
            try:
                with FileLock(f"Tokenised.lock"):
                    Path(f'{self.dir_path}').mkdir(parents=True, exist_ok=True)
                    torch.save(dataset, f'{self.dir_path}/{self.tokenised_file}')
            except:
                print("File already exist")

# same thing fot swiss judgement
class SwissJudgement(Miam):
    task_metadata = {
        "num_labels": 3,
        "label_col": "label",
        "tokenize_folder_name": "swiss_judgement",
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
        self.dir_path = os.path.join(data_dir,self.task_metadata['tokenize_folder_name'])
        self.tokenised_file=set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    def clean_data(self, example):
        # remove extra columns, combine columns, rename columns, etc.
        # return here th example only has text, label. Text is string, labels is a number
        raise NotImplementedError

    # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
            data_folder=self.task_name.split("/")[-1]
            dataset=datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column('label', "labels")
                dataset[split] = dataset[split].remove_columns(["id", "year","language","region","canton","legal area"])

            # Save this dataset to disk
            cleaned_data_path=os.path.join(os.getcwd(), "cleaned_datasets")
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
            model_name_or_path: str ="bert-base-uncased",
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
        self.dir_path = os.path.join(data_dir,self.task_metadata['tokenize_folder_name'])
        self.tokenised_file=set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True

        #  Tokenize the dataset
        # self.prepare_data()

    def clean_data(self, example):
        # remove extra columns, combine columns, rename columns, etc.
        # return here th example only has text, label. Text is string, labels is a number
        example['text'] = ["{} {}".format(title, review) for title, review in
                               zip(example['question'], example['comment'])]
        return example

    # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
            data_folder=self.task_name.split("/")[-1]
            dataset=datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split]=dataset[split].class_encode_column('topic') # since we want to classify based on topic, encode it
                dataset[split] = dataset[split].rename_column('topic', "labels")
                dataset[split] = dataset[split].map(self.clean_data, batched=True)
                dataset[split]=dataset[split].remove_columns(['question', 'id', 'question_id', 'language', 'comment', 'label', 'numerical_label', 'author'])
            
            # Save this dataset to disk
            cleaned_data_path=os.path.join(os.getcwd(), "cleaned_datasets")
            print(f'clean path-> {cleaned_data_path}')
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))

# same thing fot financial phrasebank
class FinancialPhrasebank(Miam):
    task_metadata = {
        "num_labels": 3,
        "label_col": "label",
        "tokenize_folder_name": "financial_phrasebank",
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
        self.dir_path = os.path.join(data_dir,self.task_metadata['tokenize_folder_name'])
        self.tokenised_file=set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True
    
        # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
            data_folder=self.task_name.split("/")[-1]
            dataset=datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column('label', "labels")
                dataset[split] = dataset[split].rename_column('sentence', "text")
            
            # Save this dataset to disk
            cleaned_data_path=os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))
        
# same as swiss judgement
class TargetHateCheck(Miam):
    task_metadata = {
        "num_labels": 6,
        "label_col": "label",
        "tokenize_folder_name": "target_hate_check",
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
        self.dir_path = os.path.join(data_dir,self.task_metadata['tokenize_folder_name'])
        self.tokenised_file=set_file_name(self.model_name_or_path, self.max_seq_length)
        self.prepare_data_per_node = True
    
        # onetime processing of the dataset
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
            data_folder=self.task_name.split("/")[-1]
            dataset=datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # remove the columns that are not needed
            for split in dataset.keys():
                dataset[split] = dataset[split].rename_column('label', "labels")
                dataset[split] = dataset[split].rename_column('sentence', "text")
            
            # Save this dataset to disk
            cleaned_data_path=os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))




if __name__ == "__main__":
    miam_obj=TargetHateCheck()
    miam_obj.prepare_raw_data()

    print(miam_obj)