import sys
import os
import datasets
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from typing import List, Optional, Dict
from filelock import FileLock
import json

sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))
from BoHBCode.data_modules import DataModule, set_file_name
from BoHBCode.data_augment_labels import calculate_metadata


def add_metadata(dataset,folder_path, foldername):
    metadata = calculate_metadata(dataset)
    metadata["tokenize_folder_name"] = foldername
    metadata["task_name"] = foldername

    # Write metadata to a JSON file
    with open(os.path.join(folder_path,foldername,'metadata.json'), 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

class TestDataset(DataModule):
    task_metadata = {
        "num_labels": 2,
        "label_col": "label",}
    
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
            task_name: str = 'AugmentedDataset', ## loaded from cleaned_datasets metadata
            tokenize_folder_name: str = 'Augmented',    
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            data_dir='./data/tokenized_datasets/Augmented', # has to be the exact path to the tokenized data file. 

            **kwargs,
    ):
        self.prepare_data_per_node = True
        self.n_cpu = 0
        self.task_metadata['task_name'] = task_name
        self.task_metadata['tokenize_folder_name'] = tokenize_folder_name

        self.task_name = self.task_metadata['task_name']
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.tokenised_file = set_file_name(self.model_name_or_path, self.max_seq_length)
        self.dir_path = data_dir
        self.prepare_data_per_node = True
    
    def prepare_raw_data(self):
        raise NotImplementedError("This method should be implemented in the child class")
    
    def prepare_data(self):
        # if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
        #     print("Prepare to Clean Data and Tokenize")
        #     cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets", "Augmented", self.task_metadata['tokenize_folder_name'])
        return super().prepare_data(clean_data_path="cleaned_datasets")

# test class for germeval2018
class TestGermeval2018(TestDataset):
    task_metadata = {
        "num_labels": 2,
        "label_col": "label",
        "task_name": "germeval2018",
        "tokenize_folder_name": "germeval2018"
    }
    def __init__(
            self,
            config=Optional[Dict],
            model_name_or_path: str = "dbmdz/distilbert-base-german-europeana-cased",
            task_name: str = 'germeval2018',
            tokenize_folder_name: str = 'germeval2018',
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            data_dir='./data/tokenized_datasets',
            **kwargs,
    ):
        super().__init__(config=config, model_name_or_path=model_name_or_path, task_name=task_name,
                         tokenize_folder_name=tokenize_folder_name, max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir,
                         **kwargs)
    
    def prepare_raw_data(self):
        if not os.path.isfile(f'{self.dir_path}/{self.tokenised_file}'):
            print("Prepare Data for the first time")
            print(f'Download clean')
            raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
            data_folder = self.task_name.split("/")[-1]
            dataset = datasets.load_from_disk(os.path.join(raw_data_path, data_folder))
            # train test val split 70 10 30
            train_testvalid = dataset['train'].train_test_split(test_size=0.2)
            test_valid = train_testvalid['test'].train_test_split(test_size=0.1)
            dataset = DatasetDict({
                'train': train_testvalid['train'],
                'test': dataset['test'],
                'validation': test_valid['train']})
            
            dataset = dataset.shuffle()
            dataset = dataset.class_encode_column("fine-grained")
            dataset = dataset.rename_column('fine-grained', "labels")
            dataset = dataset.rename_column('text', "sentence")
            dataset = dataset.remove_columns(["coarse-grained"])

            # Save this dataset to disk
            ###### TODO: get metadata and store that as welk
            cleaned_data_path = os.path.join(os.getcwd(), "cleaned_datasets")
            if not os.path.exists(cleaned_data_path):
                os.makedirs(cleaned_data_path)
            dataset.save_to_disk(os.path.join(cleaned_data_path, self.task_metadata['tokenize_folder_name']))
            # add metadata.json
            add_metadata(dataset,cleaned_data_path, self.task_metadata['tokenize_folder_name'])
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage=None):
        return super().setup(stage=stage)

class TestGermeval2018Coarse(TestDataset):
    task_metadata = {
        "num_labels": 2,
        "label_col": "label",
        "task_name": "germeval2018",
        "tokenize_folder_name": "germeval2018"
    }

def get_test_data(dataset_name, model_name_or_path, max_seq_length, train_batch_size, eval_batch_size, data_dir):
    if dataset_name == "germeval2018":
        return TestGermeval2018(task_name="germeval2018", model_name_or_path=model_name_or_path,
                            max_seq_length=max_seq_length,
                            train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, data_dir=data_dir)
    else:
        print("Dataset not found")
if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "tokenized_data", "germeval2018")
    dm = get_test_data(dataset_name="germeval2018", model_name_or_path="dbmdz/distilbert-base-german-europeana-cased",
                            max_seq_length=128,
                            train_batch_size=32, eval_batch_size=32, data_dir=data_dir)
    dm.prepare_raw_data()
    dm.prepare_data()
    dm.setup("fit")