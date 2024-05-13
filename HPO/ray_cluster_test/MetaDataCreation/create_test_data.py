from ast import parse
from cgi import test
from curses import meta
import sys
import argparse
import os
import json
import pandas as pd
from pyparsing import col
from datasets import load_dataset

# local
sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))
from BoHBCode.data_modules import get_datamodule
from test_data_modules import DataModule
from BoHBCode.data_augment_labels import calculate_metadata
from create_training_data import add_incumbent_config_details


def parse_dataset2019(coarse=False):
    dataset = load_dataset('hermanli/germeval19')
    dataset = dataset.rename_column('tweet', 'sentence')

    metadata={}
    if coarse:
        dataset = dataset.rename_column('task1', 'labels')
        metadata['task_name'] = 'germeval2019_coarse'
    else:
        dataset = dataset.rename_column('task2', 'labels')
        metadata['task_name'] = 'germeval2019_fine'

    dataset=dataset.class_encode_column("labels")
    metadata.update(calculate_metadata(dataset))
    
    return metadata


def parse_dataset2018(coarse=False):
    # need metadata for the dataset, so no need to preprocess. Just rename the text field to sentence, the label field to labels
    dataset = load_dataset('gwlms/germeval2018')

    if coarse:
        dataset = dataset.rename_column('text', 'sentence')
        dataset = dataset.rename_column('coarse-grained', 'labels')
        dataset=dataset.class_encode_column("labels")
        metadata = calculate_metadata(dataset)
        metadata['task_name'] = 'germeval2018_coarse'
    else:
        dataset = dataset.rename_column('text', 'sentence')
        dataset = dataset.rename_column('fine-grained', 'labels')
        dataset=dataset.class_encode_column("labels")
        metadata = calculate_metadata(dataset)
        metadata['task_name'] = 'germeval2018_fine'
    
    return metadata


def parsedataset2017():

    dataset = load_dataset('akash418/germeval_2017')
    dataset = dataset.rename_column('text', 'sentence')
    dataset = dataset.rename_column('sentiment', 'labels')
    dataset=dataset.class_encode_column("labels")
    metadata = calculate_metadata(dataset)
    metadata['task_name'] = 'germeval2017'
    return metadata



def generate_test_data(test_metadata,incumbent_config_loc):
    with open('/Users/diptisengupta/Desktop/datasets.txt') as f:
       dataset_names = [line.strip() for line in f.readlines()]
       pipelines = list(map(lambda x: x + '_incumbent', dataset_names))
       test_models=pd.DataFrame(pipelines, columns=['IncumbentOf'])
       testing_data=add_incumbent_config_details(test_models,incumbent_config_loc)
       # convert test metadata to a dataframe
       selected_keys = ['num_training_samples', 'num_labels', 'average_text_length','task_name']
       for key in selected_keys:
          testing_data[key] = test_metadata[key]
       testing_data.rename(columns={'task_name':'Dataset'}, inplace=True)
       testing_data.to_csv(f"/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_{test_metadata['task_name']}.csv", index=False)
       
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Test Data')
    parser.add_argument('--test_dataset', type=str, default='germeval2017', help='Name of the test dataset')
    parser.add_argument('--config_loc', type=str, help='The location of the incumbent config files', 
                        default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs")
    args = parser.parse_args()
    # read metadata files

    if args.test_dataset=='germeval2018_fine':
        test_metada=parse_dataset2018(coarse=False)
    elif args.test_dataset=='germeval2018_coarse':
        test_metada=parse_dataset2018(coarse=True)
    elif args.test_dataset=='germeval2017':
        test_metada=parsedataset2017()
    elif args.test_dataset=='germeval2019_fine':
        test_metada=parse_dataset2019(coarse=False)
    elif args.test_dataset=='germeval2019_coarse':
        test_metada=parse_dataset2019(coarse=True)
    
    generate_test_data(test_metada, args.config_loc)

    # with open(os.path.join(args.test_metadata_loc, args.test_dataset, 'metadata.json')) as f:
    #     metadata = json.load(f)
    #     generate_test_data(metadata, args.config_loc)


