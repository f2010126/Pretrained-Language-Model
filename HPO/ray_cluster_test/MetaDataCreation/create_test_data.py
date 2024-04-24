from ast import parse
from cgi import test
import sys
import argparse
import os
import json
import pandas as pd
from pyparsing import col
sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))
from BoHBCode.data_modules import get_datamodule
from test_data_modules import DataModule
from create_training_data import add_incumbent_config_details

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
    parser.add_argument('--test_dataset', type=str, default='germeval2018', help='Name of the test dataset')
    parser.add_argument('--config_loc', type=str, help='The location of the incumbent config files', 
                        default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs")
    parser.add_argument('--test_metadata_loc', type=str, help='The location of the test metadata files',
                        default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/cleaned_datasets")
    args = parser.parse_args()
    # read metadata files
    with open(os.path.join(args.test_metadata_loc, args.test_dataset, 'metadata.json')) as f:
        metadata = json.load(f)
        generate_test_data(metadata, args.config_loc)


