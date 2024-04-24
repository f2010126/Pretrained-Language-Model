"""
Using the performance matrix and incumbent configs, we can create a training dataset.
Each cell in the performance matrix is a training point.
"""
import argparse
import pandas as pd
from regex import P
from torch import nn
from config2Vector import create_vector
import yaml
import numpy as np
# seed for reproducibility
SEED = 42
np.random.seed(42)
# one hot encoding of all 7 models
ENCODE_MODELS = {"bert-base-uncased" : [0,0,0,0,0,0,1],
                  "bert-base-multilingual-cased":[0,0,0,0,0,1,0],
                  "deepset/bert-base-german-cased-oldvocab": [0,0,0,0,1,0,0],
                  "uklfr/gottbert-base":[0,0,0,1,0,0,0],
                  "dvm1983/TinyBERT_General_4L_312D_de":[0,0,1,0,0,0,0],
                  "linhd-postdata/alberti-bert-base-multilingual-cased":[0,1,0,0,0,0,0],
                  "dbmdz/distilbert-base-german-europeana-cased":[1,0,0,0,0,0,0],
                  }

ENCODE_OPTIM = {'Adam':[0,0,0,1], 
                'AdamW':[0,0,1,0], 
                'SGD':[0,1,0,0], 
                'RAdam':[1,0,0,0], 
                }

ENCODE_SCHED = {'linear_with_warmup': [0, 0, 0, 0, 1], 
                'cosine_with_warmup': [0, 0, 0, 1, 0], 
                'cosine_with_hard_restarts_with_warmup': [0, 0, 1, 0, 0],  
                'polynomial_decay_with_warmup': [0, 1, 0, 0, 0], 
                'constant_with_warmup': [1, 0, 0, 0, 0], 

                }

def rerank_matrix(perf_matrix):
    stacked_df = perf_matrix.stack().reset_index()
    stacked_df.columns = ['Dataset', 'IncumbentOf', 'Performance']
    stacked_df['Rank'] = stacked_df.groupby('Dataset')['Performance'].rank(ascending=False)
    
    return stacked_df

def sort_max(perf_matrix):
    for dataset_name, row in perf_matrix.iterrows():
       max_value = row.max()  # Get the maximum value in the row
       max_index = row.idxmax()  # Get the index of the maximum value
       if max_value != perf_matrix.at[dataset_name, dataset_name+'_incumbent']:
        perf_matrix.at[dataset_name, dataset_name+'_incumbent'] = max_value
          
    return perf_matrix

def read_yaml_file(model, path):
    """
    Read the yaml file and return the details as a dictionary
    eg : {'epochs': 10, 'learning_rate': 0.001, 'optimizer': 'Adam'}
    """
    with open(f"{path}/{model}.yaml", "r") as file:
        details = yaml.safe_load(file)
    # Apply one-hot encoding for model, optimizer, and scheduler
    encoded_details = {}
    encoded_details['model'] = ENCODE_MODELS.get(details['model_name_or_path'], [0] * 7)
    encoded_details['optimizer'] = ENCODE_OPTIM.get(details['optimizer_name'], [0] * 4)
    encoded_details['scheduler'] = ENCODE_SCHED.get(details['scheduler_name'], [0] * 5)
    details.update(encoded_details)
    # remove non encoded details 
    details.pop('model_name_or_path', None)
    details.pop('optimizer_name', None)
    details.pop('scheduler_name', None)
   # rename batch
    per_device_train_batch_size = details.pop('per_device_train_batch_size', None)
    details['batch_size'] = per_device_train_batch_size
    return details

def add_incumbent_config_details(df,incumbent_config_loc):
   df['Model_Details'] = df['IncumbentOf'].apply(read_yaml_file,args=(incumbent_config_loc,))
   model_details_df = df['Model_Details'].apply(pd.Series)
   model_details_df.fillna(0.0, inplace=True)
   df = pd.concat([df, model_details_df], axis=1)
   df.drop(columns=['Model_Details'], inplace=True)
   return df

def generate_training_vectors(perf_matrix, simple_feat,incumbent_config_loc):
    # index of the performance matrix is the dataset name, columns are the incumbent config names
    # for each dataset and incumbent config, we have a training point in the cell. 
    # open the incumbent config file, convert the config to a vector, and also append the performance to the vector
    training_data = []
    sorted= sort_max(perf_matrix)
    stacked_ranked = rerank_matrix(sorted)

    # add the simple features to the training data
    merged_df = pd.merge(stacked_ranked, simple_feat, left_on='Dataset', right_index=True)
    # add the details of the incumbent config to the training data
    training_data=add_incumbent_config_details(merged_df,incumbent_config_loc)
    # write the training data to a file
    training_data = pd.DataFrame(training_data)
    # save the training data
    training_data.to_csv("/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv", index=False)

def view_data():
    # load the training data
    training_data = pd.read_csv("/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/data_m.csv")
    print(training_data.head())
    print(training_data.shape)
    print(training_data.columns)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Training Data')
    parser.add_argument('--config_loc', type=str, help='The location of the incumbent config files', 
                        default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs")
    parser.add_argument('--perf_matrix_loc', type=str, help='Location of the performance matrix', 
                        default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/heatmap_OG_performance_matrix.csv")
    parser.add_argument('--simple_feat', type=str, help='Location of the text file with simple features', 
                        default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/meta_features.csv")

    args = parser.parse_args()

    # load performance matrix
    perf_matrix = pd.read_csv(args.perf_matrix_loc)
    # load features
    simple_feat = pd.read_csv(args.simple_feat)
    generate_training_vectors(perf_matrix, simple_feat,args.config_loc)
    # view_data()


