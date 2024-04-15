
"""
This script is used to generate the performance matrix from the run results.
for each incumbent configuration, it will run it against the test data of all the datasets and generate the rows.
The dataset can be in the cleaned_datasets folder or the cleaned_datasets/augmented folder.

"""
import seaborn as sns

import os
import argparse
import datasets
import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import yaml

sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))
from BoHBCode.data_modules import get_datamodule
from BoHBCode.evaluate_single_config import train_single_config


pipelines=['Bundestag-v2_incumbent', 'Bundestag-v2_1X_2Labels_incumbent', 'Bundestag-v2_1X_3Labels_incumbent', 'Bundestag-v2_1X_4Labels_incumbent', 
           'Bundestag-v2_1X_5Labels_incumbent', 'Bundestag-v2_1X_6Labels_incumbent', 'financial_phrasebank_75agree_german_incumbent', 
           'financial_phrasebank_75agree_german_1X_2Labels_incumbent', 'german_argument_mining_incumbent', 'german_argument_mining_1X_2Labels_incumbent', 
           'german_argument_mining_1X_3Labels_incumbent', 'gnad10_incumbent', 'gnad10_1X_2Labels_incumbent', 'gnad10_1X_3Labels_incumbent', 
           'gnad10_1X_4Labels_incumbent', 'gnad10_1X_5Labels_incumbent', 'gnad10_1X_6Labels_incumbent', 'gnad10_1X_7Labels_incumbent', 
           'gnad10_1X_8Labels_incumbent', 'hatecheck-german_incumbent', 'hatecheck-german_1X_2Labels_incumbent', 'hatecheck-german_1X_3Labels_incumbent', 
           'hatecheck-german_1X_4Labels_incumbent', 'hatecheck-german_1X_5Labels_incumbent', 'hatecheck-german_1X_6Labels_incumbent', 'miam_incumbent', 
           'miam_1X_10Labels_incumbent', 'miam_1X_11Labels_incumbent', 'miam_1X_12Labels_incumbent', 'miam_1X_13Labels_incumbent', 'miam_1X_14Labels_incumbent', 
           'miam_1X_15Labels_incumbent', 'miam_1X_16Labels_incumbent', 'miam_1X_17Labels_incumbent', 'miam_1X_18Labels_incumbent', 'miam_1X_19Labels_incumbent', 
           'miam_1X_20Labels_incumbent', 'miam_1X_21Labels_incumbent', 'miam_1X_22Labels_incumbent', 'miam_1X_23Labels_incumbent', 'miam_1X_24Labels_incumbent', 
           'miam_1X_25Labels_incumbent', 'miam_1X_26Labels_incumbent', 'miam_1X_27Labels_incumbent', 'miam_1X_28Labels_incumbent', 'miam_1X_29Labels_incumbent', 
           'miam_1X_2Labels_incumbent', 'miam_1X_30Labels_incumbent', 'miam_1X_3Labels_incumbent', 'miam_1X_4Labels_incumbent', 'miam_1X_5Labels_incumbent', 
           'miam_1X_6Labels_incumbent', 'miam_1X_7Labels_incumbent', 'miam_1X_8Labels_incumbent', 'miam_1X_9Labels_incumbent', 'mlsum_incumbent', 
           'mlsum_1X_10Labels_incumbent', 'mlsum_1X_11Labels_incumbent', 'mlsum_1X_2Labels_incumbent', 'mlsum_1X_3Labels_incumbent', 'mlsum_1X_4Labels_incumbent', 
           'mlsum_1X_5Labels_incumbent', 'mlsum_1X_6Labels_incumbent', 'mlsum_1X_7Labels_incumbent', 'mlsum_1X_8Labels_incumbent', 'mlsum_1X_9Labels_incumbent', 
           'mtop_domain_incumbent', 'mtop_domain_1X_10Labels_incumbent', 'mtop_domain_1X_2Labels_incumbent', 'mtop_domain_1X_3Labels_incumbent', 
           'mtop_domain_1X_4Labels_incumbent', 'mtop_domain_1X_5Labels_incumbent', 'mtop_domain_1X_6Labels_incumbent', 'mtop_domain_1X_7Labels_incumbent', 
           'mtop_domain_1X_8Labels_incumbent', 'mtop_domain_1X_9Labels_incumbent', 'multilingual-sentiments_incumbent', 'multilingual-sentiments_1X_2Labels_incumbent', 
           'senti_lex_incumbent', 'swiss_judgment_prediction_incumbent', 'tagesschau_incumbent', 'tagesschau_1X_2Labels_incumbent', 'tagesschau_1X_3Labels_incumbent', 
           'tagesschau_1X_4Labels_incumbent', 'tagesschau_1X_5Labels_incumbent', 'tagesschau_1X_6Labels_incumbent', 'tweet_sentiment_multilingual_incumbent', 
           'tweet_sentiment_multilingual_1X_2Labels_incumbent', 'x_stance_incumbent', 'x_stance_1X_2Labels_incumbent', 'x_stance_1X_3Labels_incumbent', 
           'x_stance_1X_4Labels_incumbent', 'x_stance_1X_5Labels_incumbent', 'x_stance_1X_6Labels_incumbent', 'x_stance_1X_7Labels_incumbent', 'x_stance_1X_8Labels_incumbent', 
           'x_stance_1X_9Labels_incumbent']

dataset_names=['multilingual-sentiments', 'miam', 'german_argument_mining', 'tagesschau', 'financial_phrasebank_75agree_german', 'tweet_sentiment_multilingual', 
               'mlsum', 'hatecheck-german', 'gnad10', 'mtop_domain', 'x_stance', 'senti_lex', 'Bundestag-v2', 'swiss_judgment_prediction', 'gnad10_1X_2Labels', 
               'miam_1X_24Labels', 'miam_1X_19Labels', 'tagesschau_1X_2Labels', 'mtop_domain_1X_8Labels', 'miam_1X_26Labels', 'tagesschau_1X_4Labels', 
               'gnad10_1X_4Labels', 'miam_1X_22Labels', 'x_stance_1X_8Labels', 'miam_1X_9Labels', 'mtop_domain_1X_10Labels', 'german_argument_mining_1X_2Labels', 
               'miam_1X_20Labels', 'gnad10_1X_6Labels', 'tagesschau_1X_6Labels', 'miam_1X_27Labels', 'tagesschau_1X_3Labels', 'miam_1X_18Labels', 'gnad10_1X_3Labels', 
               'miam_1X_25Labels', 'mtop_domain_1X_9Labels', 'german_argument_mining_1X_3Labels', 'miam_1X_8Labels', 'financial_phrasebank_75agree_german_1X_2Labels', 
               'x_stance_1X_9Labels', 'miam_1X_21Labels', 'gnad10_1X_7Labels', 'gnad10_1X_5Labels', 'miam_1X_23Labels', 'multilingual-sentiments_1X_2Labels', 
               'tagesschau_1X_5Labels', 'miam_1X_6Labels', 'hatecheck-german_1X_6Labels', 'x_stance_1X_7Labels', 'mtop_domain_1X_3Labels', 'miam_1X_12Labels', 
               'miam_1X_10Labels', 'x_stance_1X_5Labels', 'tweet_sentiment_multilingual_1X_2Labels', 'Bundestag-v2_1X_2Labels', 'hatecheck-german_1X_4Labels', 
               'miam_1X_4Labels', 'mtop_domain_1X_5Labels', 'miam_1X_30Labels', 'Bundestag-v2_1X_6Labels', 'miam_1X_29Labels', 'miam_1X_14Labels', 'miam_1X_16Labels',
               'miam_1X_2Labels', 'Bundestag-v2_1X_4Labels', 'hatecheck-german_1X_2Labels', 'mtop_domain_1X_7Labels', 'x_stance_1X_3Labels', 'miam_1X_11Labels', 
               'Bundestag-v2_1X_3Labels', 'hatecheck-german_1X_5Labels', 'miam_1X_5Labels', 'x_stance_1X_4Labels', 'mtop_domain_1X_2Labels', 'x_stance_1X_6Labels',
               'miam_1X_7Labels', 'gnad10_1X_8Labels', 'miam_1X_13Labels', 'miam_1X_17Labels', 'x_stance_1X_2Labels', 'mtop_domain_1X_6Labels', 'miam_1X_3Labels', 
               'Bundestag-v2_1X_5Labels', 'hatecheck-german_1X_3Labels', '1mlsum', 'mtop_domain_1X_4Labels', 'miam_1X_15Labels', 'miam_1X_28Labels']


core_datasets=['miam', 'swiss_judgment_prediction', 'x_stance', 'financial_phrasebank_75agree_german',
                 'hatecheck-german', 
                 #'mlsum', 
                 'german_argument_mining', 'Bundestag-v2', 'tagesschau',
                 # 'tyqiangz', 
                 'omp', 'senti_lex', "multilingual-sentiments", 'mtop_domain', 'gnad10',"tweet_sentiment_multilingual"]
def get_dataset_names(folder_path, exclude_folder):
    # Get list of all files in the folder
    files = os.listdir(folder_path)
    # Filter out directories and exclude specific folder
    dataset= [string for string in files if not (string.startswith('.') or string == exclude_folder )]
    return dataset


def evaluate_config(config, dataset,config_loc='IncumbentConfigs', dataset_loc='cleaned_datasets'):
   # Run the pipeline against the dataset and get the performance
   # store the performance in a under the pipeline column and dataset row metadata.json with the config and dataset name with keys added incumbent of and dataset name trained on
   """
   :param config: name of the incumbent configuration to be evaluated
   :param dataset: the name of the test dataset to be evaluated against
   :return: the performance of the pipeline on the dataset
   """
   config_file=os.path.join(config_loc,f'{config}.yaml')
   try:
      with open(config_file) as in_stream:
         metadata = yaml.safe_load(in_stream)
         data_dir=os.path.join(os.getcwd(),'tokenized_data' ,dataset) # tokenised dataset
         test_results=train_single_config(config=metadata, task_name=dataset, data_dir=data_dir, budget=1, train=False)
         return test_results['metrics']['test_f1_epoch']
   except FileNotFoundError:
    print(f"Metadata file not found at {config_file}")
    exit(1)
   except Exception as e:
      print(f"Other Error Here {e}")
      return -1.0

def make_evaluation_configs(pipelines, dataset_names, config_loc='IncumbentConfigs', dataset_loc='cleaned_datasets', save_loc='Evaluations'):
    # for each pipeline, run it against all the datasets. pipelines is the column, dataset is the row
    # for each dataset, run the pipeline and get the performance
    perf_matrix=pd.DataFrame(index=dataset_names, columns=pipelines)
    # set datatype to float
    perf_matrix=perf_matrix.astype(float)
    for index, row in perf_matrix.iterrows():
        for col_name, col_val in row.items():
        # Get the incumbent config using the col_name. 
        # Get the value using evaluate config
            new_val = evaluate_config(config=col_name, dataset=index, config_loc=config_loc, dataset_loc=dataset_loc)
            print(f"Performance of {col_name} on {index}: {new_val}")

        # Set the new value to the cell
            perf_matrix.at[index, col_name] = new_val
    
    return perf_matrix


def show_heatmap(perf_matrix,filename='heatmap'):
  
 # Create a heatmap
  sns.heatmap(perf_matrix, annot=True, cmap='viridis', cbar=True, fmt=".2f")
  plt.title('Heatmap Dataset V Pipeline')
    # Add labels
  plt.xticks(np.arange(len(perf_matrix.columns)), perf_matrix.columns)
  plt.yticks(np.arange(len(perf_matrix.index)), perf_matrix.index)  
  plt.title('Heatmap Dataset V Pipeline')
  plt.savefig(f'{filename}.png')
  perf_matrix.to_csv('performance_matrix.csv')

def show_diag(perf_matrix):
  for i, row_name in enumerate(dataset_names):
    diagonal_element = perf_matrix.at[row_name, f"{row_name}_inc_line"]
    print(f"Diagonal element at ({row_name}, {row_name}_inc_line): {diagonal_element}")


def init_p_mat(dataset_names, pipelines):
    perf_matrix=pd.DataFrame(index=dataset_names, columns=pipelines)
    for index, row in perf_matrix.iterrows():
     for col_name, col_val in row.items ():
         new_val = 0
         perf_matrix.at[index, col_name] = new_val
    
    perf_matrix=perf_matrix.astype(float)
    return perf_matrix

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate the performance matrix from the run results')
    parser.add_argument('--save_location',type=str, 
                        help='Directory with run results pkl file',default='Evaluations')
    parser.add_argument('--dataset_loc',type=str, help='Directory with dataset files',default='cleaned_datasets')
    parser.add_argument('--pipeline_loc',type=str, help='Directory with pipeline config files',default='IncumbentConfigs')
    args = parser.parse_args()
    folder_path = os.path.join(os.getcwd(),args.dataset_loc)
    exclude_folder = 'Augmented'
    dataset_names = get_dataset_names(folder_path, exclude_folder)
    folder_path = os.path.join(folder_path , "Augmented")
    dataset_names.extend(get_dataset_names(folder_path, exclude_folder))
    # names for the matrix row and columns
    dataset_names=sorted(dataset_names) # 97 datasets
    pipelines = list(map(lambda x: x + '_incumbent', dataset_names)) # 97 pipelines


    # populate the matrix with the performance values
    config_loc= os.path.join(os.getcwd(),'HPO/ray_cluster_test',args.pipeline_loc)
    dataset_loc= os.path.join(os.getcwd(), args.dataset_loc)
    p_mat = make_evaluation_configs(pipelines, dataset_names, config_loc=config_loc, dataset_loc=dataset_loc)

   # pipeline is the incumbent pipeline for that dataset. stored under IncumbentConfigs/dataset/file.yaml
    
    show_heatmap(p_mat,filename='heatmap') 
   
