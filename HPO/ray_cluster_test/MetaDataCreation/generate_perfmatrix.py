
"""
This script is used to generate the performance matrix from the run results.
for each incumbent configuration, it will run it against the test data of all the datasets and generate the rows.
The dataset can be in the cleaned_datasets folder or the cleaned_datasets/augmented folder.

"""
from ast import main, pattern
import re
import os
import argparse
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

from metadata_utils import datasets_main
np.random.seed(42)

def get_best_incumbent(perf_matrix):
   performance_sum = perf_matrix.sum(axis=0)
   average_performance = performance_sum / len(perf_matrix.index)
   best_model = average_performance.idxmax()
   print("The best model across all datasets is:", best_model)

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

def save_perf_matrix(perf_matrix, filename='performance_matrix'):
   # saves correctly
   perf_matrix.to_csv(f'{filename}_performance_matrix.csv', index_label=False)
   return f'{filename}_performance_matrix.csv'

import re
def get_tick_names(perf_matrix):
   dataset_names = perf_matrix.index
   pattern = re.compile(r'_\d+X_\d+Labels$')
   main_dataset_names = [name for name in dataset_names if not pattern.search(name)]
   main_dataset_names_with_placeholder = [name if name in main_dataset_names else ' ' for name in perf_matrix.index]
   main_dataset_incumbents = [name + '_incumbent' for name in main_dataset_names]
   main_dataset_incumbents_with_placeholder = [incumbent if incumbent in main_dataset_incumbents else ' ' for incumbent in perf_matrix.columns]
   return main_dataset_names_with_placeholder,main_dataset_incumbents_with_placeholder

def show_heatmap(perf_matrix,filename='heatmap'):
 # Create a heatmap
  plt.figure(figsize=(8, 6))
  main_dataset_names, main_dataset_x = get_tick_names(perf_matrix)
  sns.heatmap(perf_matrix, cmap='magma', cbar=True, fmt=".6f",yticklabels=main_dataset_names, xticklabels=main_dataset_x)
  plt.xlabel('Dataset Incumbents')
  plt.ylabel('Datasets')
  plt.title('Performance Matrix, F1 Metric',fontsize=16, fontweight='bold')
  main_dataset_names, main_dataset_indices = get_tick_names(perf_matrix)
  # plt.yticks(main_dataset_indices, main_dataset_names)

  plt.savefig(f'{filename}.png')
  plt.show()

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
   
