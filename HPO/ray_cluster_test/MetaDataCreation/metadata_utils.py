import os
import re
import shutil
import json
import random
import numpy as np
import pandas as pd


def generate_nearby_number(number, range_width=10):
    return number + random.randint(-range_width, range_width)


datasets_main = ['Bundestag-v2', 'Bundestag-v2_1X_2Labels', 'Bundestag-v2_1X_3Labels', 'Bundestag-v2_1X_4Labels',
                 'Bundestag-v2_1X_5Labels',
                 'Bundestag-v2_1X_6Labels', 'financial_phrasebank_75agree_german',
                 'financial_phrasebank_75agree_german_1X_2Labels',
                 'german_argument_mining', 'german_argument_mining_1X_2Labels', 'german_argument_mining_1X_3Labels',
                 'gnad10', 'gnad10_1X_2Labels',
                 'gnad10_1X_3Labels', 'gnad10_1X_4Labels', 'gnad10_1X_5Labels', 'gnad10_1X_6Labels',
                 'gnad10_1X_7Labels',
                 'gnad10_1X_8Labels',
                 'hatecheck-german', 'hatecheck-german_1X_2Labels', 'hatecheck-german_1X_3Labels',
                 'hatecheck-german_1X_4Labels', 'hatecheck-german_1X_5Labels',
                 'hatecheck-german_1X_6Labels', 'miam', 'miam_1X_10Labels', 'miam_1X_11Labels', 'miam_1X_12Labels',
                 'miam_1X_13Labels', 'miam_1X_14Labels', 'miam_1X_15Labels', 'miam_1X_16Labels', 'miam_1X_17Labels',
                 'miam_1X_18Labels', 'miam_1X_19Labels', 'miam_1X_20Labels',
                 'miam_1X_21Labels', 'miam_1X_22Labels', 'miam_1X_23Labels', 'miam_1X_24Labels', 'miam_1X_25Labels',
                 'miam_1X_26Labels', 'miam_1X_27Labels', 'miam_1X_28Labels', 'miam_1X_29Labels', 'miam_1X_2Labels',
                 'miam_1X_30Labels', 'miam_1X_3Labels', 'miam_1X_4Labels', 'miam_1X_5Labels', 'miam_1X_6Labels',
                 'miam_1X_7Labels', 'miam_1X_8Labels', 'miam_1X_9Labels', 'mlsum', 'mlsum_1X_10Labels',
                 'mlsum_1X_11Labels',
                 'mlsum_1X_2Labels', 'mlsum_1X_3Labels', 'mlsum_1X_4Labels', 'mlsum_1X_5Labels', 'mlsum_1X_6Labels',
                 'mlsum_1X_7Labels', 'mlsum_1X_8Labels', 'mlsum_1X_9Labels', 'mtop_domain', 'mtop_domain_1X_10Labels',
                 'mtop_domain_1X_2Labels', 'mtop_domain_1X_3Labels', 'mtop_domain_1X_4Labels', 'mtop_domain_1X_5Labels',
                 'mtop_domain_1X_6Labels', 'mtop_domain_1X_7Labels', 'mtop_domain_1X_8Labels', 'mtop_domain_1X_9Labels',
                 'multilingual-sentiments', 'multilingual-sentiments_1X_2Labels', 'senti_lex',
                 'swiss_judgment_prediction',
                 'tagesschau', 'tagesschau_1X_2Labels', 'tagesschau_1X_3Labels', 'tagesschau_1X_4Labels',
                 'tagesschau_1X_5Labels', 'tagesschau_1X_6Labels', 'tweet_sentiment_multilingual',
                 'tweet_sentiment_multilingual_1X_2Labels', 'x_stance', 'x_stance_1X_2Labels', 'x_stance_1X_3Labels',
                 'x_stance_1X_4Labels', 'x_stance_1X_5Labels', 'x_stance_1X_6Labels', 'x_stance_1X_7Labels',
                 'x_stance_1X_8Labels', 'x_stance_1X_9Labels', 'financial_phrasebank_75agree_german_5X_2Labels',
                 'german_argument_mining_5X_2Labels', 'german_argument_mining_5X_3Labels',
                 'hatecheck-german_5X_2Labels',
                 'hatecheck-german_5X_3Labels', 'hatecheck-german_5X_4Labels', 'hatecheck-german_5X_5Labels',
                 'hatecheck-german_5X_6Labels', 'miam_5X_10Labels', 'miam_5X_11Labels', 'miam_5X_12Labels',
                 'miam_5X_13Labels', 'miam_5X_14Labels', 'miam_5X_15Labels', 'miam_5X_16Labels', 'miam_5X_17Labels',
                 'miam_5X_18Labels', 'miam_5X_19Labels', 'miam_5X_20Labels', 'miam_5X_21Labels', 'miam_5X_22Labels',
                 'miam_5X_23Labels', 'miam_5X_24Labels', 'miam_5X_25Labels', 'miam_5X_26Labels', 'miam_5X_27Labels',
                 'miam_5X_28Labels', 'miam_5X_29Labels', 'miam_5X_2Labels', 'miam_5X_30Labels', 'miam_5X_3Labels',
                 'miam_5X_4Labels', 'miam_5X_5Labels', 'miam_5X_6Labels', 'miam_5X_7Labels', 'miam_5X_8Labels',
                 'miam_5X_9Labels', 'mtop_domain_5X_10Labels', 'mtop_domain_5X_2Labels', 'mtop_domain_5X_3Labels',
                 'mtop_domain_5X_4Labels', 'mtop_domain_5X_5Labels', 'mtop_domain_5X_6Labels', 'mtop_domain_5X_7Labels',
                 'mtop_domain_5X_8Labels', 'mtop_domain_5X_9Labels', 'multilingual-sentiments_5X_2Labels',
                 'tweet_sentiment_multilingual_5X_2Labels', 'x_stance_5X_2Labels', 'x_stance_5X_3Labels',
                 'x_stance_5X_4Labels', 'x_stance_5X_5Labels', 'x_stance_5X_6Labels', 'x_stance_5X_7Labels',
                 'x_stance_5X_8Labels', 'x_stance_5X_9Labels', 'financial_phrasebank_75agree_german_10X_2Labels',
                 'german_argument_mining_10X_2Labels', 'german_argument_mining_10X_3Labels',
                 'hatecheck-german_10X_2Labels',
                 'hatecheck-german_10X_3Labels', 'hatecheck-german_10X_4Labels', 'hatecheck-german_10X_5Labels',
                 'hatecheck-german_10X_6Labels', 'miam_10X_10Labels', 'miam_10X_11Labels', 'miam_10X_12Labels',
                 'miam_10X_13Labels', 'miam_10X_14Labels', 'miam_10X_15Labels', 'miam_10X_16Labels',
                 'miam_10X_17Labels',
                 'miam_10X_18Labels', 'miam_10X_19Labels', 'miam_10X_20Labels', 'miam_10X_21Labels',
                 'miam_10X_22Labels',
                 'miam_10X_23Labels', 'miam_10X_24Labels', 'miam_10X_25Labels', 'miam_10X_26Labels',
                 'miam_10X_27Labels',
                 'miam_10X_28Labels', 'miam_10X_29Labels', 'miam_10X_2Labels', 'miam_10X_30Labels', 'miam_10X_3Labels',
                 'miam_10X_4Labels', 'miam_10X_5Labels', 'miam_10X_6Labels', 'miam_10X_7Labels', 'miam_10X_8Labels',
                 'miam_10X_9Labels', 'mtop_domain_10X_10Labels', 'mtop_domain_10X_2Labels', 'mtop_domain_10X_3Labels',
                 'mtop_domain_10X_4Labels', 'mtop_domain_10X_5Labels', 'mtop_domain_10X_6Labels',
                 'mtop_domain_10X_7Labels',
                 'mtop_domain_10X_8Labels', 'mtop_domain_10X_9Labels', 'multilingual-sentiments_10X_2Labels',
                 'tweet_sentiment_multilingual_10X_2Labels', 'x_stance_10X_2Labels', 'x_stance_10X_3Labels',
                 'x_stance_10X_4Labels', 'x_stance_10X_5Labels', 'x_stance_10X_6Labels', 'x_stance_10X_7Labels',
                 'x_stance_10X_8Labels', 'x_stance_10X_9Labels']

pipelines = []


def create_cv_folds(folder_path, num_folds=5):
    # load metadata training set for the datasets
    with open("/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv") as f:
        metadata = pd.read_csv(f)
        datasets = metadata['Dataset'].unique()
    # create a list of dataset names as series
    dataset_series = pd.Series(datasets)
    shuffled_dataset_names = dataset_series.sample(frac=1, random_state=42).reset_index(drop=True)
    split_parts = np.array_split(shuffled_dataset_names, num_folds)
    combined_df = pd.concat([part.to_frame(name='Dataset') for part in split_parts], ignore_index=True)
    combined_df['cv_fold'] = np.concatenate([np.full(len(part), i + 1) for i, part in enumerate(split_parts)])
    combined_df.to_csv(os.path.join(folder_path, 'cv_folds.csv'), index=False)




def relabel(file_path):
    # for all .yaml files in the directory, rename them to the format: dataset_incumbent.yaml, dataset from the
    # datasets list
    current_filenames = os.listdir(file_path)
    for current_name, new_name in zip(current_filenames, datasets_main):
        current_file_path = os.path.join(file_path, current_name)
        filename, extension = os.path.splitext(current_name)
        new_filename = f'{new_name}_incumbent' + extension
        new_file_path = os.path.join(file_path, new_filename)
        os.rename(current_file_path, new_file_path)
        print(f"Renamed {current_file_path} to {new_file_path}")

def metadatafolders(folder_path):
    pattern = re.compile(r'(.+?)_(\d+)X_(\d+)Labels') #re.compile(r'.*?_1X_(\d+)Labels')
    augment_datasets=['x_stance', 'tweet_sentiment_multilingual',
                      'mtop_domain','mlsum','senti_lex',
                      'swiss_judgment_prediction','tyqiangz',
                      'miam','hatecheck-german','german_argument_mining',
                      'financial_phrasebank_75agree_german',
                      'Bundestag-v2','gnad10'
                      'tagesschau']
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs :
            match = pattern.match(dir_name)
            if match and match.group(1) in augment_datasets:
                prefix = match.group(1)
                label = match.group(3)
                new_folder_name = f"{prefix}_10X_{label}Labels"
           
                new_folder_path = os.path.join(root, new_folder_name)
                # Create the new folder if it doesn't exist
                os.makedirs(new_folder_path, exist_ok=True)
                metadata_file = os.path.join(root, dir_name, 'metadata.json')
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata['num_training_samples'] =  metadata['num_training_samples'] * 10
                metadata["tokenize_folder_name"] = new_folder_name
                metadata['task_name']= new_folder_name
                metadata["average_text_length"] = generate_nearby_number(metadata["average_text_length"])


                new_metadata_file = os.path.join(new_folder_path, 'metadata.json')
                with open(new_metadata_file, 'w') as f:
                    json.dump(metadata, f,indent=4)

def create_metafeatures_csv(folder_path):
    """
    :param folder_path: path to the top folder containing the metadata.json files
    """
    df = pd.DataFrame()
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file == 'metadata.json':
                # Read metadata.json
                metadata_file_path = os.path.join(subdir, file)
                with open(metadata_file_path) as f:
                    metadata = json.load(f)
                folder_name = os.path.basename(subdir)
                # Append metadata to DataFrame
                df[folder_name] = [metadata[key] for key in ["num_training_samples", "num_labels","average_text_length"]]
    df.index = [metadata['task']]

    # Transpose DataFrame
    df = df.transpose()
    # Save DataFrame to CSV
    df.to_csv(os.path.join(folder_path, 'metafeatures.csv'))

def read_metadata_files_to_dataframe(folder_path):
    metadata_list = []

    # Iterate through all files and directories in the folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Check if the file is named metadata.json
            if file_name == 'metadata.json':
                file_path = os.path.join(root, file_name)
                # Read the contents of the JSON file
                with open(file_path, 'r') as file:
                    metadata = json.load(file)
                    metadata_list.append(metadata)
    
    # Convert the list of dictionaries to a DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    # drop columns named 'task' and 'task_name'
    metadata_df.drop(columns=["tokenize_folder_name", "name","label_col"], inplace=True)
    metadata_df.set_index('task_name', inplace=True)
    metadata_df.to_csv('meta_features.csv', index_label=False)
    return metadata_df

# /Users/diptisengupta/Desktop
if __name__ == "__main__":    

    # get the datasets from datasets.txt as a list, remove the newline characters and all leading and trailing whitespaces
    with open("/Users/diptisengupta/Desktop/datasets.txt") as f:
        datasets = f.readlines()
        datasets = [dataset.strip() for dataset in datasets]
    # print(f"{datasets}")

    # relabel("/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/1ncumbentConfigs")
    # metadatafolders('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/cleaned_datasets')

    folder_path = '/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/cleaned_datasets'
    metadata_df = read_metadata_files_to_dataframe(folder_path) 
