"""
Data Augmentation
"""
from curses import meta
import os
import pandas as pd
from datasets import load_from_disk, load_dataset
import json
import random
from datasets import ClassLabel, Value





# Function to augment dataset by removing specific labels
def augment_dataset(original_df, labels_to_keep):
     # for a huggingface dataset, remove the given labels and relabel the dataset
    labels_to_keep.sort()
    for split in original_df:
        # update the features
        features = original_df[split].features.copy()
        og_labels = features['labels'].names
        select_labels = [og_labels[i] for i in labels_to_keep]
        features['labels'] = ClassLabel(names=select_labels)
        # filter the dataset
        original_df[split]=original_df[split].filter(lambda example: example['labels'] in labels_to_keep)
        # Re-label remaining rows        
        label_map = {label: index for index, label in enumerate(sorted(set(original_df[split]['labels'])))}
        original_df[split] = original_df[split].map(lambda example: {'labels': label_map[example['labels']], 'sentence': example['sentence']})

        original_df[split] = original_df[split].cast(features)
        # print(f" Feat MAP {original_df['train'].features['labels']._str2int}")
    return original_df

def relabel_dataset():
    original_data = {
    'sentence': ['This is sentence 1', 'Another sentence here', 'And one more sentence','Another sentence here1','Another sentence here2'],
    'label': [0, 1, 2,2,0]
    }
    original_df = pd.DataFrame(original_data)

    # Specify the labels you want to remove
    labels_to_remove = [1, 2]

    # Augment the dataset
    augmented_df = augment_dataset(original_df, labels_to_remove)

    # Display the augmented dataset
    print("Original Dataset:")
    print(original_df)
    print("\nAugmented Dataset with labels {} removed:".format(labels_to_remove))
    print(augmented_df)

def calculate_metadata(dataset):
    # Calculate the number of training samples
    num_training_samples = len(dataset["train"])

    # Calculate the number of unique labels
    num_labels = dataset["train"].features["labels"].num_classes

    # Calculate the average text length
    average_text_length = sum(len(example["sentence"].split()) for example in dataset["train"]) / num_training_samples

    return {
        "num_training_samples": num_training_samples,
        "num_labels": num_labels,
        "average_text_length": average_text_length,
        "label_col": "labels",
    }

def add_metadata(folder_path):
    # Iterate over subdirectories (datasets) in the folder
    for dataset_name in os.listdir(folder_path):
        dataset_path = os.path.join(folder_path, dataset_name)

        # Check if the path is a directory
        if os.path.isdir(dataset_path):
            # Load the dataset
            dataset = load_from_disk(dataset_path)
            metadata = calculate_metadata(dataset)
            metadata["tokenize_folder_name"] = dataset_name
            metadata["task_name"] = dataset_name

            # Write metadata to a JSON file
            with open(os.path.join(dataset_path,'metadata.json'), 'w') as json_file:
                json.dump(metadata, json_file, indent=4)


def start_aug_datasets(data_dir):
    for dataset_name in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_name)
        # Check if the path is a directory
        if os.path.isdir(dataset_path) and dataset_name != "Augmented":
            vary_labels(data_dir, dataset_name)

def vary_labels(data_dir, name):
    # for each dataset, pick labels to keep and remove the rest. 
    # proceed till theres only a dataset with 2 labels.

    dataset = load_from_disk(os.path.join(data_dir, name))
    # num_labels = dataset["train"].features["labels"].num_classes
    labels = set(dataset["train"]["labels"])
    target_label_counts = range(len(labels) - 1, 1, -1)

    for target_count in target_label_counts:
    # Randomly select labels to keep
        labels_to_keep = random.sample(labels, target_count)
        # load a clean copy of the dataset
        dataset = load_from_disk(os.path.join(data_dir, name))
        augmented_dataset = augment_dataset(dataset, labels_to_keep=labels_to_keep)
        aug_dataset_name = name + "_1X_" +str(len(labels_to_keep))+ "Labels"
        metadata = calculate_metadata(dataset)
        metadata["tokenize_folder_name"] = aug_dataset_name
        metadata["task_name"] = aug_dataset_name
        # print labels 
        print(f'Augemented Labels--->{augmented_dataset["train"].features["labels"].names}')
        augmented_dataset.save_to_disk(os.path.join(data_dir,"Augmented" ,aug_dataset_name))
                    # Write metadata to a JSON file
        with open(os.path.join(data_dir,"Augmented" ,aug_dataset_name,'metadata.json'), 'w') as json_file:
                json.dump(metadata, json_file, indent=4)
    

def datastest():
    # load yelp reviews dataset
    dataset = load_dataset("yelp_review_full")
    # remove all the labels except 2,3,4
    labels_to_keep = [2,3,4]
    # rename label column to labels and text to sentence
    dataset = dataset.rename_column("label", "labels")
    dataset=dataset.rename_column("text", "sentence")
    dataset = augment_dataset(dataset, labels_to_keep=labels_to_keep)
    print(dataset["train"].features["labels"].names)


if __name__ == "__main__":
    dataset_list = ['Bundestag-v2', 'tagesschau', 'german_argument_mining', 
                    'mlsum', 'hatecheck-german', 'financial_phrasebank_75agree_german', 
                    'x_stance', 'swiss_judgment_prediction', 'miam']
    data_folder = "Bundestag-v2"
    data_dir = os.path.join(os.getcwd(),"cleaned_datasets")
    start_aug_datasets(data_dir)
    # datastest()
    # mlsum 
   


