# Description: This script is used to evaluate the performance of the baselines on the test set.
import argparse
from ast import parse
from math import e
import yaml
import os
import sys
from datasets import load_dataset, DatasetDict
from transformers import pipeline, AutoTokenizer
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import lightning as pl
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))

from BoHBCode.train_module import PLMTransformer


def processTestData():
    dataset = load_dataset("gwlms/germeval2018")
    dataset = dataset.class_encode_column("fine-grained")
    dataset = dataset.rename_column('fine-grained', "labels")
    dataset = dataset.rename_column('text', "sentence")
    dataset = dataset.remove_columns(["coarse-grained"])
    return dataset



def evaluateModel(config):
    model_config = {'model_name_or_path': config['model_name_or_path'],
                        'optimizer_name':config['optimizer_name'],
                    'learning_rate': config['learning_rate'],
                    'scheduler_name': config['scheduler_name'],
                    'weight_decay': config['weight_decay'],
                    'sgd_momentum': 0.9,
                    'warmup_steps': config['warmup_steps'],}
    model = PLMTransformer(
        config=model_config, 
        num_labels=4,)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
    model.eval()
    germeval_2018 = processTestData()
        
    batch_size = config['per_device_train_batch_size']  
    test_dataloader = DataLoader(germeval_2018['test'], batch_size=batch_size)
    predictions = []
    true_labels = []
    def preprocess_text(text):
            tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            return tokens
    with torch.no_grad():
         for batch in tqdm(test_dataloader, desc="Predicting", unit="batch", leave=False):
            inputs = preprocess_text(batch["sentence"])
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
        
            predictions.extend(predicted_labels.tolist())
            true_labels.extend(batch["labels"].tolist())
        
    
    f1 = f1_score(true_labels, predictions, average="weighted")
    print("F1 Score:", f1)
    return f1


def evaluateBest(seed=42):
    best='german_argument_mining_incumbent'
    yaml_path='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs'
    with open(f'{yaml_path}/{best}.yaml') as f:
        best_config = yaml.load(f, Loader=yaml.FullLoader)
        print(best_config)
    f1 = evaluateModel(best_config)
    return f1


def evaluateRandom(seed=42):
    yaml_path='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs'
    yaml_files = [file for file in os.listdir(yaml_path) if file.endswith('.yaml')]
    if not yaml_files:
        raise FileNotFoundError("No YAML files found in the folder.")

    # Choose a random YAML file from the list
    random_yaml_file = random.choice(yaml_files)

    # Load the contents of the random YAML file
    with open(os.path.join(yaml_path, random_yaml_file), 'r') as file:
        yaml_contents = yaml.safe_load(file)
    
    f1=evaluateModel(yaml_contents)
    return f1


def evaluateGluon(seed=42):
    best='german_argument_mining_incumbent'
    yaml_path='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs'
    with open(f'{yaml_path}/{best}.yaml') as f:
        best_config = yaml.load(f, Loader=yaml.FullLoader)
        print(best_config)
    f1 = evaluateModel(best_config)
    return f1

def evaluateMetamodel():
    return 0

def evaluateGluon(seed=42):
    best='financial_phrasebank_75agree_german_incumbent'
    yaml_path='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/Full_IncumbentConfigs'
    with open(f'{yaml_path}/{best}.yaml') as f:
        best_config = yaml.load(f, Loader=yaml.FullLoader)
        print(best_config)
    f1 = evaluateModel(best_config)
    return f1

def box_plots(performance_metrics):
    f1_scores = [[metrics['f1_score'] for metrics in performance_metrics[baseline].values()] for baseline in baselines]
    # Plot the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(f1_scores, labels=baselines)
    plt.xlabel('Baseline Models')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Baseline Model')
    plt.grid(True)
    plt.savefig(f'Boxplots.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Baseline')
    parser.add_argument('--baseline', type=str,
                        default="autogluon", help='best|random|metamodel|autogluon')
    args = parser.parse_args()
    seeds =[42, 55, 8, 78, 123]
    baselines=['autogluon', 'best', 'random']
    performance_metrics = defaultdict(dict)
    for baseline, seed in itertools.product(baselines, seeds):
        print(f"Running {baseline} with seed {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        pl.seed_everything(seed)
        if baseline == "best":
            performance_metrics[baseline][seed] = evaluateBest(seed)
        elif baseline == "random":
            performance_metrics[baseline][seed]=evaluateRandom(seed)
        elif baseline == "autogluon":
            performance_metrics[baseline][seed]=evaluateGluon(seed)

    
    print(performance_metrics)
    # save the performance metrics
    with open('performance_metrics.yaml', 'w') as file:
        yaml.dump(performance_metrics, file)
    box_plots(performance_metrics)
        



