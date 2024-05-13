
import argparse
from ast import parse
from logging import config
import re
import sched
import json
from unittest import loader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import ndcg_score

# local 
from metamodel_train import MLP
from metamodel_data import get_predict_loader


def get_loader(task, batch_size, seed):
    if task == "germeval2018_fine":
        t_path="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_germeval2018_fine.csv"
    elif task == "germeval2018_coarse":
        t_path="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_germeval2018_coarse_test.csv"
    elif task == "germeval2019_fine":
        t_path="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_germeval2019_fine.csv"
    elif task == "germeval2019_coarse":
        t_path="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_germeval2019_coarse.csv"
    elif task == "germeval2017":
        t_path="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_germeval2017.csv"


    loader = get_predict_loader(task=task, batch_size=batch_size, seed=seed, t_path=t_path)
    return loader

def predict_model(model, task, batch_size, seed):
    model.eval()
    loader=get_loader(task, batch_size, seed)
    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch
            predictions = model(inputs)
            all_predictions.append(predictions.cpu().numpy())  # Convert predictions to numpy array and move to CPU

    all_predictions = np.concatenate(all_predictions, axis=0)
    max_index= np.argmax(all_predictions)
    print(f"index of max prediction: {max_index} with value: {np.max(all_predictions)}")
    pipeline=loader.dataset.pipelines['test'][max_index]
    print(f"Pipeline: {pipeline}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the Surrogate Model by predicting the data")
   
    parser.add_argument("--batch_size", type=int, default=204)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="germeval2017")
    parser.add_argument("--model_path", type=str, default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/best_metamodel_cvfold_2_loss_regression.pkl")
    parser.add_argument("--config", type=str, default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/best_regression.json")
    args=parser.parse_args()

    input_size = 27 # number of features encoded + dataset
    hidden_size = 64
    output_size = 1 # performance

   
    # load model into MLP
    with open(args.config, "r") as f:
        config = json.load(f)
    try:
        model=MLP(input_size=input_size, output_size=output_size,
                  num_hidden_layers=config['num_hidden_layers'], 
                  neurons_per_layer=config['num_hidden_units'], 
                  dropout_prob=config['dropout_rate'])
        weights= torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)  # Load the model's state dictionary

    except FileNotFoundError:
            print("Model not found")
    
    print("Model loaded")
    predict_model(model, args.task, args.batch_size, args.seed)
