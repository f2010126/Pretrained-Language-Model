
import argparse
from ast import parse
from logging import config
import re
import sched
import json
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the Surrogate Model by predicting the data")
   
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="germeval2018_fine")
    parser.add_argument("--model_path", type=str, default="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/best_metamodel_cvfold_3_loss_bpr.pkl")
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
