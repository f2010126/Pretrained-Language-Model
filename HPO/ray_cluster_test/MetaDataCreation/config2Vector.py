"""
Convert the hyperparementer yaml file to a training vector
"""
# convert the config yaml to a vector 
import os
import sys
import yaml
from pathlib import Path
import pandas as pd


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

"""
:param config_path: the path to the yaml file
:return: a dictionary of the hyperparameters
"""
def create_vector(config_path):
    with open(config_path) as stream:
         model_config = yaml.safe_load(stream)
    
    config = model_config['model_config']
    temp_config = dict()
    temp_config.update(config['model'])
    temp_config.update(config['optimizer'])
    temp_config.update(config['training'])
    temp_config.update(config['dataset'])
    config = temp_config

    # add the numerical hps to the vector
    hp_vector = []

    hp_vector.append(config['lr'])
    hp_vector.append(config['momentum'])
    hp_vector.append(config['weight_decay'])
    hp_vector.append(config['adam_epsilon'])
    hp_vector.append(config['warmup'])
    hp_vector.append(config['gradient_accumulation'])
    hp_vector.append(config['seq_length'])
    hp_vector.append(config['batch'])

    # Categorical
    hp_vector += ENCODE_MODELS[config['model']] 
    hp_vector += ENCODE_OPTIM[config['type']] 
    hp_vector += ENCODE_SCHED[config['scheduler']] 

    return dict(zip(hp_vector))
     


if __name__ == "__main__":
       create_vector()
