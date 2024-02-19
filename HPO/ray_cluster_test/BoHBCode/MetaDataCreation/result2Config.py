# Created date: 2021-05-20
import argparse
from ast import mod
import os
from pyexpat import model
import re
import hpbandster.core.result as hpres
import yaml
from copy import deepcopy

"""
Read the result.pkl for the runs and convert the best config to a yaml file
The results location needs the configs, results jsons and the .pkl file.
:param working_dir: the directory with the results
"""

def incumbent_to_yaml(incumbent_config, default_config):
    # write the incumbent to the format as default
    mc = deepcopy(default_config)
    mc['model_config']['model'] = incumbent_config['model_name_or_path']
    mc['model_config']['optimizer']['type'] = incumbent_config['optimizer_name']
    mc['model_config']['optimizer']['lr']=incumbent_config['learning_rate']
    mc['model_config']['optimizer']['scheduler']=incumbent_config['scheduler_name']
    mc['model_config']['optimizer']['weight_decay']=incumbent_config['weight_decay']
    mc['model_config']['optimizer']['adam_epsilon']=incumbent_config['adam_epsilon']

    mc['model_config']['training']['warmup']=incumbent_config['warmup_steps']
    mc['model_config']['training']['gradient_accumulation']=incumbent_config['gradient_accumulation_steps']

    mc['model_config']['dataset']['name']=''
    mc['model_config']['dataset']['seq_length'] = incumbent_config['max_seq_length']
    mc['model_config']['dataset']['batch']= incumbent_config['per_device_train_batch_size']

    return mc

def create_yaml(working_dir, dataset='gnad10', result_dir='IncumbentConfigs'):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(working_dir)
    id2conf = result.get_id2config_mapping()

    # (best configuration)
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']

    # Read the default config
    default_config_path=os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/MetaDataCreation','default.yaml')
    with open(default_config_path) as in_stream:
        default_config = yaml.safe_load(in_stream)
    
    format_incumbent = incumbent_to_yaml(inc_config, default_config)
    format_incumbent['model_config']['dataset']['name'] = f'{dataset}'
    # model and dataset are the last part of the name
    model_name = format_incumbent['model_config']['model'].split('/')[-1]
    dataset_name = format_incumbent['model_config']['dataset']['name'].split('/')[-1]
    # create the output folder
    if not os.path.exists(os.path.join(os.getcwd(),result_dir,dataset)):
        os.makedirs(os.path.join(os.getcwd(),result_dir,dataset))
    output_path = os.path.join(os.getcwd(),result_dir,dataset, f"{model_name}_{dataset_name}_incumbent.yaml")
    with open(output_path, "w+") as out_stream:
        yaml.dump(format_incumbent, out_stream)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Run to Yaml')
    parser.add_argument('--result_directory',type=str, 
                        help='Directory with run results pkl file',default='bohb_gnad10_seed_9_150_trials')
    parser.add_argument('--dataset',type=str, 
                        help='Dataset name',default='gnad10')
    parser.add_argument('--save_location',type=str, 
                        help='Directory with run results pkl file',default='IncumbentConfigs')
    # store the result under IncumbentConfigs/dataset/file.yaml 
    args = parser.parse_args()
    # where all the run artifacts are kept
    
    working_dir = os.path.join(os.getcwd(),'../datasetruns' ,args.result_directory)
    working_dir = os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/datasetruns' ,args.result_directory)
    create_yaml(working_dir=working_dir, dataset=args.dataset, result_dir=args.save_location)

   