# Created date: 2021-05-20
import argparse

import os

import hpbandster.core.result as hpres
import yaml

from copy import deepcopy
import json

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

    mc['model_config']['dataset']['seq_length'] = incumbent_config['max_seq_length']
    mc['model_config']['dataset']['batch']= incumbent_config['per_device_train_batch_size']

    return mc

def create_yaml(working_dir, dataset='gnad10', result_dir='IncumbentConfigs', metadata = None):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(working_dir)
    id2conf = result.get_id2config_mapping()

    # (best configuration)
    inc_id = result.get_incumbent_id()
    inc_runs=result.get_runs_by_id(inc_id)
    run_info=[]
    for run in inc_runs:
        run_info.append({'budget':run.budget, 'info':run.info})

    inc_config = id2conf[inc_id]['config']

    # Read the default config
    default_config_path=os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/MetaDataCreation','default.yaml')
    with open(default_config_path) as in_stream:
        default_config = yaml.safe_load(in_stream)
    
    format_incumbent = incumbent_to_yaml(inc_config, default_config)
    # add the metadata
    format_incumbent['incumbent_for'] = metadata['name'] # its from the datasetruns so HPO
    format_incumbent['model_config']['dataset']['name']=metadata['name'] # That it trained on. NOT THE SAME AS THE INCUMBENT where it is best.
    format_incumbent['model_config']['dataset']['num_labels'] = metadata['num_labels']
    format_incumbent['model_config']['dataset']['average_text_length'] = metadata['average_text_length']
    format_incumbent['model_config']['dataset']['num_training_samples'] = metadata['num_training_samples']
    format_incumbent['run_info']=json.dumps(run_info)
    # model and dataset are the last part of the name
    model_name = format_incumbent['model_config']['model'].split('/')[-1]
    dataset_name = format_incumbent['model_config']['dataset']['name'].split('/')[-1]

    # create the output folder
    if not os.path.exists(os.path.join(os.getcwd(),result_dir,dataset)):
        os.makedirs(os.path.join(os.getcwd(),result_dir,dataset))
    output_path = os.path.join(os.getcwd(),result_dir,dataset, f"{model_name}_{dataset_name}_incumbent.yaml")
    with open(output_path, "w+") as out_stream:
        yaml.dump(format_incumbent, out_stream)


def read_pkls(working_dir, result_dir):
    dataset_strings=['miam', 'swiss_judgment_prediction', 'x_stance', 'financial_phrasebank_75agree_german',
                 'hatecheck-german', 
                 #'mlsum', 
                 'german_argument_mining', 'Bundestag-v2', 'tagesschau',
                 # 'tyqiangz', 
                 'omp', 'senti_lex', "multilingual-sentiments", 'mtop_domain', 'gnad10',"tweet_sentiment_multilingual"]
    for folder_name, _, files in os.walk(working_dir):
        for file_name in files:
            if file_name.endswith('.pkl') and any(string in file_name for string in dataset_strings):
                # pkl is here, get the dataset name
                dataset_name = [string for string in dataset_strings if string in file_name]
                # load the metadata
                metadata_loc = os.path.join(os.getcwd(),args.metadata_loc, dataset_name[0], 'metadata.json')
                file_path = os.path.join(folder_name, file_name)
                try:
                    with open(metadata_loc) as in_stream:
                        metadata = yaml.safe_load(in_stream)
                        print(f"Data loaded from {file_path}: {metadata}")
                        create_yaml(working_dir=folder_name, dataset=dataset_name[0], result_dir=result_dir, metadata=metadata)  
                except FileNotFoundError:
                    print(f"Metadata file not found at {metadata_loc}")
                    exit(1)
                except Exception as e:
                    print(f"Error Here {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Run to Yaml')
    parser.add_argument('--result_directory',type=str, 
                        help='Directory with run results pkl file',default='financial_phrasebank_Bohb_P2_25_5')
    parser.add_argument('--dataset',type=str, 
                        help='Dataset name',default='tweet_sentiment_multilingual')
    parser.add_argument('--metadata_loc',type=str, 
                        help='File where metadata is stored',default='cleaned_datasets')
    parser.add_argument('--save_location',type=str, 
                        help='Directory with run results pkl file',default='IncumbentConfigs')
    # store the result under IncumbentConfigs/dataset/file.yaml 
    args = parser.parse_args()
    # where all the run artifacts are kept
    # metadata_loc = os.path.join(os.getcwd(),args.metadata_loc, args.dataset, 'metadata.json')
    # try:
    #     with open(metadata_loc) as in_stream:
    #         metadata = yaml.safe_load(in_stream)    
    # except FileNotFoundError:
    #     print(f"Metadata file not found at {metadata_loc}")
    #     exit(1)
    working_dir = os.path.join(os.getcwd(),'../datasetruns')
    working_dir = os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/datasetruns')
    # create_yaml(working_dir=working_dir, dataset=args.dataset, result_dir=args.save_location, metadata=metadata)
    read_pkls(working_dir=working_dir,result_dir=args.save_location)



####################################################################################################
    # redo financial_phrasebank_Bohb_P2_25_5
    # mlsum is missing


   