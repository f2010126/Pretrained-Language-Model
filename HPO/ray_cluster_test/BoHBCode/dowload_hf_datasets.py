# download a list of datasets from Hugging Face to a given location
import os
from re import M
import uuid
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# Swiss judegment Test at https://huggingface.co/datasets/rcds/occlusion_swiss_judgment_prediction


# Download a list of datasets from Hugging Face to a given location
# parse with models and max_seq_length

def preprocess_dataset():
    models=["bert-base-uncased", "bert-base-multilingual-cased","deepset/bert-base-german-cased-oldvocab",
            "uklfr/gottbert-base","dvm1983/TinyBERT_General_4L_312D_de",
            "linhd-postdata/alberti-bert-base-multilingual-cased",
            "dbmdz/distilbert-base-german-europeana-cased"]
    
    max_seq_length=[128, 256, 512]
    raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
    data_folder=[dataset_name.split("/")[-1] for dataset_name in dataset_list]
     # All datasets here have colums text, label. Label is a number

def print_dataset_details(dataset_name):
    raw_data_path=os.path.join(os.getcwd(), "raw_datasets")
    data_folder=dataset_name.split("/")[-1]
    dataset=load_from_disk(dataset_name, data_dir=os.path.join(raw_data_path, data_folder))
    print(dataset) 

    pass

def download_raw_datasets(dataset_list):
    dowloadpath=os.path.join(os.getcwd(), "raw_datasets")
    # check for download path
    if not os.path.exists(dowloadpath):
        os.makedirs(dowloadpath)
    for dataset_name in dataset_list:
        print("Downloading dataset: ", dataset_name)
        try:
            dataset=load_dataset(dataset_name)
        except Exception as e:
            print("Error downloading dataset: ", dataset_name)
            print(e)
            continue
        
        dataset_folder=dataset_name.split("/")[-1] # last part of the url    
        dataset.save_to_disk(os.path.join(dowloadpath, dataset_folder))
        print("Done downloading dataset: ", dataset_name)


if __name__ == "__main__":
    dataset_list=[ "gnad10",  "amazon_reviews_multi", 
                  'cardiffnlp/tweet_sentiment_multilingual', 'miam',"tyqiangz/multilingual-sentiments",
                  "senti_lex", "mteb/mtop_domain", "omp", 
                  
                  'rcds/swiss_judgment_prediction','x_stance', 'Paul/hatecheck-german',
                   'senti_ws','mlsum', 'tillschwoerer/tagesschau' # news articles,
                   'threite/Bundestag-v2' # who said what in the German Bundestag, 
                   'joelniklaus/german_argument_mining', # argument mining
                   'scherrmann/financial_phrasebank_75agree_german', # financial phrasebank'
                   'Brand24/mms', # might be teh amazon sets
                    'akash418/germeval_2017',
                    'gwlms/germeval2018']
    download_raw_datasets(['gwlms/germeval2018'])

    for dataset_name in dataset_list:
        print("Dataset: ", dataset_name)
        try:
            print_dataset_details(dataset_name)
        except Exception as e:
            print("Error show dataset: ", dataset_name)
            continue
    
    print("Done")

    