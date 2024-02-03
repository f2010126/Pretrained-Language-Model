import os

from data_modules import get_datamodule
import datasets

def get_augmented_df():
    # call the lib, get the augmented df
    df_augmented = None
    return df_augmented

def augment_labels(data_folder,data_dir,num_labels=50):
    """ 
    Augment the labels of a dataset by adding a number of labels to the dataset
    :param data_folder: the name of the dataset folder
    :param data_dir: the path to the dataset folder
    :param num_labels: max number of labels to keep in the dataset, default max is 50 min is 3
    """
    dataset=datasets.load_from_disk(os.path.join(data_dir, data_folder))
    # for each split remove all rows with label greater than 3
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda example: example['label'] < 4)
        # for each dataset split get the augmented dataframe
        df_augmented = get_augmented_df()
        # replace the original dataframe with the augmented dataframe
        dataset[split] = df_augmented
    
    # save the augmented dataset
    dataset.save_to_disk(os.path.join(data_dir, data_folder+f"_augmented_{num_labels}"))


if __name__ == "__main__":
    dataset_list = ['Bundestag-v2', 'tagesschau', 'german_argument_mining', 
                    'mlsum', 'hatecheck-german', 'financial_phrasebank_75agree_german', 
                    'x_stance', 'swiss_judgment_prediction', 'miam']
    data_folder = "Bundestag-v2"
    data_dir = os.path.join(os.getcwd(),"cleaned_data")
    augment_labels(data_folder,data_dir)