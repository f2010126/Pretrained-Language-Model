# Handle the data for the metamodel. Convert the data to a format that the metamodel can understand based on the loss function.
import pandas as pd
import numpy as np
import random
from regex import R
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




def preprocess_data(training_set):
    """
    Given a training set, preprocess the data by converting the model, optimizer, and scheduler columns to one-hot encoding.
    
    """
    # remove the dataset and incumbent columns since its not metadata per se
    # training_set = training_set.drop(columns=['Dataset', 'IncumbentOf']) comment out for now
    # convert the model, optimizer, and scheduler columns to one-hot encoding colmums   
    model_dummies = pd.get_dummies(training_set['model'], prefix='Model')
    model_dummies = model_dummies.astype(int)
    optimizer_dummies = pd.get_dummies(training_set['optimizer'], prefix='Optimizer')
    optimizer_dummies = optimizer_dummies.astype(int)
    scheduler_dummies = pd.get_dummies(training_set['scheduler'], prefix='Scheduler')
    scheduler_dummies = scheduler_dummies.astype(int)

    training_set = pd.concat([training_set.drop(columns=['model','optimizer','scheduler']), 
                                      model_dummies,optimizer_dummies,scheduler_dummies], axis=1)
    # expands to 29 columns with one-hot encoding   
    # change the model, optimizer, and scheduler columns to one-hot encoding
    # X = training_set #.drop(columns=['Performance', 'Rank']) # 27 features 
    # y = training_set['Performance']
    return training_set



class TrainMetaData(Dataset):

    # internal function to get the item for a regression task
    def __getitem__(self, index):
        if self.loss_func == "regression":
            return self.__get_regression_item__(index)
        elif self.loss_func == "bpr" or self.loss_func == "tml" or self.loss_func == "hingeloss":
            return self.__get_bpr_item__(index)
    
    def __len__(self):
        return len(self.y[self.set])

    # just return the regression item for the given train/test/Validation set
    def __get_regression_item__(self, index):
        x = self.x[self.set][index]
        y = self.y[self.set][index] 
        # ystar is the performance in a given dataset
        ybest  = self.y_best[self.set][index]

        return x, y, ybest # return the features, accuracy and best performance

    # return the BPR item for the given train/test/Validation set
    def __get_bpr_item__(self, index):
        x = self.x[self.set][index]
        y = self.y[self.set][index]
        r = self.ranks_flat[self.set][index] # flatten rank of that one

        try:
            larger_idx  = random.choice(self.larger_set[index]) # randomly select a larger index from the larger set
        except ValueError:
            larger_idx=index
        except IndexError:
            # the array is empty
            larger_idx = index
        except Exception as e:
            # other error
            print(f"Other Exception: {e}")
            larger_idx=index

        try:
            smaller_idx = random.choice(self.smaller_set[index]) # randomly select a smaller index from the smaller set
        except ValueError:
            smaller_idx = index
        except IndexError:
            # the array is empty
            smaller_idx = index
        except Exception as e:
            print(f"Other Exception: {e}")
            smaller_idx = index

        # get the features for the larger and smaller indices
        s = self.x[self.set][smaller_idx]
        r_s = self.ranks_flat[self.set][smaller_idx] # rank of the smaller index
        l = self.x[self.set][larger_idx]
        r_l = self.ranks_flat[self.set][larger_idx] # rank of the larger index  

        # returns features, accuracy and ranks of the item and the larger and smaller items
        return (x,s,l), (y, self.y[self.set][smaller_idx], self.y[self.set][larger_idx]), (r,r_s,r_l)
    

    def set_bpr_sampling(self, dataframe):
        # for the models in each dataset, list the ones with higher and lower accuracy
        self.larger_set = []
        self.smaller_set = []
        grouped = dataframe.groupby('Dataset')
        # Iterate over each group
        for _, group in grouped:
            for idx, row in group.iterrows():
                performance = row['Performance']
                lower_models = group[group['Performance'] < performance].index.tolist()
                higher_models = group[group['Performance'] > performance].index.tolist()
        
                self.smaller_set.append(lower_models)
                self.larger_set.append(higher_models)

    

    def split_dataset(self):
        # given the entire training data, split it into training, validation and test sets
        # drop IncumbentOf and Dataset columns

        # add a new column that adds the value of the best accuracy for a given dataset
        self.full_train_data["best_value"] = self.full_train_data.groupby("Dataset")["Performance"].transform("max")
        # add flattened rank per dataset
        self.full_train_data['flat_rank'] = self.full_train_data.groupby('Dataset')['Rank'].transform(lambda x: x / x.max())


        cv_datasets=self.cv_folds[self.cv_folds['cv_fold'] == self.cv_fold_no]['Dataset']
        # based on the datasets used for CV, split the data into training and validation sets
        validation_data = self.full_train_data[self.full_train_data['Dataset'].isin(cv_datasets)]
        training_data = self.full_train_data[~self.full_train_data['Dataset'].isin(cv_datasets)]

 

        # shuffle the training data and split into 2 dataframes 10% and remaining 90%
        shuffle_data = training_data.sample(frac=1).reset_index(drop=True)
        training_data=shuffle_data.sample(frac=0.9, random_state=self.seed)
        test_data=shuffle_data.drop(training_data.index).reset_index(drop=True)

        # get the shuffled set. Reset the index
        training_data.reset_index(drop=True, inplace=True)
        # do the pairwise sampling for the BPR loss function
        if self.loss_func=='bpr' or self.loss_func=='tml' or self.loss_func=='hingeloss':
            self.set_bpr_sampling(training_data)

        # drop the dataset and incumbent columns
        self.datasetnames = {'train':training_data['Dataset'],
                             'valid':validation_data['Dataset'],
                             'test':test_data['Dataset']}
        self.pipelines = {'train':training_data['IncumbentOf'],
                            'valid':validation_data['IncumbentOf'],
                            'test':test_data['IncumbentOf']}
        validation_data = validation_data.drop(columns=['Dataset', 'IncumbentOf'])
        training_data = training_data.drop(columns=['Dataset', 'IncumbentOf'])
        test_data = test_data.drop(columns=['Dataset', 'IncumbentOf'])


        # Drop performance and rank columns after assinging to y_train and rank_train
        y_train = training_data['Performance']
        X_train = training_data.drop(columns=['Performance', 'Rank','best_value','flat_rank']) 

        y_valid = validation_data['Performance']
        X_valid = validation_data.drop(columns=['Performance', 'Rank','best_value','flat_rank'])

        y_test = test_data['Performance']
        X_test = test_data.drop(columns=['Performance', 'Rank','best_value','flat_rank'])

        # set target values for the regression task

        self.values = {"train":training_data['Performance'].values,
                        "valid":validation_data['Performance'].values,
                        "test":test_data['Performance'].values}
        
        self.y_best = {"train":training_data['best_value'].values,
                       "valid":validation_data['best_value'].values,
                       "test":test_data['best_value'].values}
        
        self.ranks ={ "train":training_data['Rank'].values,
                      "valid":validation_data['Rank'].values,
                      "test":test_data['Rank'].values}

        self.ranks_flat = {"train":training_data['flat_rank'],
                            "valid":validation_data['flat_rank'],
                            "test":test_data['flat_rank']}  

        return X_train, X_valid, y_train, y_valid, X_test, y_test

    def initialize(self, loss_func="regression"):
        # seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        # load the data and cv folds
        self.cv_folds = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/cv_folds.csv')
        dataframe = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
        
        self.full_train_data = preprocess_data(dataframe)

        # sets up the X and y for the dataset as per the mode (currently just regression and CV)
        X_train, X_valid, y_train, y_valid, X_test, y_test  = self.split_dataset()

        # NOW scale, standardize the data and prep the Binary Pairwise Ranking (BPR) data

        
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_valid.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
        self.x = {'train': X_train_tensor, 'valid': X_val_tensor, 'test': X_test_tensor}
        self.y = {'train': y_train_tensor, 'valid': y_val_tensor, 'test': y_test_tensor}
        


class TrainingDataCV(TrainMetaData):
        def __init__(self, seed, batch_size=32, fold_no=1, loss_func="regression"):
            super(TrainMetaData, self).__init__()
            self.seed=seed
            self.batch_size = batch_size
            self.cv_fold_no=fold_no
            self.loss_func=loss_func
            self.set = 'train'
            self.initialize(loss_func=loss_func)
        


class OldTrainingDataCV():
    def __init__(self, batch_size=32,seed=42):
        """
        CustomData constructor.

        Parameters:
        - cv fold: split of the data into training and validation sets.
        - dataframe (DataFrame): DataFrame containing the data.

        Returns data loader for training, test and validation sets.
        """
        #training_set = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
        # load folds into csv
        self.cv_folds = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/cv_folds.csv')
        self.dataframe = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
        self.batch_size = batch_size
        self.seed=seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_data_loaders(self, cv_fold=1):
        # based on the cv_fold, create the training and validation data loaders.
        # all datasets with the given fold number are in the validation set, the rest are in the training set
        cv_datasets=self.cv_folds[self.cv_folds['cv_fold'] == cv_fold]['Dataset']
        validation_data = self.dataframe[self.dataframe['Dataset'].isin(cv_datasets)]
        training_data = self.dataframe[~self.dataframe['Dataset'].isin(cv_datasets)]
        # shuffle the training data and split into 2 dataframes 10% and remaining 90%
        shuffle_data = training_data.sample(frac=1).reset_index(drop=True)
        training_data=shuffle_data.sample(frac=0.9, random_state=self.seed)
        test_data=shuffle_data.drop(training_data.index).reset_index(drop=True)

        X_val, y_val = preprocess_data(validation_data)
        X_train, y_train = preprocess_data(training_data)
        X_test, y_test = preprocess_data(test_data)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader
    
    def get_loaders(self, cv_fold=1):
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders(cv_fold)
        return self.train_loader, self.val_loader, self.test_loader


class TestData():
    def __init__(self, batch_size=32,seed=42):
        self.batch_size = batch_size
        self.seed=seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def create_data(self):
        test_data=pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/test_germeval2018.csv')
        # add a fake performance column with zeros
        
        test_data['Performance'] = 0
        test_data['Rank'] = 0
        X_test, y_test = preprocess_data(test_data)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor)


def  get_data_loader(batch_size, cv_fold, seed,loss_func="regression"):
    # get the data and create the data loaders
    training_data = TrainingDataCV(batch_size=batch_size, fold_no=cv_fold, seed=seed, loss_func=loss_func)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  
    return train_loader
    
if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser("Data Creation")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--cv_fold', type=int, default=5, help='cv fold')
    parser.add_argument('--loss_func', type=str, default='hingeloss', help='loss function can be regression|bpr|hingeloss')
    args = parser.parse_args()
    
    datsetloader = get_data_loader(batch_size=args.batch_size, cv_fold=args.cv_fold, seed=args.seed, loss_func=args.loss_func)