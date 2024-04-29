
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config2Vector import ENCODE_MODELS
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

optimizer_columns=['Adam','AdamW''SGD','RAdam']
scheduler_columns=['linear_with_warmup', 'cosine_with_warmup', 
                   'cosine_with_hard_restarts_with_warmup', 
                   'polynomial_decay_with_warmup', 'constant_with_warmup']
                
model_columns=["bert-base-uncased",
               "bert-base-multilingual-cased",
               "deepset/bert-base-german-cased-oldvocab",
               "uklfr/gottbert-base",
               "dvm1983/TinyBERT_General_4L_312D_de",
               "linhd-postdata/alberti-bert-base-multilingual-cased",
               "dbmdz/distilbert-base-german-europeana-cased"]

# seed everything
def preprocess_data(training_set):
    """
    Given a training set, preprocess the data by converting the model, optimizer, and scheduler columns to one-hot encoding.
    Also drop the 'Dataset' and 'IncumbentOf' columns.
    """
    # remove the dataset and incumbent columns since its not metadata per se
    training_set = training_set.drop(columns=['Dataset', 'IncumbentOf'])
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
    X = training_set.drop(columns=['Performance', 'Rank']) # 27 features 
    y = training_set['Performance']
    return X, y


def  load_process_data(batch_size=32, seed=42):
    training_set = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
    X, y = preprocess_data(training_set)
    print(X.dtypes)
    # Preprocess the data
    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)  # 0.25 x 0.8 = 0.2

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def calculate_r_squared(outputs, labels):
    # Calculate R-squared
    mean_actual = torch.mean(labels)
    total_sum_of_squares = torch.sum((labels - mean_actual)**2)
    residual_sum_of_squares = torch.sum((outputs - labels)**2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared.item()

class CustomData():
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
        
        # self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders(self.select_cv_fold)
        

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

# trains model for n epochs
def training_loop(model, criterion, optimizer, train_loader, num_epochs=10):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss}")
    return model

def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    val_r_squared = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            val_loss += loss.item()
            val_r_squared += calculate_r_squared(outputs, labels)
        val_loss /= len(val_loader)
        val_r_squared /= len(val_loader)
    print(f"Validation Loss: {val_loss}, Validation R-squared: {val_r_squared}")
    return model

def test_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    test_r_squared = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            # print number times the outputs are close to the labels

            test_r_squared += calculate_r_squared(outputs, labels)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        test_r_squared /= len(test_loader)
    print(f"Test Loss---> {test_loss}, Test R-squared: {test_r_squared}")
    return model

def run_training(input_size, hidden_size, output_size, epochs=10, lr=0.001):
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    customdata=CustomData( batch_size=32,seed=42)
    train_loader, val_loader, test_loader = customdata.get_loaders()

    num_epochs = epochs
    num_folds = 5
    custom_data = CustomData(batch_size=32, seed=42)
    for fold in range(num_folds):
        train_loader, val_loader, test_loader = custom_data.get_loaders(cv_fold=fold + 1)
        model = training_loop(model, criterion, optimizer, train_loader, num_epochs)
        # Validation
        model=validate(model, criterion, val_loader)
        # Test
        model=test_model(model, criterion, test_loader)



## XGBoost
def create_dmatrix(loader):
    X, y = next(iter(loader))
    return xgb.DMatrix(X.numpy(), label=y.numpy())

def train_xgboost():
    training_set = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
    X, y = preprocess_data(training_set)
    X.columns=[col.replace('[', '').replace(']', '').replace('<', '') for col in X.columns]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',             # Evaluation metric
    'eta': 0.1,                        # Learning rate
    'max_depth': 6,                    # Maximum depth of tree
    'min_child_weight': 1,             # Minimum sum of instance weight needed in a child
    'subsample': 0.7,                  # Subsample ratio of the training instance
    'colsample_bytree': 0.7,           # Subsample ratio of columns when constructing each tree
    'seed': 42                         # Random seed
    }
    num_rounds = 100  # Number of boosting rounds
    eval_results = {}
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_rounds, evals=evals, early_stopping_rounds=10)
    # Evaluate the model on train, validation, and test sets
    preds = model.predict(dtest)
    y_label = dtest.get_label()
    mse = mean_squared_error(y_label, preds)
    print(f"Mean Squared Error on Test set: {mse}")


if __name__ == "__main__":
    input_size = 27 # number of features encoded + dataset
    hidden_size = 64
    output_size = 1 # performance
    run_training(input_size, hidden_size, output_size,epochs=50, lr=0.001)
    # train_xgboost()
